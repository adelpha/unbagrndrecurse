//! Preprocessing, inference and postprocessing for background removal.
//!
//! Everything here runs entirely on-device via the `ort` ONNX Runtime
//! bindings. No image, filename or metadata this module touches ever
//! leaves the machine.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use image::{imageops::FilterType, ImageBuffer, Luma, Rgba, RgbaImage};
use ndarray::Array4;
use ort::ep;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use crate::models::{ModelSpec, OutputMode};

/// A pool of lazily-built inference sessions, per model. A `Session` isn't
/// safe to `run()` from several threads at once, so each inference call
/// takes a session out of the pool for its own exclusive use (building one
/// if none is free) and hands it back afterwards. That lets a batch run
/// several images side by side, each on its own session with its share of
/// the CPU cores, while a lone call gets one session using every core.
pub struct InferenceState {
    pool: Mutex<HashMap<String, Vec<PooledSession>>>,
    /// Run on the GPU (DirectML, Windows only) when possible.
    use_gpu: AtomicBool,
    /// Set once the GPU has failed (out of video memory, "device removed"
    /// after a driver reset...). Everything then runs on the CPU until the
    /// next [`InferenceState::set_use_gpu`] call, instead of every image
    /// failing on the GPU first.
    gpu_failed: AtomicBool,
    /// Held for the whole of a GPU inference. A big model can fill a whole
    /// card on its own (BiRefNet at 1024px takes ~6.5 GB on DirectML), and
    /// two at once overflow it into system RAM or crash the GPU driver, so
    /// GPU work is one image at a time. Parallel batch workers still
    /// overlap decoding and saving with it.
    gpu_gate: Mutex<()>,
}

struct PooledSession {
    /// Built with the GPU execution provider (see [`InferenceState::set_use_gpu`]).
    gpu: bool,
    /// Intra-op threads the session was built with. A session built for a
    /// different level of parallelism is dropped rather than reused, so
    /// N side-by-side sessions never oversubscribe the cores.
    threads: usize,
    session: Session,
}

impl Default for InferenceState {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceState {
    pub fn new() -> Self {
        Self {
            pool: Mutex::new(HashMap::new()),
            use_gpu: AtomicBool::new(true),
            gpu_failed: AtomicBool::new(false),
            gpu_gate: Mutex::new(()),
        }
    }

    /// Whether new inference sessions should run on the GPU. Only has an
    /// effect on Windows (DirectML); elsewhere everything runs on the CPU.
    /// Sessions already built for the other device are dropped as they're
    /// next picked up, so a change takes effect on the next image.
    /// Also clears an earlier GPU failure, so each new run tries the GPU
    /// again.
    pub fn set_use_gpu(&self, use_gpu: bool) {
        self.use_gpu.store(use_gpu, Ordering::SeqCst);
        self.gpu_failed.store(false, Ordering::SeqCst);
    }

    fn wants_gpu(&self) -> bool {
        cfg!(windows)
            && self.use_gpu.load(Ordering::SeqCst)
            && !self.gpu_failed.load(Ordering::SeqCst)
    }

    /// Gives up on the GPU for this run and frees every GPU session (after
    /// a device reset they're unusable anyway).
    fn mark_gpu_failed(&self) {
        self.gpu_failed.store(true, Ordering::SeqCst);
        if let Ok(mut pool) = self.pool.lock() {
            for free in pool.values_mut() {
                free.retain(|s| !s.gpu);
            }
        }
    }

    /// Runs `spec` on `original` with a pooled session: on the GPU (one at
    /// a time) when enabled and working, otherwise on the CPU. A GPU
    /// failure, whether loading the model or running it, switches to the
    /// CPU for this image and the rest of the run.
    fn infer(
        &self,
        spec: &ModelSpec,
        model_path: &Path,
        original: &image::DynamicImage,
        parallel: usize,
    ) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, String> {
        let threads = threads_per_session(parallel);
        let mut gpu_error = None;
        if self.wants_gpu() {
            let _gate = self.gpu_gate.lock().unwrap_or_else(|p| p.into_inner());
            let outcome = self
                .take(spec.key, model_path, threads, true)
                .and_then(|mut pooled| {
                    let mask = run_inference(&mut pooled.session, spec, original)?;
                    self.give_back(spec.key, pooled, 1);
                    Ok(mask)
                });
            match outcome {
                Ok(mask) => return Ok(mask),
                Err(e) => {
                    eprintln!("unbagrnd: GPU inference failed, using the CPU from now on: {e}");
                    self.mark_gpu_failed();
                    gpu_error = Some(e);
                }
            }
        }

        let mut pooled = self.take(spec.key, model_path, threads, false)?;
        match run_inference(&mut pooled.session, spec, original) {
            Ok(mask) => {
                self.give_back(spec.key, pooled, parallel);
                Ok(mask)
            }
            // A failed session isn't handed back, so whatever memory it was
            // holding is released and the next image starts from a clean
            // slate instead of failing the same way again.
            Err(e) => Err(friendly_inference_error(match gpu_error {
                Some(gpu) => format!("GPU: {gpu}; CPU: {e}"),
                None => e,
            })),
        }
    }

    /// A free session for `key` built with `threads` intra-op threads, or a
    /// newly built one if there's none. Built outside the lock so other
    /// workers aren't held up while a model loads.
    fn take(
        &self,
        key: &str,
        model_path: &Path,
        threads: usize,
        gpu: bool,
    ) -> Result<PooledSession, String> {
        {
            let mut pool = self
                .pool
                .lock()
                .map_err(|_| "inference session lock was poisoned".to_string())?;
            if let Some(free) = pool.get_mut(key) {
                // Sessions built for another parallelism level are dropped
                // here, freeing their memory.
                while let Some(candidate) = free.pop() {
                    if candidate.threads == threads && candidate.gpu == gpu {
                        return Ok(candidate);
                    }
                }
            }
        }
        Ok(PooledSession { gpu, threads, session: build_session(model_path, threads, gpu)? })
    }

    /// Returns a session to the pool, keeping at most `keep` free sessions
    /// per model so memory shrinks back after a wider batch.
    fn give_back(&self, key: &str, session: PooledSession, keep: usize) {
        if let Ok(mut pool) = self.pool.lock() {
            let free = pool.entry(key.to_string()).or_default();
            if free.len() < keep.max(1) {
                free.push(session);
            }
        }
    }
}

/// Intra-op threads for each of `parallel` sessions running side by side:
/// the machine's cores split evenly between them.
fn threads_per_session(parallel: usize) -> usize {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (cores / parallel.max(1)).max(1)
}

fn ensure_ort_environment() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        // `commit()` is safe to call at most once per process; later calls
        // are simply ignored, so guarding with `Once` just avoids a wasted
        // build of the execution-provider list.
        ort::init().with_name("unbagrnd").commit();
    });
}

fn build_session(model_path: &Path, threads: usize, gpu: bool) -> Result<Session, String> {
    ensure_ort_environment();
    let cpu = ep::CPU::default().with_arena_allocator(false).build();
    // DirectML first, CPU as the fallback for any operator it can't run. If
    // DirectML can't be registered at all (no DX12 GPU, driver trouble) ORT
    // skips it and the session simply runs on the CPU.
    #[cfg(windows)]
    let providers = if gpu {
        vec![
            ep::DirectML::default()
                .with_performance_preference(ep::directml::PerformancePreference::HighPerformance)
                .build(),
            cpu,
        ]
    } else {
        vec![cpu]
    };
    #[cfg(not(windows))]
    let providers = {
        let _ = gpu;
        vec![cpu]
    };

    // The CPU arena allocator and the memory-pattern planner both reserve
    // large contiguous blocks up front and never hand them back. For the
    // 1024px models (BiRefNet, IS-Net) that means single 800MB+ requests
    // that fail with "BFCArena::AllocateRawInternal Failed to allocate
    // memory" on machines with little free RAM. Turning both off makes ORT
    // allocate each tensor on demand and free it as soon as it's dead, so
    // peak usage is the real working set instead of an over-sized arena.
    Session::builder()
        .map_err(|e| format!("failed to initialize the inference session: {e}"))?
        .with_execution_providers(providers)
        .map_err(|e| format!("failed to configure the inference session: {e}"))?
        .with_memory_pattern(false)
        .map_err(|e| format!("failed to configure the inference session: {e}"))?
        // On DirectML, ORT's graph optimisations (Level1 as well as Level3)
        // made BiRefNet fill an 8 GB card, spill ~16 GB into system RAM and
        // reset the GPU driver (887A0005) on the second image. With them
        // off it stayed stable, used less RAM and ran ~3x faster than the
        // CPU. The CPU path keeps full optimisation.
        .with_optimization_level(if gpu {
            GraphOptimizationLevel::Disable
        } else {
            GraphOptimizationLevel::Level3
        })
        .map_err(|e| format!("failed to configure the inference session: {e}"))?
        .with_intra_threads(threads)
        .map_err(|e| format!("failed to configure the inference session: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("failed to load the model: {e}"))
}

/// Removes the background from an already-decoded image using the given
/// model, returning the result as an RGBA image with the cut-out subject on
/// a transparent background, at the original image's resolution.
pub fn remove_background(
    state: &InferenceState,
    spec: &ModelSpec,
    model_path: &Path,
    original: &image::DynamicImage,
) -> Result<RgbaImage, String> {
    remove_background_parallel(state, spec, model_path, original, 1)
}

/// [`remove_background`] for a caller running `parallel` of these at once
/// (a batch): each call then uses its share of the CPU cores, and up to
/// `parallel` sessions of the model are kept ready between calls.
pub fn remove_background_parallel(
    state: &InferenceState,
    spec: &ModelSpec,
    model_path: &Path,
    original: &image::DynamicImage,
    parallel: usize,
) -> Result<RgbaImage, String> {
    let (orig_w, orig_h) = (original.width(), original.height());
    if orig_w == 0 || orig_h == 0 {
        return Err("image has zero width or height".to_string());
    }

    let mask_native = state.infer(spec, model_path, original, parallel)?;

    let mask_full = image::imageops::resize(&mask_native, orig_w, orig_h, FilterType::Lanczos3);

    let original_rgba = original.to_rgba8();
    let mut out: RgbaImage = ImageBuffer::new(orig_w, orig_h);
    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let src = original_rgba.get_pixel(x, y);
        let alpha = mask_full.get_pixel(x, y).0[0];
        *pixel = Rgba([src.0[0], src.0[1], src.0[2], alpha]);
    }

    Ok(out)
}

/// Adds an actionable hint to ONNX Runtime's raw out-of-memory failure.
fn friendly_inference_error(message: String) -> String {
    if message.contains("Failed to allocate memory") {
        format!(
            "not enough free memory for this model. Close other apps or pick a \
             lighter model (e.g. Silueta). Details: {message}"
        )
    } else {
        message
    }
}

/// Runs one forward pass and returns the predicted alpha mask at the
/// model's native square input resolution.
fn run_inference(
    session: &mut Session,
    spec: &ModelSpec,
    original: &image::DynamicImage,
) -> Result<ImageBuffer<Luma<u8>, Vec<u8>>, String> {
    let size = spec.input_size;
    let resized = original.resize_exact(size, size, FilterType::Lanczos3);
    let rgb = resized.to_rgb8();

    // Matches rembg's `BaseSession.normalize` exactly: every channel is
    // divided by the single brightest pixel value in the resized image
    // (not a fixed 255), then centered per-channel around this model's
    // mean/std. This is the input distribution each model was trained on.
    let max_val = rgb
        .pixels()
        .flat_map(|p| p.0)
        .map(|c| c as f32)
        .fold(1e-6_f32, f32::max);

    let mut input = Array4::<f32>::zeros((1, 3, size as usize, size as usize));
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let [r, g, b] = pixel.0;
        let (x, y) = (x as usize, y as usize);
        input[[0, 0, y, x]] = (r as f32 / max_val - spec.mean[0]) / spec.std[0];
        input[[0, 1, y, x]] = (g as f32 / max_val - spec.mean[1]) / spec.std[1];
        input[[0, 2, y, x]] = (b as f32 / max_val - spec.mean[2]) / spec.std[2];
    }

    let input_tensor =
        TensorRef::from_array_view(&input).map_err(|e| format!("failed to prepare input: {e}"))?;
    let outputs = session
        .run(ort::inputs![input_tensor])
        .map_err(|e| format!("inference failed: {e}"))?;

    let mask = outputs[0]
        .try_extract_array::<f32>()
        .map_err(|e| format!("failed to read model output: {e}"))?;

    match spec.output_mode {
        OutputMode::Alpha => {
            let values: Vec<f32> = if spec.apply_sigmoid {
                mask.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect()
            } else {
                mask.iter().copied().collect()
            };

            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &v in &values {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                }
            }
            let range = (max - min).max(1e-6);

            let mask_bytes: Vec<u8> = values
                .iter()
                .map(|&v| (((v - min) / range) * 255.0).round().clamp(0.0, 255.0) as u8)
                .collect();

            ImageBuffer::from_raw(size, size, mask_bytes)
                .ok_or_else(|| "unexpected model output size".to_string())
        }
        OutputMode::ClothSegAnyForeground => {
            // Output shape is (1, num_classes, H, W): a per-pixel class
            // score (background / upper / lower / full-body garment).
            // Anything the model doesn't classify as background is kept.
            let arr = mask
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| format!("unexpected model output shape: {e}"))?;
            let (_, num_classes, h, w) = arr.dim();

            let mut mask_bytes = vec![0u8; h * w];
            for y in 0..h {
                for x in 0..w {
                    let mut best_class = 0usize;
                    let mut best_val = f32::NEG_INFINITY;
                    for c in 0..num_classes {
                        let v = arr[[0, c, y, x]];
                        if v > best_val {
                            best_val = v;
                            best_class = c;
                        }
                    }
                    mask_bytes[y * w + x] = if best_class != 0 { 255 } else { 0 };
                }
            }

            ImageBuffer::from_raw(w as u32, h as u32, mask_bytes)
                .ok_or_else(|| "unexpected model output size".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models;

    /// Exercises the full decode -> preprocess -> inference -> postprocess
    /// pipeline against a real cached model file. Skipped unless
    /// `UNBAGRND_TEST_MODEL_KEY`, `UNBAGRND_TEST_MODEL_PATH` and
    /// `UNBAGRND_TEST_IMAGE_PATH` are set, so `cargo test` works offline in
    /// CI without a multi-hundred-MB model download.
    ///
    /// Run locally with:
    /// ```sh
    /// UNBAGRND_TEST_MODEL_KEY=silueta \
    /// UNBAGRND_TEST_MODEL_PATH=/path/to/silueta.onnx \
    /// UNBAGRND_TEST_IMAGE_PATH=/path/to/a/photo.jpg \
    /// cargo test --release removes_background_from_a_real_photo -- --nocapture
    /// ```
    #[test]
    fn removes_background_from_a_real_photo() {
        let (Ok(model_key), Ok(model_path), Ok(image_path)) = (
            std::env::var("UNBAGRND_TEST_MODEL_KEY"),
            std::env::var("UNBAGRND_TEST_MODEL_PATH"),
            std::env::var("UNBAGRND_TEST_IMAGE_PATH"),
        ) else {
            eprintln!(
                "skipping: set UNBAGRND_TEST_MODEL_KEY, UNBAGRND_TEST_MODEL_PATH and \
                 UNBAGRND_TEST_IMAGE_PATH to run this test"
            );
            return;
        };
        let spec = models::find_model(&model_key).expect("unknown UNBAGRND_TEST_MODEL_KEY");

        let original = image::open(&image_path).expect("test image should decode");
        let (orig_w, orig_h) = (original.width(), original.height());

        let state = InferenceState::new();
        let result = remove_background(&state, spec, Path::new(&model_path), &original)
            .expect("background removal should succeed");

        assert_eq!(result.width(), orig_w);
        assert_eq!(result.height(), orig_h);

        // A real photo should produce a mask with genuine variation, not a
        // flat "everything transparent" or "everything opaque" output.
        let alphas: Vec<u8> = result.pixels().map(|p| p.0[3]).collect();
        let min = *alphas.iter().min().unwrap();
        let max = *alphas.iter().max().unwrap();
        assert!(
            max > min + 32,
            "expected the alpha mask to show real subject/background contrast, got min={min} max={max}"
        );

        // Sanity-check the RGB channels of visible pixels still match the
        // source photo (we must never alter color, only alpha).
        let original_rgba = original.to_rgba8();
        for (x, y, pixel) in result.enumerate_pixels() {
            let src = original_rgba.get_pixel(x, y);
            assert_eq!(pixel.0[0..3], src.0[0..3]);
        }
    }
}

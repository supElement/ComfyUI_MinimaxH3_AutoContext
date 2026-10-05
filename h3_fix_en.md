# MinimaxH3 Face Fix Pipeline Guide (Face_Cut → Face_Resample → Face_Blend)

### Preserving Facial Identity

**Method 1: Reference Images (Recommended — refine while keeping identity)**

Pass reference images through the `ref_images` port and declare them in the prompt with Picture tags:

- Tags must be written in the prompt of **the segment where that character appears** — Resample's prompts take effect per segment; whichever segment a tag is written into, the reference image is only passed to the blocks mapped to that segment;
- For multi-character videos, the prompt must describe which character in the reference image corresponds to which role (for model disambiguation); single-character videos can omit this;
- Tip: you can directly pick the sharpest frame of that character's face from the source video (from Face_Cut's crop output) as the reference image — identity consistency is most stable this way.

**Method 2: Lower the Resampling Strength (zero extra setup, trades fix quality for identity retention)**

The denoise strength is determined by the **starting value** of the σ schedule (img2img noise added = starting σ × noise); lowering the starting σ makes the refinement more conservative:

- Use the SplitSigmas node: `low_sigmas` output → Face_Resample's `sigmas` port (the low side ends with 0, naturally satisfying the port requirement), and **increase SplitSigmas' step**;
- The step must be **smaller than** the total sampling steps, and leave at least a few steps on the low side — as step approaches the total step count, low_sigmas shrinks to just the final 0 and refinement degenerates completely (in the extreme case it's nearly a pass-through);
- On newer ComfyUI versions you can use the SplitSigmasDenoise node instead, which splits directly by denoise value without step/σ conversion.

**How they relate**: complementary, not either/or. Lowering the starting σ preserves the identity of the **original crop** (flaws and blemishes come along too); reference images preserve the identity of the **reference** (refines as usual while pulling the face toward the reference). Recommended: reference images as the foundation + a moderately lowered starting σ as a safety net.

> Note: changes to the σ schedule automatically invalidate the Resample cache — no need to clear_cache manually.
>
> Cache disk usage: a directory-level cap has been added, 32 GB by default; the oldest files are cleaned automatically when exceeded (adjustable via the `H3_CACHE_MAX_GB` environment variable; set to 0 to disable).

Three nodes chained together perform face repair on the output of the main sampler node (Minimax_H3_AutoContext_Sampler):

| Node | Step | Role |
|---|---|---|
| Minimax_H3_Face_Cut | ① Detection & Cropping | Shot detection + YOLO detection + optional SeC-4B tracking, crops into uniform res² tiles with per-frame smoothed windows |
| Minimax_H3_Face_Resample | ② Refinement | Block-level img2img resampling with the same models as the main sampler (block structure mirrors the main sampler's segmentation + inter-block anchoring) |
| Minimax_H3_Face_Blend | ③ Paste-back | Pastes the refined faces back onto the original frames pixel-by-pixel following the geometry ledger and masks (zero VAE) |

## 1. Standard Wiring

| From | To |
|---|---|
| Main sampler info | Face_Cut.info and Face_Resample.info |
| Main sampler output frames | Face_Cut.images (images mode recommended) and Face_Blend.images |
| Face_Cut.crop_images | Face_Resample.crop_images |
| Face_Cut.face_pack | Face_Resample.face_pack |
| Face_Cut.shot_info | parameter node (assigns shot prompts) |
| Face_Resample.images | Face_Blend.canvas |
| Face_Resample.bbox | Face_Blend.bbox |
| Face_Cut.masks | Face_Blend.masks (**direct connection**, bypassing Resample) |
| Face_Blend.images | Final output frames |

## 2. Face_Cut — Detection & Cropping

Pipeline: PySceneDetect shot detection → YOLO detection → optional SeC-4B identity tracking → per-frame smoothed-window cropping. Multi-person scenes: identity arbitration compares per-pixel masks (boxes only as fallback); detections inside tracking gaps are first attributed back to existing tracks before considering new identities — crossing/swap and brief mutual occlusion no longer shatter the same identity into pieces; residual small gaps are bridged by in-track interpolation (default cap 24 frames).

Model download: [huggingface](https://huggingface.co/cglearned/Minimax_H3_Face_Cut/tree/main)

### Model Directories

| Model | Parameter | Directory | Notes |
|---|---|---|---|
| YOLO face detection | face_model | ComfyUI/models/elementEasy/ | .pt/.pth/.onnx/.engine/.torchscript; errors out if none selected |
| SeC-4B identity tracking | sec_model | ComfyUI/models/sams/ | fp16 recommended; None = single-face mode |
| Upscale model | upscale_model | ComfyUI/models/upscale_models/ | Image super-resolution only (ESRGAN/RealESRGAN/UltraSharp type); avoid GFPGAN/CodeFormer face-restoration models |

After placing models into the directories, **refresh or restart ComfyUI** for the new files to appear in the dropdowns.

PySceneDetect is a Python dependency, not a model: if missing, `pip install scenedetect`; it automatically degrades to upstream-segment isolation only (warns instead of erroring).

### Key Parameters

| Parameter | Description |
|---|---|
| images / latent | Choose one. images mode recommended: detection at the external frames' native size, latent completely untouched; latent mode decodes probe frames via VAE |
| yolo_threshold | Detection confidence (default 0.3); lower it if detections are missed |
| shot_threshold | Shot detection threshold; higher = fewer cuts (default 40) |
| upscale_model | Optional SR chain: repeatedly upscales until ≥ canvas side (≤2 passes) then bicubics to the exact size, fixing ringing artifacts and color blotches on heavily upscaled small faces; None = bicubic only |
| pre_blur | Applies a slight Gaussian blur to the crop window before upscaling, softening source video noise and interpolation jaggies; for noisy footage try 0.5–1.5; the Resample output may become softer or even blurry |
| res / expand | Canvas side length (default 512) / crop window margin % (default 20) |
| skip_ratio | Face ≥ res × this ratio → skip resampling (default 0.8) |
| sec_model / sec_threshold / max_identities | None = single face (largest face per frame, no mask); selecting weights = multi-identity tracking + SeC masks; SeC-4B always unloads automatically after tracking finishes |
| sr_batch | Frames per SR forward pass in the crop upscale stage (default 4, 1–16). Higher = faster but higher VRAM peak; 4 on 16GB, 8–16 on 24GB+. Affects only speed and VRAM, not results |
| unload_main_models | After detection & SeC tracking and before crop/SR upscaling, moves the H3 main model/VAE/CLIP out of VRAM (default on; recommended on 12–16GB cards). SR models go straight into VRAM via spandrel and bypass ComfyUI's model management — a resident main model pushes VRAM past the physical limit; Windows masks the OOM as "shared GPU memory" overflow, which shows up as a sudden speed drop. Regardless of this switch, SeC-4B always unloads after tracking finishes |

## 3. Face_Resample — Canvas Refinement

Crop rows serve as the canvas; block-level img2img refinement; the block structure mirrors the main sampler's segmentation with inter-block continuation anchoring.

**Two wiring modes**: Integrated (info connected to the main sampler; model/vae/clip come from info.h3_runtime, local ports ignored) / Standalone (info left empty; local model/vae/clip required; prompts and segmentation come from parameter). The parameter port is required in both modes.

| Parameter | Description |
|---|---|
| sigmas | Refinement σ schedule (from a scheduler output); fewer steps = a more conservative fix |
| seed | Block sampling seed (auto-offset per block); keep the widget's control_after_generate set to fixed, otherwise row caches never hit |
| face_prompt | Face-repair prompt dedicated to describing the facial features/skin look you want. The video's segment prompts are written for whole-shot generation and can actively mislead the face at high σ — use this port to control the repair direction on its own. Effect is limited in some scenarios |
| prompt_mode | How the repair prompt combines with the original segment prompt: prepend: repair prompt + original prompt (keeps scene and reference declarations); replace_text: repair prompt + reference declarations only (first choice for high-σ structural repair); replace_all: repair prompt only. To use this node's ref_image, a declaration reference is required, i.e. `<Picture N>`, otherwise the reference won't take effect |
| identity_ref | Automatic identity anchor: automatically picks the sharpest frame of that person and injects it as a reference into the resampling, keeping identity and appearance consistent across frames/blocks at high σ (the key anti-jitter switch) , For each ID, the clearest facial frame (with the largest visible area) is extracted to serve as the reference image, defaulting to <Picture 1>. If reference images are connected and declared, the reference index for the automatically anchored frame shifts to "number of reference images + 1" (e.g., if two reference images are connected, the auto-anchored frame serves as <Picture 3>). If this negatively impacts the generated results, simply disable the `identity_ref` parameter.|
| color_match | Per-subtrack Reinhard color match back to the source crops (default on) |
| ref_images | Reference images; passed only when declared with a Picture tag in the prompt |

Block-level principle: subtracks of the same identity and contiguous in time are merged into one sequence → split into blocks along the main sampler's real segment boundaries (prompts map to segments precisely by block midpoint) → each block's encoding is padded to the 17n+5 grid (last frame repeated, trimmed after decode) → at boundaries with large window jumps, anchoring is anchored or disabled depending on the tracking mode.

## 4. Face_Blend — Paste-back Compositing

Per-subtrack scaling + pixel paste-back, **zero VAE** (entirely in pixel space, no encode/decode quality loss).

| Port / Parameter | Description |
|---|---|
| images | Original video frames |
| canvas | Refined canvas from Face_Resample |
| bbox | Face_Resample's face_pack. ⚠ Do not connect shot_info (a shot map is not a geometry pack); Face_Cut's face_pack also works (automatically falls back to crop_off with a warning, but Resample's is recommended) |
| masks (optional) | Face_Cut's SeC masks, direct connection, rows 1:1 aligned |
| use_sec_mask | **Default False** (feathered box, whole-window paste-back). When True, mask source priority: masks port > built-in pack masks > feathered-box fallback with a warning |
| feather_px | Feather pixels (default 16, 0–128): box mode feathers the rectangle edge; mask mode = erode feather/2 + Gaussian blur σfeather/2 (falls back to the original mask when small faces get hollowed out by erosion) |
| lowfreq_lock | Pyramid low-frequency lock: structure and color locked to the original frames, texture detail taken only from the resampled canvas. For low-σ detail sharpening only; keep OFF when repairing corrupted faces — otherwise it preserves the original frames' wrong facial geometry |
| motion_align | Paste-back position correction: automatically aligns the repainted face back to the real face position in the original frames, with motion fully inherited from the source video, eliminating paste-back jitter (especially at high σ / segment boundaries). Falls back to plain paste-back automatically if estimation fails |

## 5. Common Issues

| Symptom | Fix |
|---|---|
| Results don't change after changing parameters/code | Cache hit — check clear_cache and run once; always clear the cache after changing code |
| Resample reports missing inputs | Integrated mode without info / standalone mode without model+vae+clip |
| Blend reports a bbox port error | Connected shot_info — must be face_pack |
| Want to paste only the face, background untouched | Blend's use_sec_mask=True + Face_Cut's masks direct connection (the default feathered box replaces the whole window's content) |
| Hard paste-back edges | Increase feather_px; single-face mode has no mask — the feathered-box fallback is normal |
| UNCOVERED warning | That range has no face at all, passed through untouched; if unexpected, check the YOLO threshold |
| Masks all zero | Normal in single-face mode; in multi_sec mode check the SeC logs above |
| Color blotches remain on fixed small faces | Pick a RealESRGAN_x4plus-type SR in Face_Cut; if they persist, they come from multiple VAE round-trips — lower the step count and observe |
| SR fails to load | Falls back to lanczos automatically; only image super-resolution models are supported (video models like RIFE are not) |
| Per-subtrack detailed logs | Set _VERBOSE = True at the top of h3_facefix.py |
| Cache location | output/cache/node_<node id>/, can be deleted manually as a whole directory |
| Same identity shattered when two people cross / phantom short-lived identities | Auto-fixed since v20.4: mask arbitration + gap attribution + in-track interpolation bridging. Seeing "N frame(s) attributed to existing track(s)" in the SeC log means the fix is active. If still shattered, lower sec_threshold |
| 16GB card: crop/SR stage suddenly slows down with no OOM error | Windows silently spills VRAM overflow into "shared GPU memory" without an error. Keep unload_main_models on and lower sr_batch to 2 |
| Checked clear_cache on Face_Cut but Face_Resample still uses old results | clear_cache only clears that node's own cache. Face_Resample's crop-row cache invalidates automatically when crop pixels change (content hash) or the σ schedule changes; after node ids change, delete each node's cache directory output/cache/node_<node id>/ |
| After upgrading, old workflow ports turn red / get reset | Since v20.3, sec_auto_unload is removed (SeC now always unloads automatically), and unload_main_models & sr_batch are added — rewire and re-check |

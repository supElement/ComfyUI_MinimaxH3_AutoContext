# MinimaxH3 Face-Fix Pipeline (Face_Cut → Face_Resample → Face_Blend)

Three nodes chained, running on the output of the main sampler node (Minimax_H3_AutoContext_Sampler):

| Node | Step | Role |
|---|---|---|
| Minimax_H3_Face_Cut | ① detect & crop | shot detection + YOLO detection + optional SeC-4B tracking; crops each face into a uniform res² patch via per-frame smoothed windows |
| Minimax_H3_Face_Resample | ② refine | block-wise img2img resampling with the same model as the main sampler (block structure mirrors the main sampling segmentation + inter-block anchoring) |
| Minimax_H3_Face_Blend | ③ blend back | pastes the refined faces back onto the original frames per the geometry ledger and mask (zero VAE) |

## 1. Standard Wiring

| From | To |
|---|---|
| Main sampler info | Face_Cut.info and Face_Resample.info |
| Main sampler output frames | Face_Cut.images (images mode recommended) and Face_Blend.images |
| Face_Cut.crop_images | Face_Resample.crop_images |
| Face_Cut.face_pack | Face_Resample.face_pack |
| Face_Cut.shot_info | parameter node (assign per-shot prompts) |
| Face_Resample.images | Face_Blend.canvas |
| Face_Resample.bbox | Face_Blend.bbox |
| Face_Cut.masks | Face_Blend.masks (**direct wire**, not through Resample) |
| Face_Blend.images | Final output frames |

## 2. Face_Cut — Detection & Cropping

Pipeline: PySceneDetect shot detection → YOLO face detection → optional SeC-4B identity tracking → per-frame smoothed-window cropping.

Model download: [huggingface](https://huggingface.co/cglearned/Minimax_H3_Face_Cut/tree/main)

### Model folders

| Model | Parameter | Folder | Notes |
|---|---|---|---|
| YOLO face detection | face_model | ComfyUI/models/elementEasy/ | .pt/.pth/.onnx/.engine/.torchscript; error if none selected |
| SeC-4B identity tracking | sec_model | ComfyUI/models/sams/ | fp16 recommended; None = single-face mode |
| Upscale model | upscale_model | ComfyUI/models/upscale_models/ | image SR only (ESRGAN/RealESRGAN/UltraSharp etc.); avoid GFPGAN/CodeFormer face-restoration models |

After placing model files, **refresh or restart ComfyUI** so they appear in the dropdowns.
PySceneDetect is a Python dependency, not a model: pip install scenedetect if missing — the node downgrades to upstream-segmentation-only isolation (warning, not error).

### Key parameters

| Parameter | Notes |
|---|---|
| images / latent | Pick one. images mode recommended: detect on external frames at native size, never touches latent; latent mode decodes probe frames via VAE |
| yolo_threshold | Detection confidence (default 0.3); lower it if faces are missed |
| shot_threshold | Shot-cut threshold; higher = fewer cuts (default 40) |
| upscale_model | Optional SR chain: repeatedly upscale until ≥ canvas size (≤3 passes), then lanczos to the exact size; fixes ringing-induced color blotches on tiny faces; None = lanczos only |
| res / expand | Canvas size (default 512) / crop-window padding % (default 20) |
| skip_ratio | Faces ≥ res × this ratio skip resampling (default 0.8) |
| sec_model / sec_threshold / max_identities | None = single-face mode (largest face per frame, no mask); select a weight = multi-identity tracking + SeC masks |

## 3. Face_Resample — Canvas Refinement

Uses the crop rows as a canvas and refines them block by block via img2img; block structure mirrors the main sampling segmentation, blocks continue via anchoring.

**Two wiring modes**: integrated mode (info from the main sampler; model/vae/clip come from info.h3_runtime, local ports ignored) / standalone mode (leave info empty; local model/vae/clip required; prompts and segmentation come from parameter). The parameter port is required in both modes.

| Parameter | Notes |
|---|---|
| sigmas | Refinement σ schedule (from a scheduler); fewer steps = more conservative retouch |
| seed | Block sampling seed (auto-offset per block) |
| color_match | Per-subtrack Reinhard color match back to the source crops (default on) |
| ref_images | Reference images; passed only when declared as a Picture tag in the prompt |

Block principle: subtracks of the same identity and contiguous in time merge into one sequence → split into blocks at the main sampler's real segment boundaries (each block's prompt maps precisely to its segment by block midpoint) → each block's encoding is padded to the 17n+5 grid (tail frames duplicated, cropped after decoding) → **inter-block anchoring: each non-first block receives the previous block's real tail latent in its context head (frozen)**, preventing "tail frames look un-processed"; on large window jumps at boundaries, anchoring is kept or disabled based on the tracking mode.

## 4. Face_Blend — Blend-Back

Per-subtrack scaling + pixel paste-back, **zero VAE** (entirely in pixel space, no encode/decode loss).

| Port / parameter | Notes |
|---|---|
| images | Original video frames |
| canvas | Refined canvas from Face_Resample |
| bbox | face_pack from Face_Resample. ⚠ do NOT connect shot_info (a shot map is not geometry); connecting Face_Cut's face_pack also works (falls back to crop_off with a warning), but Resample is recommended |
| masks (optional) | SeC masks from Face_Cut, wired directly, rows 1:1 |
| use_sec_mask | **default False** (feathered box replaces the whole window). When True, mask source priority: masks port > built-in pack masks > fallback to feathered box with a warning |
| feather_px | Feather in pixels (default 16, 0–128): box mode feathers the rectangle edge; mask mode = erode feather/2 + gaussian blur σfeather/2 (falls back to the raw mask if erosion hollows out tiny faces) |

## 5. FAQ

| Symptom | Fix |
|---|---|
| Results unchanged after tweaking params/code | Cache hit — run once with clear_cache checked; always clear cache after code edits |
| Resample reports missing inputs | Integrated mode: info not connected / standalone mode: model+vae+clip not connected |
| Blend errors on the bbox port | You connected shot_info — it must be face_pack |
| Want to paste only the face, background untouched | Blend: use_sec_mask=True + wire Face_Cut's masks directly (the default feathered box replaces the whole window) |
| Hard edges after blend-back | Increase feather_px; single-face mode has no mask — feathered-box fallback is normal |
| UNCOVERED warning | That range has no face at all — frames pass through untouched; if unexpected, check the YOLO threshold |
| All-zero masks | Normal in single-face mode; in multi_sec mode check the SeC log above |
| Color blotches on small faces after refinement | Pick a RealESRGAN_x4plus-class SR model in Face_Cut; if they persist the source is repeated VAE round-trips — lower the step count to check |
| SR model fails to load | Auto-fallback to lanczos; image SR models only (video/interpolation models like RIFE unsupported) |
| Per-subtrack verbose log | Set _VERBOSE = True at the top of h3_facefix.py |
| Cache location | output/cache/node_<node id>/ — the whole folder can be deleted |

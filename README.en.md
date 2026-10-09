<div align="center">

[![Chinese](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

One-click MiniMax H3 long video auto-generation node: **Segmented reasoning + inter-segment anchoring + prompt timeline slicing + secondary sampling (2-samp) + seam correction + face repair (test branch)**.  
Under limited GPU memory, long videos are split into multiple independent reasoning segments, achieving seamless inter-segment concatenation through overlay enhancement methods, while automatically slicing prompts along the timeline to align each segment's generated content with the prompt rhythm; audio-video references are similarly sliced and aligned; only the referenced references in the current segment participate in reasoning.  
H3's GPU memory requirements grow exponentially with resolution, duration (each +5s doubles), and precision (fp8→bf16), so limited memory is relative—32GB GPU memory will overflow and slow down when generating 15s 1080p with bf16, and while segmentation increases the overhead of anchoring frames, peak GPU memory depends only on the length of a single segment, unrelated to the total target length, making actual reasoning time actually faster than generating long videos in one go.  
Supports secondary sampling. Video continuation, video pre-roll, dual video concatenation.  
Supports latent cache read/write, making it convenient to quickly skip already reasoned segments if reasoning is interrupted for some reason, with cache files stored per segment. When upstream parameters of the sampling node remain unchanged, existing latent cache files can be read.  
Cache disk usage: Added directory-level limits, default 32GB, with automatic cleanup of the oldest files when exceeded (adjustable via environment variable H3_CACHE_MAX_GB, set to 0 to disable)

⚠️Note: Changing models, including LoRA, sageattention, and other acceleration nodes, will not detect latent changes, so latent caches must be deleted (except versions V0.9.1 and above, which now support detection; if invalid, they can also be cleared using this method). There are two ways to delete latent caches:  
- Enable the clear_cache parameter on the Minimax_H3_AutoContext_Sampler node, which will force the creation of this node's cache files at the start of sampling.  
- Manually delete the corresponding folder in the cache directory (\ComfyUI\output\cache), with the folder name being "node_" + "node ID".

<img width="2230" height="976" alt="image" src="https://github.com/user-attachments/assets/5634914a-6f98-4d4f-b573-2c8b41e0c57e" />


<img width="2209" height="1030" alt="image" src="https://github.com/user-attachments/assets/9bbdda2a-d4ce-4836-b108-e359e72e31de" />

## BUG Fixes, Optimizations, and New Features

### Update Summary (V0.8.5–V0.9.2)

- **Face Repair Node** - Detailed description: [Chinese](https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext/blob/test/h3_fix_zh.md) | [English](https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext/blob/test/h3_fix_en.md)
  - `Minimax_H3_Face_Cut`: Detection and cropping, storyboard + YOLO detection + optional SeC-4B tracking.  
  - `Minimax_H3_Face_Resample`: Refinement, using the same model as main sampling for block-level img2img resampling (block structure mirrors main sampling segments + block-to-block anchoring).  
  - `Minimax_H3_Face_Blend`: Reattachment, refined faces are reattached to the original image using geometric accounting and masks per pixel.

<img width="1509" height="747" alt="image" src="https://github.com/user-attachments/assets/ce8931d0-a711-4cdc-8d3b-f1386008ce60" />

- Semantic Bridge
  - Integrated MiniMax-H3-Semantic-Bridge: Mixes the ~11MB student adapter distilled from SenseNova U1.5's teacher bridge into H3 conditioning to enhance prompt adherence (spatial relationships / counting / materials / reflections, etc.), eliminating the need for SenseNova participation during reasoning.  
  - ⚠️ The Semantic Bridge is not a universal method and may degrade performance in some cases; do not enable it unless necessary. Upstream v1 only verified FL2VA / text paths; Ref2VA reference paths (`ref_video` / `ref_audio`, including lip sync) are unverified and may degrade lip sync and vocal performance; use A/B testing to decide before enabling.  

- Cache and Performance
  - Supports checkpoint / accelerated LoRA / precision changes / function-level patch caching detection.  
  - When prompts are fixed, the text encoder is no longer reloaded per block; and fingerprint verification (pixel content fingerprint) is strengthened to avoid false positives when switching images.  

- Attention Correction Node
  - Added `Minimax_H3_TST_AttentionPatch` attention correction node: H3 TST attention correction — Spectrum tension diagnosis, suppressing temporal flickering / small face collapse.  
  - `tau` Suggested intensity 0.2-0.3, must be placed downstream of other attention patches.  


### Stability Update Summary (V0.5.8–V0.7.2)
- Parameter node compatibility fixes: Fixed issues where H3Parameter's total_frames / chunk_frames / context_frames connections to Math Expression nodes caused dead loops in segmented estimation requests or ComfyUI page freezes.  
- Added video_guide parameter: Optimizes video continuation, video pre-roll, and dual video middle concatenation, supporting segmentation. When non-none, the reference port of the sampling node will strictly crop based on context_frames; referencing rules are the same as for normal references and must be declared in the prompt.  

- Added ignore_latent_hash: Can ignore input_latent hash verification, avoiding latent judgment information changes from nodes causing cache misinvalidation. Nodes that do not change latent noise characteristics can be set to false; tested with [Minimax_H3-LatentUpscaler_Adv node](https://github.com/supElement/ComfyUI_Element_easy).  
- Hash Detection Enhancement: Improved hash detection parameters and fixed tensor mismatches caused by parameter changes upstream of the sampler.  
- Seam Correction Optimization: Minimax_H3_Seam_Correction removed the shot detection model, using PySceneDetect (pure CPU, no potential contamination), avoiding white screens in sampler previews
## 📖 Table of Contents

- [Node List](#nodes)
- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [Two-Sample and SplitSigmas High/Low Frequency](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Precautions (Node Limitations)](#limitations)

## <a id="nodes"></a> 🧩 Node List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter group node: Centralizes prompts/splits/resolution/audio parameters, outputs `parameter`, and provides real-time preview of "Estimated Splits" |
| **Minimax_H3_AutoContext_Sampler** | Main node: Split inference + anchor continuation + sampling (shared by one-sample and two-sample) |
| **Minimax_H3_Seam_Correction** | Seam correction node: Performs pixel-domain seam correction on decoded video segments |
| Minimax_H3_Face_Cut | ① Detection and Cropping: Storyboard + YOLO detection + optional SeC-4B tracking, frame-by-frame smooth window cropping into uniform res² small images |
| Minimax_H3_Face_Resample | ② Refinement: Uses the same model as the main sampling for block-level img2img resampling (mirrored block structure of main sampling splits + block anchor) |
| Minimax_H3_Face_Blend | ③ Blending: Refined faces are blended back into the original video frame by frame according to geometric ledger and mask (zero VAE) |
| Minimax_H3_TST_AttentionPatch | Attention Correction: H3 TST attention correction — spectral tension diagnosis, suppresses temporal flickering and small face collapse (limited effect) |

> Usage: `parameter 节点 --parameter--> 主节点`. Prompts are filled in the parameter node, and the main node receives `parameter` (required).

## <a id="features"></a> ✨ Core Features

### 🧩 Split Inference

- Split into multiple segments by `total_frames` / `chunk_frames` (frame units), recommended frame counts: 5, 22, 39, 56, 73, 90…
- The last segment is automatically extended to fill, avoiding excessively short tail segments
- `fps` is only used for audio synchronization and prompt second conversion

### 🔗 Inter-Segment Continuation

- **Overlay Enhancement**: Non-first segments automatically "take over" the ending frame of the previous segment, with new content naturally continuing from the end of the previous segment, eliminating pauses or position jumps at the seams
- The ending of the previous segment is used as a motion reference for the current segment, helping to continue the direction and speed of motion
- The audio of the previous segment is also passed in as "previous content" to help the sound continue naturally
- Inter-segment audio fades smoothly, aligned with the video frame count

> Frame count rules: `total_frames` / `chunk_frames` / `context_frames` all take 5, 22, 39, 56, 73, 90… (17n+5), the node will automatically align, generally no need for manual calculation.

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **Clip_Tag** | Splits prompts by user-defined tags (e.g., `段1`/`段2`), each tag corresponds to an independent video segment; segment duration is determined by the prompt content (tag duration > segment time markers > `total_frames/fps` fallback). |
| **timeline** | Splits prompts by explicit time markers (e.g., `0-2s`/`2-6s`), each time interval corresponds to a video segment; segment duration = interval length × `fps` and automatically snaps to legal grids; **ignores `total_frames` and `chunk_frames`**, completely determined by the prompt for total duration. Global segments (`【全局】`) remain in their original positions and are not extracted together. |
| **sequential** | Distributes prompts in sentence order evenly across the entire video timeline without splitting the prompt itself; video segmentation still follows `chunk_frames`. | 
| **global** | The entire prompt is used for all video segments (after `【全局】` tags are stripped), video segmentation follows `chunk_frames`. |

> In `Clip_Tag` and `timeline` modes, `total_frames` and `chunk_frames` parameters are ignored (segment length determined by prompt), only used when modes degrade (e.g., no tags/time markers detected) and fallback to these values.

### 🏷️ Clip_Tag Tag Splitting Mode

- Splits prompts by user-defined tags (e.g., `段1`/`段2`/`段3`), each segment = one chunk = all prompts for that segment
- Segment duration is determined by the prompt content (three levels of priority):
  1. Duration immediately following the tag line (e.g., `段1:0-5秒` → 5s; `段1:3-8秒` → 5s)
  2. Maximum end value of time markers within the segment (e.g., `【0-2秒】`+`【2-5秒】` → 5s)
  3. `total_frames / fps` default value fallback (single segment total frames fit `total_frames`)
- Time markers within the segment are **relative time** (starting from 0 for each segment), not global absolute time
- Overlapping frames are automatically generated for connection in non-first segments, and trimmed after generation
- Total duration automatically aligns to the target total frame count, as close as possible to the expected duration
- During inference, tags themselves are removed, and the rest of the prompt content is output according to `prompt_format`

### 🎯 Reference Intelligent Filtering (Image / Video / Audio)

- Automatically identifies reference images/videos/audios used in each segment of the prompt, only passing referenced materials to that segment
- Video is bound to its paired audio track, avoiding cross-talk between image and sound

### 🖌️ Two-Sample (Two-Sample)

- Main node `latent_input` inputs one-sample latent (or via latent amplification node) to enter two-sample mode
- Two-sample resolution **takes input latent as reference** (ignores width/height), achieving low-quality one-sample → high-quality two-sample
- `denoise` controls redraw intensity; `sigmas` supports custom sigma sequences (same as `SamplerCustomAdvanced`)
- `lock_audio`: Two-sample only redraws video, reuses one-sample audio

### 🎵 Audio Drive

- `drive_audio` (AUDIO, optional) + `audio_drive` switch
- Enabled, video follows this audio for generation, output audio = source audio itself (lip sync/rhythm driven by it)

## <a id="install"></a> 📦 Installation

### Method 1: Manual Installation (Manual Installation)

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
```

### Method 2: Install via Manager (Install using Manager)

Search `ComfyUI_MinimaxH3_AutoContext` in ComfyUI Manager and click Install.

### classic branch (original main branch)

First installation (only want to use the old classic branch):

      git clone -b classic https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
Already installed, switch from main to classic:

      git fetch origin  → git checkout classic  → git pull
## <a id="params"></a> ⚙️ Node Parameters

### Minimax_H3_AutoContext_parameter（Parameter Group Node）

| Parameter | Default Value | Description |
|----------|---------------|------------|
| long_prompt | — | Prompt (passed to the main node for inference, also used for "Estimated Segmentation" preview) |
| **clip_mode** | `Clip_Tag` | How the prompt maps to video segments: `Clip_Tag` / `timeline` / `sequential` / `global`. Modes `Clip_Tag` and `timeline` ignore `total_frames` and `chunk_frames`. |
| clip_tag | `段1` | Clip_Tag segmentation tag template (must end with a numeric sequence number), only effective when `clip_mode=Clip_Tag` |
| prompt_format | `official` | Prompt output format: `official` / `legacy` / `raw`. `official` uses the official MiniMax H3 [Shot] format, `legacy` is the old-style time tag, `raw` outputs as-is (used for Clip_Tag mode) |
| crop_mode | `stretch` | Reference image/first/last frame/reference video scaling cropping: `center` / `stretch` / `none` |
| ref_sync_mode | `segmented` | Whether reference video/audio is sliced per segment: `global` (uses complete material per segment) / `segmented` (slices by time ratio per segment) |
| width × height | 960×544 | One-batch resolution (overridden by latent_input during two-batch processing) |
| total_frames | 362 | Total number of frames to generate (17n+5); only serves as a fallback in `Clip_Tag`/`timeline` modes (when no labels/time tags), ultimately overridden by the sum of each segment |
| fps | 24 | Frame rate, used for audio synchronization and prompt second conversion |
| chunk_frames | 90 | Number of frames generated per segment (17n+5), only effective in `sequential` / `global` modes |
| context_frames | 22 | Inter-segment continuation frames (17n+5: 5/22/39/56…), recommended to be 22 or higher |
| lock_audio | `true` | Lock audio area during two-batch processing (noise_mask audio=0): resample video only, keep one-batch audio unchanged |
| audio_drive | `false` | Audio drive switch, after enabling, video follows drive_audio generation |
| video_guide | `none` | Video extension parameter, supports segmentation. none: disabled; pre_guide: video continuation (samples ref_video_0 or + ref_video_audio_0 port); post_guide: video pre-push (samples ref_video_0 or + ref_video_audio_0 port); pre_post_guide: dual-video middle connection (samples ref_video_0 or + ref_video_audio_0 port, ref_video_1 or + ref_video_audio_1 port). Anchored frame count determined by context_frames. Note: When not none, the reference port of the sampling node is forcibly clipped to the value set in the context_frames parameter. Reference logic is the same as normal references (only referenced if declared in the prompt) |
| semantic_bridge | false | Semantic bridge switch: distills SenseNova's ~11MB student adapter and mixes it into the condition tensor via C = H + alpha*(S-H), enhancing prompt adherence. Only transforms conditions, does not move model weights; inter-segment continuation anchors and reference channels are unaffected. Adapters are placed in models/semantic_bridge/ |
| semantic_bridge_adapter | none | Semantic bridge adapter file (models/semantic_bridge/ directory). none = disabled. New files added after startup require ComfyUI restart to appear in the dropdown |
| semantic_bridge_alpha | 0.10 | Fusion strength. Official recommendation starting point 0.10, official A/B examples use 0.15, suggest tuning between 0.10~0.15 |
| semantic_bridge_magnitude | per_token | Amplitude alignment before fusion: per_token RMS alignment per token (official recommendation) / global scalar alignment / none (no alignment) |

> The node displays "Estimated Segmentation" preview in real-time (calculated by frontend JS, does not participate in inference).

About Semantic Bridge

- Upstream project: Speach1sdef178/MiniMax-H3-Semantic-Bridge (adapter download: [HuggingFace](https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge))
- Connection method: **No workflow wiring changes required** — Enable `semantic_bridge` on the parameter node and select the adapter, the main sampling node applies it automatically per segment
- Scope of effect: Only mixes text/FL2VA condition tensors; inter-segment continuation anchors and reference material channels are unaffected
- Adapter installation: Download `.safetensors` and place in `ComfyUI/models/semantic_bridge/` (directory is created automatically), restart ComfyUI and select in the dropdown under `semantic_bridge_adapter`
- Caching behavior: Bridge parameters participate in segmentation cache fingerprinting; changing alpha / swapping adapter / toggling bridge causes corresponding segments to rebuild automatically
- ⚠️ Applicable scope: Upstream v1 only verified for FL2VA/text path; Ref2VA (reference video/audio) not verified, may degrade lip-sync and singing, please perform A/B testing yourself

### Minimax_H3_AutoContext_Sampler（Main Node）

| Parameter | Default Value | Description |
|----------|---------------|------------|
| model / vae / audio_vae / clip | — | MiniMax H3 model components |
| parameter | Required | Parameter group input (from parameter node) |
| sampler | Optional | External sampler object (SAMPLER), overrides built-in sampler_name/scheduler |
| sigmas | Optional | Custom sigma sequence (SIGMAS), highest priority |
| latent_input | Optional | Two-batch input latent (enables two-batch processing upon connection) |
| info | Optional | Parameter inheritance input (for multi-batch chaining, ensures segment consistency) |
| first_frame / last_frame | Optional | First/last frame anchoring (FL2VA) |
| video_context_denoise | 0.0 | Inter-segment continuation strength (only for non-first segments): 0=exact continuation of previous segment's end, 1=regenerate, intermediate values=soft mix. When connected to SplitSigmas for two-batch processing, it's recommended to set 1 to avoid screen artifacts |
| seed | 0 | Random seed (control_after_generate) |
| steps / cfg | 30 / 1.0 | Sampling steps / CFG |
| sampler_name / scheduler | euler / simple | Built-in sampler / scheduler |
| denoise | 1.0 | Redraw strength (1=full resampling, smaller value preserves more original structure) |
| enable_cache | true | Store/read latent cache, automatically creates folders named "node+nodeID" in "\ComfyUI\output\cache" directory, upstream nodes or parameter changes will overwrite existing latent cache files |
| clear_cache | false | Forcefully rebuild latent cache files |
| ignore_latent_hash | false | Ignore hash validation of input port input_latent. Practical scenario: Some latent processing nodes alter latent judgment information, causing minor latent changes to result in cache incompatibility and wasted inference time. Suggest setting to true in such cases |
| ref_image_N / ref_video_N / ref_video_audio_N / ref_audio_N | Optional | Reference materials (Autogrow dynamic ports) |
| drive_audio | Optional | Audio drive source |
## <a id="output"></a> 📤 Output

| Output | Description |
|------|------|
| **latent** | Merged audio-video latent, connected to VAE Decode, or upscaled then followed by binary sampling |
| **denoised_latent** | Clean latent output, used for binary sampling continuation / preview |
| **info** | Segment parameters (Dict), passed to the next main node's info input, ensuring consistent multi-sampling segmentation |



## <a id="second-pass"></a> 🔄 Binary Sampling and SplitSigmas High/Low Frequency

### Basic Binary Sampling (Low-Resolution First Sample → High-Resolution Second Sample)

```text
parameter node ──parameter──> Main node (first sample, 864×480)
    └─ latent / denoised_latent ──> [Separate AV] ──> video_latent ──> latent upscaled ──> [Merge AV] ──> Main node(second sample).latent_input
Binary sampling node: parameter shared (or info inherited), optional denoise 0.4~0.6
```

- Binary sampling resolution is based on `latent_input`, ignores parameter's width/height

### SplitSigmas High/Low Frequency (Save Time, Enhance Clarity)

> ⚠️ **Audio Constraint**: High/Low frequency **only affects video** (audio segments need complete sampling), audio should maintain complete sampling.

```text
First sample node: Complete sampling (not connected to high_sigmas, audio complete denoising)
          → denoised_latent → Separate amplify video (audio unchanged) → Merge → second sample.latent_input
Second sample node: sigmas ← low_sigmas (only run low sigma segments to enhance details)
          lock_audio = True (reuse first sample complete audio)
          video_context_denoise = 1.0 (continuation area redraws together with new area, avoids flickering)
```

> 💡 **Second sample `video_context_denoise`**: When connected to SplitSigmas, setting 0 (precise continuation) may cause flickering at the boundary between continuation area and newly redrawn area; setting 1.0 makes the continuation area redraw synchronously to avoid it. If the seam is slightly discontinuous, it can be reduced to 0.3~0.5 for a compromise. First sample remains default 0.

## <a id="seam"></a> 🧵 Seam Correction Node (Minimax_H3_Seam_Correction)

| Parameter | Default Value | Description |
|------|--------|------|
| `fix_color_preset` | `"medium"` | **Color/Exposure Processing Level**<br>`off`: No processing; <br>`low`: Per-channel brightness gain, correction amount halved, most conservative, no color bias; <br>`medium`: Per-channel brightness gain, only corrects seam level jumps (recommended); <br>`high`: MKL linear color migration, longer statistical window, more stable during large motion; <br>`max`: Frame-by-frame brightness normalization across the entire clip, eliminates intra-segment gradient drift, but will flatten the actual brightness variation in the frame (e.g., sky darkening/entering a tunnel), near-black frames are ineffective (reported in logs). |
| `fix_motion_preset` | `"off"` | **Seam Continuity (Optical Flow Alignment+Blending) Level**<br>`off`: No processing (recommended to first observe the effect with color level); <br>`low/medium/high/max`: The higher the level, the more frames and intensity participate in blending, but it may introduce slight blurring or breathing effect. |
| `fix_flash` | `false` | **Flash Frame Processing** (instantaneous brightness jumps at boundaries). Independent switch, uses temporal fusion logic. If the scene has reasonable rapid brightness changes like lightning, explosions, etc., suppression will flatten these effects. Still effective independently even when `fix_motion_preset=off` is active. |
| `flash_threshold` | `0.30` | Transient correction selection threshold (percentage of anomalous pixels), smaller value is more aggressive (corrects more frames), recommended `0.20` ~ `0.40`. |
| `cut_threshold` | `15.0` | PySceneDetect's sensitivity threshold (range `5.0` ~ `50.0`), smaller value is more sensitive, recommended `10` ~ `20`. |
| `blend_frames` | `2` | Seam level transition window (frames, 0~8): After exposure alignment, smooth the brightness transition of `blend_frames` frames before and after the boundary as a smooth ramp; larger value results in smoother transition, more natural, but large motion scenes may cause slight blurring/breathing effect; `0` means disabled. |
| `use_gpu` | `true` | Use CUDA GPU for statistics, color transformation, and optical flow calculation (automatically fallback to CPU if unavailable). |

⚠️ Removed the scene detection model, using PySceneDetect (pure CPU, no potential contamination).

> Usage: `VAE Decode → H3_Seam_Correction → Save/Video`.

> ⚠️ Note: This node only does seam correction for the visual, cannot fix artifacts generated by the upstream binary sampling.

## <a id="prompt-examples"></a> ✍️ Prompt Writing Examples

### Timeline Mode (auto / timeline)

```text
0-5s: ...
5-10s: ...

integrated_multimodal_description
....

overall_soundscape
```

> Paragraphs marked with `0-5s` are split by time, unmarked paragraphs (styles/effects/prohibitions) are automatically combined into each window.

### Global Mode (global)

> The entire prompt applies to all segments, suitable for homogeneous actions in a single take throughout.

### Clip_Tag Mode (Segment by Tags)

> `clip_mode` set to `Clip_Tag`, `clip_tag` filled with tag template (must end with a numeric sequence).

**Tag Template Examples**

| Template | Match |
|------|------|
| `段1` | `段1` / `段2` / `段3` (prefix "Segment"+number) |
| `A01` | `A01` / `A02` / `A03` (prefix "A"+number) |
| `[片段001]` | `[片段001]` / `[片段002]` (prefix "[Clip"+number+suffix"]") |

**Tag Writing**: Tags occupy a line as a separator, recommended to newline after the tag. Without newline, it can also be processed (separator skipped to take segment content):

```text
Segment 1:3s
Video:
...
Audio design:
...


Segment 2:3-8s
Video:
0-2 seconds:
...
2-5 seconds:
...
Audio design:
0-5 seconds:...
```

**Segment Duration Rules** (three levels of priority):

1. Duration immediately following the tag line: `段1:0-5秒` → 5 seconds; `段1:3-8秒` → 5 seconds (duration markers are removed from the prompt)
2. In-segment time markers 0-based: `【0-2秒】`+`【2-5秒】` → 5 seconds
3. None → `chunk_frames / fps` as fallback

**prompt_format Selection**

- `official` / `legacy`: In-segment time markers automatically converted to relative coordinates within the segment for rendering
- `raw`: Output as is after removing tags, time markers remain unchanged (suitable for structured prompts generated by large models)
## <a id="limitations"></a> 📝 Prompt Considerations (Node Limitations)

> The following considerations **do not apply** to simple, always-effective prompt scenarios (i.e., all segments share the same prompt, global mode),
> such as: voice-over digital humans (of course, lines need to be segmented), camera/composition changes are minimal in videos, or video character replacement, etc., common prompt scenarios.

### 1️⃣ Core Principle: Temporal Exclusivity

> When using segmented reasoning (Chunks), please strictly adhere to the **temporal exclusivity** principle—each segment's prompt can only describe the **new changes** that are "occurring" in that segment relative to the end of the previous segment.

- **Segments as "Relays"**: When generating the Nth segment, its starting frame state (position, action posture, camera position) is completely implicitly provided by the "anchored frames (Context Frames)" at the end of the previous segment. You don't need to repeat this starting state in the prompt.
- **Prohibited "Retrospection" and "Overlap"**: The prompt for the Nth segment absolutely cannot repeat actions or camera movements already completed in the N-1th segment. If repeated, the model will receive conflicting instructions with the anchored frame's visuals, leading to jerky generation, illogical motion, or repeated actions.
- **Zeroing at Boundaries**: When switching segments, zero out the "ongoing actions" of the previous segment. The new segment's prompt should act like a "new instruction after pressing the shutter," targeting only the displacement, actions, or new elements occurring within the current new time frame.

**❌ Incorrect Usage (Conflicting Overlap)**

```text
Segment 1: 3s
"Object A moves to position B"
Segment 2: 3-6s
"Object A moves to position B and then turns around at position B"
```

> Problem Analysis: At the end of the 1st segment, the anchored frame shows Object A has arrived at position B and just stopped. However, the 2nd segment's prompt forcibly requires "Object A moving to position B," which conflicts with the anchored frame's static result "already arrived," causing the model to attempt "re-movement," resulting in creepy or jump-cut effects.

**✅ Correct Usage (Seamless Progression)**

```text
Segment 1: 3s
"Object A moves to position B and stops finally at position B" (emphasizing action closure)
Segment 2: 3-6s
"Stand still, then Object A slowly turns direction" (directly describe the new action after the end of the previous segment)
```

> Correct Logic: The 2nd segment completely discards the description of the "movement process," assuming "stopped at B point" is a given fact, and only describes the subsequent "turning" new action, allowing the model to perfectly continue using the anchored frame.

> 🚀 **In a nutshell**: The end of the previous segment is the "result," the beginning of the next segment is the "new action after the result," don't put the "process that led to the result" into the next segment.

### 2️⃣ Core Principle: Per-Segment Reference Declaration

> When using segmented reasoning with reference images/videos (image1, video1, etc.), please strictly adhere to the **per-segment reference declaration** principle—each segment's prompt must independently and completely declare all reference materials required for that segment, references are not "memorized" or "inherited" to the next segment (only the referenced references participate in reasoning for the current segment).

- **No Global Memory**: The node parses the reference labels explicitly mentioned in the current segment's prompt to accurately determine which materials are needed for that segment. Writing image1 in the 1st segment only means it was used in the 1st segment; the next segment will re-scan.
- **No Write, No Transfer**: If the Nth segment doesn't write image1 again, that reference image won't be passed in, causing inconsistency in characters/objects.

**❌ Incorrect Usage (Implicit Inheritance)**

```text
Segment 1[3s]: image1 is Object A, Object A is moving forward.
Segment 2[3-6s]: Object A stops, turns to look at the camera. (No image1 written)
```

**✅ Correct Usage (Explicit Per-Segment)**

```text
Segment 1[3s]: image1 is Object A, Object A is moving forward.
Segment 2[3-6]: image1 is Object A, Object A stops, turns to look at the camera.
```
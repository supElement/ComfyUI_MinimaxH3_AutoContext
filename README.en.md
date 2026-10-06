<div align="center">

[![Chinese](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

One-click MiniMax H3 long video automated generation node: **segmented reasoning + inter-segment anchoring + prompt timeline slicing + secondary sampling (2-sampling) + seam correction + face repair (test branch)**. In limited GPU memory, long videos are split into multiple independent segments for reasoning, achieving seamless inter-segment concatenation through overlay enhancement methods, while automatically slicing prompts along the timeline to align each segment's generated content with the prompt rhythm; audio-visual references are similarly sliced and aligned; only the referenced references in the current segment participate in reasoning.  
H3's GPU memory requirements increase exponentially with resolution, duration (each +5s doubles), and precision (fp8→bf16), so limited memory is relative—32GB GPU memory will overflow and slow down when generating 15s 1080p with bf16, and while segmentation increases the overhead of anchoring frames, the peak GPU memory only depends on the length of a single segment, unrelated to the target total length, making actual reasoning time actually faster than generating a long video in one go.  
Supports secondary sampling. Video continuation, video forward, dual video concatenation.  
Supports latent cache read/write, facilitating quick skipping of already reasoned segments if reasoning is interrupted for some reason, with cache files stored per segment. Under unchanged upstream parameters for the sampling node, existing latent cache files can be read.  
Cache disk usage: Added directory-level limit, default 32 GB, with automatic cleanup of oldest files when exceeded (adjustable via environment variable H3_CACHE_MAX_GB, set to 0 to disable)

⚠️Note: Changing the model, including LoRA, SageAttention, and other acceleration nodes, will not detect latent changes, so latent cache must be deleted (except for test branch V0.9.1, which now supports detection; if invalid, it can also be cleared using this method), with two methods to delete latent cache:  
- Enable the clear_cache parameter on the Minimax_H3_AutoContext_Sampler node, which forces the creation of this node's cache files at the start of sampling.  
- Manually delete the corresponding folder in the cache directory (\ComfyUI\output\cache), with folder name "node_" + "node ID".

<img width="2230" height="976" alt="image" src="https://github.com/user-attachments/assets/5634914a-6f98-4d4f-b573-2c8b41e0c57e" />


<img width="2209" height="1030" alt="image" src="https://github.com/user-attachments/assets/9bbdda2a-d4ce-4836-b108-e359e72e31de" />

## BUG Fixes, Optimizations, and New Features

### test branch

First installation (for test branch):

      git clone -b test https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
Already installed, switching from main to test:

      git fetch origin  → git checkout test  → git pull
### V0.9.2 (test branch)

- Optimized SR face zoom processing speed, the zoom model is used first for a one-time process only for frames with a side length less than 192px.
- Optimized the logic for automatically extracting id reference selection.
- Added output port identity_refs to Minimax_H3_Face_Cut and Minimax_H3_Face_Resample nodes, used only for verifying each id reference image automatically captured, both nodes output the same.

### V0.9.1 (test branch)

- Supports checkpoint / accelerated LoRA / precision change / function-level patch cache detection.
- Added automatic ID face feature anchoring, extracting the clearest (largest proportion) face frame as the reference image, default <Picture 1>; if connected and declared reference images, the reference number of the automatically anchored frame changes to the reference image count + 1, e.g., if 2 reference images are connected, the automatically anchored frame becomes <Picture 3> reference; if it negatively affects the generation result, the identity_ref parameter can be turned off.
- Added separate face repair prompt input parameter.
- Optimized peak GPU memory usage; optimized face reply stability; optimized cache logic stability.

### V0.9.0 (test branch)

- Integrated semantic bridge capability from the project MiniMax-H3-Semantic-Bridge: mixing the ~11MB student adapter distilled from SenseNova U1.5 teacher bridge into H3 conditions to enhance prompt adherence (spatial relationships/counting/materials/reflections, etc.), no SenseNova participation required during reasoning. The semantic bridge is not a universal method and may degrade performance in some cases, so do not enable it unless necessary.
- `Minimax_H3_AutoContext_parameter` node added 4 parameters: `semantic_bridge` (toggle), `semantic_bridge_adapter` (adapter file), `semantic_bridge_alpha` (fusion strength, default 0.10, official A/B examples use 0.15), `semantic_bridge_magnitude` (amplitude alignment method, default per_token).
- Only changes the condition tensor, does not modify H3 DiT weights; inter-segment anchoring and reference material channels remain unaffected and can be enabled or disabled at any time.
- ⚠️ Upstream v1 only verified FL2VA/text path; Ref2VA reference path (ref_video / ref_audio, including lip sync) not verified, tested to potentially degrade lip sync and singing performance, use A/B comparison before enabling/disabling.
- Note: Face repair chain (Face_Cut / Face_Resample / Face_Blend) does not apply semantic bridge.

Optimizations:
- Optimized face repair node processing logic; optimized memory and GPU memory usage; optional unloading main model/VAE/CLIP before SR zoom.
- When fixing prompts, no longer reload text encoder per block, and in this case, strengthen fingerprint verification (pixel content fingerprint) to avoid false hits when switching images.
- Added sr_batch (default 4, range 1~16) SR frames per forward; only affects speed and peak GPU memory, does not affect results.
- Removed sec_auto_unload, SeC-4B tracking now always unloads automatically, no toggle needed.

### v0.8.5 (test branch)

<img width="2156" height="629" alt="image" src="https://github.com/user-attachments/assets/b0b9373a-258b-485e-b5e8-0b3778f744e3" />  <br>
  
Added Minimax_H3_TST_AttentionPatch attention correction node; H3 TST attention correction — chord tension diagnosis + video row query adaptive scaling, suppress temporal flicker/small face collapse. tau strength (0.2), needs to be placed downstream of other attention patches.  

Added and optimized face repair node. Detailed description: [Chinese](https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext/blob/test/h3_fix_zh.md) | [English](https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext/blob/test/h3_fix_en.md)</sub>
- Minimax_H3_Face_Cut: Detection and cropping, storyboard + YOLO detection + optional SeC-4B tracking.
- Minimax_H3_Face_Resample: Refinement, uses the same sampling model for block-level img2img resampling (block structure mirrors main sampling segments + block-to-block anchoring) .
- Minimax_H3_Face_Blend: Paste back, refined faces are pasted back into the original scene pixel by pixel using geometric accounting and masks.

### Stability Updates Summary (V0.5.8–V0.7.2)
- Parameter node compatibility fixes: Fixed issues where connecting H3Parameter's total_frames / chunk_frames / context_frames to Math Expression nodes caused dead loops in segmented estimation requests or ComfyUI page freezes.
- Added video_guide parameter: Optimized video continuation, video forward, dual video middle concatenation, supports segmentation. When non-none, the reference port of the sampling node will be strictly trimmed according to context_frames; reference rules are the same as normal references, need to be declared in prompts.

- Added ignore_latent_hash: Can ignore input_latent hash verification, avoiding nodes that change latent judgment information from causing cache misinvalidation. Nodes that do not change latent noise characteristics can be set to false, [Minimax_H3-LatentUpscaler_Adv node](https://github.com/supElement/ComfyUI_Element_easy) tested.
- Hash detection enhancement: Improved hash detection parameters, fixed tensor mismatch caused by parameter changes upstream of the sampler.
- Seam correction optimization: Minimax_H3_Seam_Correction removed shot detection model, replaced with PySceneDetect (pure CPU, no potential contamination), avoids sampler preview white screen
## 📖 Table of Contents

- [Node List](#nodes)
- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [Two-Stage & SplitSigmas High/Low Frequency](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Precautions (Node Limitations)](#limitations)

## <a id="nodes"></a> 🧩 Node List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter group node: Centralizes prompts/segments/resolution/audio parameters, outputs `parameter`, and provides real-time preview of "Estimated Segments" |
| **Minimax_H3_AutoContext_Sampler** | Main node: Segment inference + anchor continuation + sampling (one-stage/two-stage shared) |
| **Minimax_H3_Seam_Correction** | Seam correction node: Performs pixel-domain correction on inter-segment seams of decoded video |

> Usage: `parameter 节点 --parameter--> 主节点`. Prompts are filled in the parameter node, and the main node receives `parameter` (required) through.

## <a id="features"></a> ✨ Core Features

### 🧩 Segment Inference

- Segments into multiple parts based on `total_frames` / `chunk_frames` (frame units), frame counts recommended: 5, 22, 39, 56, 73, 90…
- Automatically pads the last segment to avoid excessively short tail segments
- `fps` is only used for audio synchronization and prompt second conversion

### 🔗 Inter-segment Continuation

- **Overlay Enhancement**: Non-first segments automatically "take over" the ending frame of the previous segment, with new content naturally continuing from where the previous segment ended, eliminating pauses or position jumps at the seams
- The ending frame of the previous segment is used as a motion reference for the current segment, helping to maintain motion direction and speed
- The audio of the previous segment is also passed as "previous content" to help the sound continue naturally
- Inter-segment audio fades smoothly, aligned with the video frame count

> Frame count rules: `total_frames` / `chunk_frames` / `context_frames` all take 5, 22, 39, 56, 73, 90… (17n+5), the node automatically aligns, generally no need for manual calculation.

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **Clip_Tag** | Segments prompts based on user-defined tags (e.g., `段1`/`段2`), each tag corresponds to an independent video segment; segment duration is determined by the prompt content (tag-following duration > segment time markers > `total_frames/fps` fallback). |
| **timeline** | Segments prompts based on explicit time markers (e.g., `0-2s`/`2-6s`), each time interval corresponds to a video segment; segment duration = interval length × `fps` and automatically snaps to a legal grid; **ignores `total_frames` and `chunk_frames`**, entirely determined by the prompt. Global segments (`【全局】`) remain in their original positions and are not extracted together. |
| **sequential** | Distributes the prompt in sentence order evenly across the entire video timeline without splitting the prompt itself; video segmentation still follows `chunk_frames`. | 
| **global** | The entire prompt is used for all video segments (`【全局】` tags stripped), video segmentation follows `chunk_frames`. |

> In `Clip_Tag` and `timeline` modes, `total_frames` and `chunk_frames` parameters are ignored (segment length determined by the prompt), only used when modes degrade (e.g., no tags/time markers detected) and fallback to these values.

### 🏷️ Clip_Tag Tag Segmentation Mode

- Segments prompts based on user-defined tags (e.g., `段1`/`段2`/`段3`), each segment = one chunk = all prompts for that segment
- Segment duration determined by prompt content (three priority levels):
  1. Duration immediately following the tag line (e.g., `段1:0-5秒` → 5s; `段1:3-8秒` → 5s)
  2. Maximum end value of time markers within the segment (e.g., `【0-2秒】`+`【2-5秒】` → 5s)
  3. `total_frames / fps` default value fallback (single segment total frames fit `total_frames`)
- Time markers within the segment are **relative time** (starting from 0 for each segment), not global absolute time
- Overlapping frames are automatically generated for non-first segments to ensure smooth transitions, and trimmed after generation
- Total duration automatically aligns to the target total frame count, as close as possible to the expected duration
- Tags themselves are removed during inference, and the remaining prompt content is output based on `prompt_format`

### 🎯 Reference Intelligent Filtering (Image / Video / Audio)

- Automatically identifies reference images/videos/audios used in each segment of the prompt, only passing referenced materials to that segment
- Video and its paired audio track are bound together to avoid cross-talk between visuals and sound

### 🖌️ Two-Stage Sampling (Two-Stage)

- Main node `latent_input` inputs one-stage latent (or latent amplified node) to enter two-stage mode
- Two-stage resolution **takes input latent as reference** (ignores width/height), achieving low-quality one-stage → high-quality two-stage
- `denoise` controls redraw intensity; `sigmas` supports custom sigma sequences (same as `SamplerCustomAdvanced`)
- `lock_audio`: Two-stage only redraws video, reuses one-stage audio

### 🎵 Audio Drive

- `drive_audio` (AUDIO, optional) + `audio_drive` switch
- Enabled, video follows this audio to generate, output audio = source audio itself (lip sync/rhythm driven by it)

## <a id="install"></a> 📦 Installation

### Method 1: Manual Installation (Manual Installation)

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
```

### Method 2: Install via Manager (Install using Manager)

Search `ComfyUI_MinimaxH3_AutoContext` in ComfyUI Manager and click Install.

## <a id="params"></a> ⚙️ Node Parameters

### Minimax_H3_AutoContext_parameter (Parameter Group Node)

| Parameter | Default Value | Description |
|------|--------|------|
| long_prompt | — | Prompt (passed to main node for inference, also used for "Estimated Segments" preview) |
| **clip_mode** | `Clip_Tag` | How prompts map to video segments: `Clip_Tag` / `timeline` / `sequential` / `global`. `Clip_Tag` and `timeline` modes ignore `total_frames` and `chunk_frames`. |
| clip_tag | `段1` | Clip_Tag segmentation tag template (must end with a numeric sequence number), only effective when `clip_mode=Clip_Tag` |
| prompt_format | `official` | Prompt output format: `official` / `legacy` / `raw`. `official` uses the official MiniMax H3 [Shot] format, `legacy` is the old-style time tag, `raw` outputs as-is (used for Clip_Tag mode) |
| crop_mode | `stretch` | Reference image/first/last frame/reference video scaling/cropping: `center` / `stretch` / `none` |
| ref_sync_mode | `segmented` | Whether reference video/audio is sliced per segment: `global` (uses full material per segment) / `segmented` (slices by time ratio per segment) |
| width × height | 960×544 | One-stage resolution (latent_input overrides during two-stage) |
| total_frames | 362 | Total frames to generate (17n+5); only acts as a fallback in `Clip_Tag`/`timeline` modes (no tags/no time markers), ultimately covered by the sum of each segment |
| fps | 24 | Frame rate, used for audio synchronization and prompt second conversion |
| chunk_frames | 90 | Frames per segment generated (17n+5), only effective in `sequential` / `global` modes |
| context_frames | 22 | Inter-segment continuation frames (17n+5: 5/22/39/56…), recommended 22 or higher |
| lock_audio | `true` | Lock audio region during two-stage (noise_mask audio=0): resamples video only, keeps one-stage audio unchanged |
| audio_drive | `false` | Audio drive switch, enabled video follows drive_audio to generate |
| video_guide | `none` | Video extension parameter, supports per-segment. none: disabled; pre_guide: video continuation (sampler node ref_video_0 or + ref_video_audio_0 port); post_guide: video pre-push (sampler node ref_video_0 or + ref_video_audio_0 port); pre_post_guide: dual-video middle connection (sampler node ref_video_0 or + ref_video_audio_0 port, ref_video_1 or + ref_video_audio_1 port). Anchor frame count determined by context_frames. Note: Non-none values force the corresponding reference port of the sampler node to be forcibly cropped to the value set in the context_frames parameter. Reference logic is the same as normal references (declared in the prompt, only then referenced) |
| semantic_bridge | false | Semantic bridge switch: Mixes SenseNova distilled ~11MB student adapters into the condition tensor via C = H + alpha*(S-H), enhancing prompt adherence. Only transforms conditions, does not move model weights; inter-segment continuation anchors and reference channels are unaffected. Adapters placed in models/semantic_bridge/ |
| semantic_bridge_adapter | none | Semantic bridge adapter file (models/semantic_bridge/Underneath .safetensors). none = disabled. New files added after startup require ComfyUI restart to appear in the dropdown |
| semantic_bridge_alpha | 0.10 | Mixing strength. Official recommendation starts at 0.10, official A/B examples use 0.15, suggest tuning between 0.10~0.15 |
| semantic_bridge_magnitude | per_token | Amplitude alignment before mixing: per_token RMS alignment per token (official recommendation) / global scalar tensor alignment / none no alignment |

> The node real-time displays "Estimated Segments" preview (frontend JS calculates, does not participate in inference).

About Semantic Bridge

- Upstream project: Speach1sdef178/MiniMax-H3-Semantic-Bridge (adapter download: [HuggingFace](https://huggingface.co/speach1sdef178/MiniMax-H3-Semantic-Bridge))
- Connection method: **No workflow wiring changes needed** — Enable `semantic_bridge` on the parameter node and select the adapter, the main sampling node applies it automatically per segment
- Scope of effect: Only mixes text/FL2VA condition tensors; inter-segment continuation anchors and reference material channels are unaffected
- Adapter installation: Download `.safetensors` and place in `ComfyUI/models/semantic_bridge/` (directory auto-created), restart ComfyUI and select in the dropdown under `semantic_bridge_adapter`
- Caching behavior: Bridge parameters participate in segment caching fingerprint, changing alpha / swapping adapters / toggling bridge causes corresponding segments to rebuild automatically
- ⚠️ Applicable scope: Upstream v1 only verified FL2VA/text path; Ref2VA (reference video/audio) not verified, tested may degrade lip sync/singing, please A/B test yourself

### Minimax_H3_AutoContext_Sampler (Main Node)

| Parameter | Default Value | Description |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 model components |
| parameter | Required | Parameter group input (from parameter node) |
| sampler | Optional | External sampler object (SAMPLER), overrides built-in sampler_name/scheduler |
| sigmas | Optional | Custom sigma sequence (SIGMAS), highest priority |
| latent_input | Optional | Two-stage input latent (connected to enable two-stage) |
| info | Optional | Parameter inheritance input (multi-samplingseries, ensures segment consistency) |
| first_frame / last_frame | Optional | First/last frame anchors (FL2VA) |
| video_context_denoise | 0.0 | Inter-segment continuation strength (only non-first segments): 0=exact continuation of previous segment ending, 1=regenerate, intermediate values=soft mix. When connected to SplitSigmas, suggest setting 1 to avoid screen artifacts |
| seed | 0 | Random seed (control_after_generate) |
| steps / cfg | 30 / 1.0 | Sampling steps / CFG |
| sampler_name / scheduler | euler / simple | Built-in sampler / scheduler |
| denoise | 1.0 | Redraw strength (1=full resampling, smaller keeps more original structure) |
| enable_cache | true | Store/read latent cache, automatically creates folders named "node+nodeID" in "\ComfyUI\output\cache" directory, latent cache files are overwritten when upstream nodes or parameters change |
| clear_cache | false | Force rebuild latent cache files |
| ignore_latent_hash | false | Ignore hash validation of input port input_latent. Use case: Some latent processing nodes alter latent judgment information, making tiny latent changes cause cache incompatibility and waste inference time, suggest setting true |
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
    └─ latent / denoised_latent ──> [Separate AV] ──> video_latent ──> latent upscaled ──> [Merge AV] ──> Main node (second sample).latent_input
Binary sampling node: parameter shared (or info inherited), optional denoise 0.4~0.6
```

- Binary sampling resolution is based on `latent_input`, ignores parameter's width/height

### SplitSigmas High/Low Frequency (Save Time, Enhance Clarity)

> ⚠️ **Audio Constraint**: High/Low frequency **only affects video** (audio segment transitions require full sampling), audio should remain full sampled.

```text
First sample node: Full sampling (no connection to high_sigmas, audio fully denoised)
          → denoised_latent → Separate amplify video (audio unchanged) → Merge → second sample.latent_input
Second sample node: sigmas ← low_sigmas (only run low sigma segments to enhance details)
          lock_audio = True (reuse first sample full audio)
          video_context_denoise = 1.0 (continuation area redraws together with new area, avoids flickering)
```

> 💡 **Binary Sampling `video_context_denoise`**: When connected to SplitSigmas, setting 0 (precise continuation) may cause flickering at the boundary between continuation area and newly redrawn area; setting 1.0 synchronizes the continuation area redraw to avoid it. If the seam appears slightly discontinuous, it can be reduced to 0.3~0.5 as a compromise. First sample remains default 0.

## <a id="seam"></a> 🧵 Seam Correction Node (Minimax_H3_Seam_Correction)

| Parameter | Default Value | Description |
|------|--------|------|
| `fix_color_preset` | `"medium"` | **Color/Exposure Processing Level**<br>`off`: No processing; <br>`low`: Per-channel brightness gain, correction amount halved, most conservative, no color bias; <br>`medium`: Per-channel brightness gain, only corrects seam level jumps (recommended); <br>`high`: MKL linear color migration, longer statistical window, more stable during large motion; <br>`max`: Full frame per-channel brightness normalization, eliminates intra-segment gradient drift, but flattens the actual brightness variation in the frame itself (e.g., sky darkening/entering a tunnel), invalid for near-black frames (reported in logs). |
| `fix_motion_preset` | `"off"` | **Seam Continuity (Optical Flow Alignment+Blending) Level**<br>`off`: No processing (recommended to first observe the effect with color level); <br>`low/medium/high/max`: The higher the level, the more frames and intensity participate in blending, but it may introduce slight blurring or breathing effect. |
| `fix_flash` | `false` | **Flash Frame Processing** (instantaneous brightness jumps at boundaries). Independent switch, uses temporal fusion logic. If the frame has reasonable rapid brightness changes like lightning, explosions, etc., suppression will flatten these effects. Still effective independently even when `fix_motion_preset=off` is active. |
| `flash_threshold` | `0.30` | Transient correction selection threshold (percentage of anomalous pixels), smaller value is more aggressive (corrects more frames), recommended `0.20` ~ `0.40`. |
| `cut_threshold` | `15.0` | PySceneDetect's sensitivity threshold (range `5.0` ~ `50.0`), smaller value is more sensitive, recommended `10` ~ `20`. |
| `blend_frames` | `2` | Seam level transition window (frames, 0~8): After exposure alignment, smooth the brightness transition of `blend_frames` frames before and after the boundary as a smooth ramp; larger value results in smoother transition, more natural, but large motion scenes may cause slight blurring/breathing effect; `0` means disabled. |
| `use_gpu` | `true` | Use CUDA GPU for statistics, color transformation, and optical flow calculation (automatically fallback to CPU if unavailable). |

⚠️ Removed the scene detection model, using PySceneDetect (pure CPU, no potential contamination).

> Usage: `VAE Decode → H3_Seam_Correction → Save/Video`.

> ⚠️ Note: This node only performs visual seam correction, cannot fix artifacts generated by the upstream binary sampling.

## <a id="prompt-examples"></a> ✍️ Prompt Writing Examples

### Timeline Mode (auto / timeline)

```text
0-5s: ...
5-10s: ...

integrated_multimodal_description
....

overall_soundscape
```

> Paragraphs with `0-5s` markers are split by time, unmarked paragraphs (styles/effects/prohibitions) are automatically combined into each window.

### Global Mode (global)

> The entire prompt applies to all segments, suitable for homogeneous actions in a single take throughout.

### Clip_Tag Mode (Segment by Tags)

> `clip_mode` set to `Clip_Tag`, `clip_tag` filled with tag template (must end with a numeric sequence number).

**Tag Template Examples**

| Template | Match |
|------|------|
| `段1` | `段1` / `段2` / `段3` (prefix "Segment"+number) |
| `A01` | `A01` / `A02` / `A03` (prefix "A"+number) |
| `[片段001]` | `[片段001]` / `[片段002]` (prefix "[Clip"+number+suffix"]") |

**Tag Writing**: Tags occupy a line as a separator, recommended to add a newline after the tag. Without newline, it can also be processed (separator skipped to take segment content):

```text
Segment 1:3s
Video:
...
Audio design:
...


Segment 2:3-8s
Video:
0-2s:
...
2-5s:
...
Audio design:
0-5s:...
```

**Segment Duration Rules** (three levels of priority):

1. Duration immediately following the tag line: `段1:0-5秒` → 5 seconds; `段1:3-8秒` → 5 seconds (duration markers are removed from the prompt)
2. In-segment time markers 0-based: `【0-2秒】`+`【2-5秒】` → 5 seconds
3. None → `chunk_frames / fps` as a fallback

**prompt_format Selection**

- `official` / `legacy`: In-segment time markers automatically converted to relative coordinates within the segment for rendering
- `raw`: Output as is after removing tags, time markers remain unchanged (suitable for structured prompts generated by large models)
## <a id="limitations"></a> 📝 Prompt Considerations (Node Limitations)

> The following considerations **do not apply** to simple, always-effective prompt scenarios (i.e., all segments share the same prompt, global mode),
> such as: voice-over digital humans (of course, lines need to be segmented), camera/composition changes are minimal in videos, or video character replacement, etc., common prompt scenarios.

### 1️⃣ Core Principle: Temporal Exclusivity

> When using segmented reasoning (Chunks), please strictly adhere to the **temporal exclusivity** principle—each segment's prompt can only describe the **new changes** that are "occurring" in that segment relative to the end of the previous segment.

- **Segments as "Relays"**: When generating the Nth segment, its starting frame state (position, action posture, camera position) is completely implicitly provided by the "anchored frames (Context Frames)" at the end of the previous segment. You don't need to repeat this starting state in the prompt.
- **Prohibited "Retrospection" and "Overlap"**: The prompt for the Nth segment absolutely cannot repeat actions or camera movements already completed in the N-1th segment. If repeated, the model will receive conflicting instructions with the anchored frame's visuals (instruction conflict), leading to jerky generation, illogical motion, or repeated actions.
- **Zeroing at Boundaries**: When switching segments, zero out the "ongoing actions" of the previous segment. The new segment's prompt should act like a "new instruction after taking a snapshot," targeting only the displacement, actions, or new elements occurring within the current new time period.

**❌ Incorrect Usage (Conflicting Overlap)**

```text
Segment 1: 3s
"Object A moves to position B"
Segment 2: 3-6s
"Object A moves to position B and then turns around at position B"
```

> Problem Analysis: At the end of the 1st segment, the anchored frame shows Object A has arrived at position B and just stopped. However, the 2nd segment's prompt forcibly requires "Object A moving to position B," which conflicts with the anchored frame's static result "already arrived." The model will attempt to "re-move," causing creepy or jump-cut effects.

**✅ Correct Usage (Seamless Progression)**

```text
Segment 1: 3s
"Object A moves to position B and stops finally at position B" (emphasizing action closure)
Segment 2: 3-6s
"Stand still, then Object A slowly turns direction" (directly describe the new action after the end of the previous segment)
```

> Correct Logic: The 2nd segment completely discards the description of the "movement process," assuming "stopped at B point" is a given fact, and only describes the subsequent "turning" new action. The model can then perfectly continue using the anchored frame.

> 🚀 **In a nutshell**: The end of the previous segment is the "result," the beginning of the next segment is the "new action after the result," don't put the "process that led to the result" into the next segment.

### 2️⃣ Core Principle: Per-Segment Reference Declaration

> When using segmented reasoning with reference images/videos (image1, video1, etc.), please strictly adhere to the **per-segment reference declaration** principle—each segment's prompt must independently and completely declare all reference materials required for that segment. References are not "memorized" or "inherited" to the next segment (only the referenced references participate in reasoning for the current segment).

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
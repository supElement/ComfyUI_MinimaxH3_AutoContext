<div align="center">

[![Chinese](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

One-click MiniMax H3 long video automated generation node: **segmented reasoning + inter-segment continuation anchoring + prompt timeline slicing + secondary sampling (2-sampling) + seam correction**.
Under limited GPU memory, split long videos into multiple independent reasoning segments, achieve seamless inter-segment concatenation through overlay enhancement methods, and automatically slice prompts along the timeline to align each segment's generated content with the prompt rhythm; perform the same slicing and alignment on audio-video references; only the referenced references in the current segment participate in reasoning. Supports secondary sampling. Video continuation, video frontward, dual video concatenation.
Supports latent cache storage and retrieval, making it convenient to quickly skip already reasoned segments if reasoning is interrupted for some reason. Cache files are stored per segment. If upstream parameters of the sampling node remain unchanged, existing latent cache files can be read.

Note: When changing the model, including LoRA, SageAttention, and other acceleration nodes, latent detection will not detect the changes, so latent caches must be deleted. There are two methods to delete latent caches:
- Enable the `clear_cache` parameter on the Minimax_H3_AutoContext_Sampler node, which will force the creation of a new cache file for this node at the start of sampling.
- Manually delete the corresponding folder in the cache directory (`\ComfyUI\output\cache`), with the folder name being "node_" + "node ID".

<img width="2230" height="976" alt="image" src="https://github.com/user-attachments/assets/5634914a-6f98-4d4f-b573-2c8b41e0c57e" />


<img width="2209" height="1030" alt="image" src="https://github.com/user-attachments/assets/9bbdda2a-d4ce-4836-b108-e359e72e31de" />

## BUG Fixes and Optimizations

V0.7.3

I. Attention Correction

<img width="1730" height="502" alt="image" src="https://github.com/user-attachments/assets/9744cdc5-d812-4498-9df8-99f6de141aa9" />

- H3TSTPatch（Minimax_H3_TST_AttentionPatch）
- Purpose: H3 Time State Transfer (TST) attention correction. Diagnose frame-level transfer states (over-mixing/fragmentation) through spectral tension, and perform per-head adaptive temperature scaling on video row queries
- Problem Solved: temporal flickering, small face collapse
- Connection: Model loading → This node → Sampling node; supports chain-connection with other attention patch nodes (must be placed downstream)
- Parameters: tau correction strength (0 = pass-through, can be used as A/B baseline; commonly used 0.1~0.3)

II. High-Resolution Local Face Repair Three-Step Pipeline (3 nodes)

<img width="1641" height="921" alt="image" src="https://github.com/user-attachments/assets/3d7fb8a4-0c5b-474c-b03b-2a40e40c67bf" />

- This is a "detection → redraw → reattach" face repair workflow (only supports single person):

- H3FaceCut（Minimax_H3_Face_Cut）— Step 1 of face repair
- Full latent decoding → Frame-by-frame YOLO face detection (model pulled from models/elementEasy directory) → Missing filling + sliding median smoothing → Fixed side length central cropping (ensuring the face is always centered, always the same size)

- H3FaceResample（Minimax_H3_Face_Resample）— Step 2 of face repair
- Resize the cropped sequence to a res² canvas using Lanczos → Refine the face using the H3 model for resampling → Decode output to the original resolution canvas. The sigmas port determines the number of resampling steps and denoising strength.

- H3FaceBlend（Minimax_H3_Face_Blend）— Step 3 of face repair
- Scale the redrawn canvas back to the cropped size S×S, and smoothly reattach it to the original image frame by frame, feathering the edges for blending.

III. Node annotations/prompts changed to bilingual (Chinese-English).

V0.7.2

- Fixed a bug where, when the input endpoint (total_frames / chunk_frames / context_frames) of the H3Parameter parameter node is connected to a node like Math Expression, the expected segmentation request enters an infinite loop, causing ComfyUI web page to freeze.

V0.7.1

- Added `video_guide` parameter, used to optimize video continuation, video frontward, dual video concatenation (generating intermediate segments), and supports segmentation. Note: When not set to none, the reference at the corresponding reference port of the sampling node will be forcibly cut to the value set in the `context_frames` parameter. The reference logic is the same as normal references (only if declared in the prompt will it be referenced).

V0.6.5

- Optimized latent cache processing logic, removing manual cache directory specification, and changed to automatically assign a unique cache directory to each node ("node + node ID") to prevent accidental operations from causing the latent cache logic of sampling nodes to overlap.
- Establish cache and validation logic in a segmented manner. If the upstream node only adds prompts or increases segmentation without changing other prompts submitted to sampling, and the other parameters associated with the sampling node remain unchanged, the existing corresponding cache is still considered valid and called, and new segments will automatically create latent caches. Downstream sampling nodes (2-sampling) will also retain existing latent caches and call them, only creating caches for newly added segments.
- The position of prompt changes determines which latent caches can be reused. Segments after the prompts that are changed will be forcibly rebuilt. Downstream nodes adopt the same processing logic.
- `ignore_latent_hash`, ignores the hash value check of the input port `input_latent`. Practical scenarios: Some latent processing nodes change the latent judgment information (e.g., Minimax H3 Latent Upscaler (3D) node), making slight changes in latent cause latent caches to become invalid, wasting reasoning time. In such cases, it is recommended to set it to true. I only tested the Minimax_H3-LatentUpscaler_Adv node in my other repository github.com/supElement/ComfyUI_Element_easy extension, similar nodes have not been tested. When using latent processing nodes that do not change latent noise characteristics, you can set the `ignore_latent_hash` parameter to false.

V0.5.8
- Improved hash value detection parameters to resolve errors caused by changes in parameter parameters of upstream nodes of the sampler, leading to tensor mismatches.
- Minimax_H3_Seam_Correction node, removed the shot detection model, as the detection model would cause the sampler node preview to display a "white screen". Replaced with PySceneDetect method (pure CPU, no potential contamination).
## 📖 Table of Contents

- [Node List](#nodes)
- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [Two-Stage Sampling & SplitSigmas Frequency](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Precautions (Node Limitations)](#limitations)

## <a id="nodes"></a> 🧩 Node List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter group node: Centralizes prompts/splits/resolution/audio parameters, outputs `parameter`, and provides real-time preview of "Estimated Splits" |
| **Minimax_H3_AutoContext_Sampler** | Main node: Split inference + anchor continuation + sampling (shared by one-stage and two-stage) |
| **Minimax_H3_Seam_Correction** | Seam correction node: Performs pixel-domain correction on inter-segment seams of decoded video |

> Usage: `parameter node --parameter--> Main node`. Prompts are filled in the parameter node, and the main node receives them via `parameter` (required).

## <a id="features"></a> ✨ Core Features

### 🧩 Split Inference

- Splits into multiple segments based on `total_frames` / `chunk_frames` (in frame units), recommended frame counts are 5, 22, 39, 56, 73, 90…
- Automatically pads the last segment to avoid excessively short tail segments
- `fps` is only used for audio synchronization and prompt second conversion

### 🔗 Inter-Segment Continuation

- **Overlay Enhancement**: Non-first segments automatically "take over" the ending frame of the previous segment, with new content naturally continuing from where the previous segment ended, eliminating pauses or position jumps at the seams
- The ending of the previous segment is used as motion reference for the current segment, helping to maintain motion direction and speed
- The audio of the previous segment is also passed as "previous content" to help the sound flow naturally
- Inter-segment audio fades smoothly, aligned with the number of video frames

> Frame count rules: `total_frames` / `chunk_frames` / `context_frames` all take 5, 22, 39, 56, 73, 90… (17n+5), the node aligns them automatically, generally no need for manual calculation.

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **Clip_Tag** | Splits prompts based on user-defined tags (e.g., `Segment1`/`Segment2`), each tag corresponds to an independent video segment; segment duration is determined by the prompt content (duration after tag > segment time markers > `total_frames/fps` fallback). |
| **timeline** | Splits prompts based on explicit time markers (e.g., `0-2s`/`2-6s`), each time range corresponds to a video segment; segment duration = range length × `fps` and automatically snaps to legal grid; **ignores `total_frames` and `chunk_frames`**, entirely determined by the prompt. Global segments (`【global】`) remain in their original positions and are not extracted together. |
| **sequential** | Distributes the prompt in order of sentence reading evenly across the entire video timeline without splitting the prompt itself; video segmentation still follows `chunk_frames`. | 
| **global** | The entire prompt is used for all video segments (after stripping `【global】` tags), video segmentation follows `chunk_frames`. |

> In `Clip_Tag` and `timeline` modes, `total_frames` and `chunk_frames` parameters are ignored (segment length determined by the prompt), only fallback to these values when the mode degrades (e.g., no tags/time markers detected).

### 🏷️ Clip_Tag Tag Splitting Mode

- Splits prompts based on user-defined tags (e.g., `Segment1`/`Segment2`/`Segment3`), each segment = one chunk = all prompts for that segment
- Segment duration is determined by the prompt content (three levels of priority):
  1. Duration immediately following the tag line (e.g., `Segment1:0-5s` → 5 seconds; `Segment1:3-8s` → 5 seconds)
  2. Maximum end value of time markers within the segment (e.g., `【0-2秒】`+`【2-5秒】` → 5 seconds)
  3. `total_frames / fps` default value as fallback (single segment aligns to `total_frames`)
- Segment time markers are **relative time** (starting from 0 for each segment), not global absolute time
- Overlapping frames are automatically generated for transitions in non-first segments and cropped after generation
- Total duration automatically aligns to the target total frames, trying to match the expected duration
- Labels are removed during inference, and the rest of the prompt content is output based on `prompt_format`

### 🎯 Reference Intelligent Filtering (Image / Video / Audio)

- Automatically identifies reference images/videos/audios used in each segment of the prompt and only passes the referenced materials to that segment
- Video and its paired audio track are bound together to avoid cross-talk between visuals and audio

### 🖌️ Two-Stage Sampling (Two-Stage)

- Main node `latent_input` receives one-stage latent (or via latent amplification node) to enter two-stage mode
- Two-stage resolution **takes input latent as reference** (ignores width/height), achieving low-quality one-stage → high-quality two-stage
- `denoise` controls redraw intensity; `sigmas` supports custom sigma sequences (same as `SamplerCustomAdvanced`)
- `lock_audio`: Two-stage only redraws video, reuses one-stage audio

### 🎵 Audio-Driven (Audio Drive)

- `drive_audio` (AUDIO, optional) + `audio_drive` switch
- Enabled, video generates following this audio, output audio = source audio itself (lip sync/rhythm driven by it)

## <a id="install"></a> 📦 Installation

### Method 1: Manual Installation (Manual Installation)

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
```

### Method 2: Install via Manager (Install using Manager)

Search for `ComfyUI_MinimaxH3_AutoContext` in the ComfyUI Manager and click Install.


## <a id="params"></a> ⚙️ Node Parameters

### Minimax_H3_AutoContext_parameter (Parameter Group Node)

| Parameter | Default Value | Description |
|------|--------|------|
| long_prompt | — | Prompt (passed to the main node for inference, also used for "Estimated Splits" preview) |
| **clip_mode** | `Clip_Tag` | How prompts map to video segments: `Clip_Tag` / `timeline` / `sequential` / `global`. In `Clip_Tag` and `timeline` modes, ignores `total_frames` and `chunk_frames`. |
| clip_tag | `Segment1` | Clip_Tag splitting tag template (must end with a numeric sequence number), only effective when `clip_mode=Clip_Tag` |
| prompt_format | `official` | Prompt output format: `official` / `legacy` / `raw`. `official` uses the official [Shot] format of MiniMax H3, `legacy` is the old-style time tag, `raw` outputs as-is (used for Clip_Tag mode) |
| crop_mode | `stretch` | Reference image/first/last frames/reference video scaling/cropping: `center` / `stretch` / `none` |
| ref_sync_mode | `segmented` | Whether reference video/audio is sliced per segment: `global` (uses the full material per segment) / `segmented` (slices by the time ratio of the segment) |
| width × height | 960×544 | One-stage resolution (latent_input overrides during two-stage) |
| total_frames | 362 | Total frames to generate (17n+5); only used as a fallback in `Clip_Tag`/`timeline` modes (when no tags/time markers), ultimately covered by the sum of each segment |
| fps | 24 | Frame rate, used for audio synchronization and prompt second conversion |
| chunk_frames | 90 | Frames per segment generated (17n+5), only effective in `sequential` / `global` modes |
| context_frames | 22 | Inter-segment continuation frames (17n+5: 5/22/39/56…), recommended 22 or higher |
| lock_audio | `true` | Lock audio region during two-stage (noise_mask audio=0): resamples video only, keeps one-stage audio unchanged |
| audio_drive | `false` | Audio drive switch, enables video to generate following `drive_audio` |
| video_guide | `none` | Video extension parameter, supports per-segment. none: disabled (does not modify video reference logic); pre_guide: video continuation (sample node ref_video_0 or + ref_video_audio_0 port); post_guide: video push forward (sample node ref_video_0 or + ref_video_audio_0 port); pre_post_guide: dual-video middle connection (sample node ref_video_0 or + ref_video_audio_0 port, ref_video_1 or + ref_video_audio_1 port). Anchor frame count determined by context_frames. Note: Non-none values force the reference port of the sampling node to be forcibly cropped to the value set in the context_frames parameter. Reference logic is the same as normal references (only referenced if declared in the prompt) |

> The node displays real-time "Estimated Splits" preview (calculated by frontend JS, does not participate in inference).

### Minimax_H3_AutoContext_Sampler (Main Node)

| Parameter | Default Value | Description |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 model components |
| parameter | Required | Parameter input (from parameter node) |
| sampler | Optional | External sampler object (SAMPLER), overrides built-in sampler_name/scheduler |
| sigmas | Optional | Custom sigma sequence (SIGMAS), highest priority |
| latent_input | Optional | Two-stage input latent (connected to enable two-stage) |
| info | Optional | Parameter inheritance input (multi-stage chaining, ensures segment consistency) |
| first_frame / last_frame | Optional | First/last frame anchoring (FL2VA) |
| video_context_denoise | 0.0 | Inter-segment continuation strength (only non-first segments): 0=exact continuation of the previous segment ending, 1=regenerate, intermediate values=soft mix. When connected to SplitSigmas in two-stage, set to 1 to avoid screen artifacts |
| seed | 0 | Random seed (control_after_generate) |
| steps / cfg | 30 / 1.0 | Sampling steps / CFG |
| sampler_name / scheduler | euler / simple | Built-in sampler / scheduler |
| denoise | 1.0 | Redraw strength (1=full resampling, smaller value retains more original structure) |
| enable_cache | true | Store/read latent cache, automatically creates a folder named "node+nodeID" in “\ComfyUI\output\cache” directory, and overrides existing latent cache files when upstream nodes or parameters change |
| clear_cache | false | Force rebuild latent cache files|
| ignore_latent_hash | false | Ignore hash validation of input port input_latent. Useful scenario: Some latent processing nodes alter latent judgment information, causing minor latent changes to result in cache incompatibility and waste inference time, suggesting setting to true |
| ref_image_N / ref_video_N / ref_video_audio_N / ref_audio_N | Optional | Reference materials (Autogrow dynamic ports) |
| drive_audio | Optional | Audio drive source |
## <a id="output"></a> 📤 Output

| Output | Description |
|------|------|
| **latent** | Concatenated audio-video latent, followed by VAE Decode, or upscaled then followed by binary sampling |
| **denoised_latent** | Clean latent output, used for binary sampling continuation / preview |
| **info** | Segmentation parameters (Dict), passed to the next main node's info input, ensuring consistent segmentation across multiple samples |



## <a id="second-pass"></a> 🔄 Binary Sampling and SplitSigmas High/Low Frequency

### Basic Binary Sampling (Low-Resolution First Sample → High-Resolution Second Sample)

```
parameter node ──parameter──> Main node (First Sample, 864×480)
    └─ latent / denoised_latent ──> [Separate AV] ──> video_latent ──> latent upscaled ──> [Combine AV] ──> Main node (Second Sample).latent_input
Binary Sampling node: parameter shared (or info inherited), optional denoise 0.4~0.6
```

- Binary sampling resolution is based on `latent_input`, ignoring parameter's width/height

### SplitSigmas High/Low Frequency (Save Time, Enhance Clarity)

> ⚠️ **Audio Constraint**: High/Low frequency **only affects video** (audio segments need complete sampling), audio should maintain complete sampling.

```
First Sample node: Complete sampling (no connection to high_sigmas, audio complete denoising)
          → denoised_latent → Separate amplify video (audio unchanged) → Combine → Second Sample.latent_input
Second Sample node: sigmas ← low_sigmas (only run low sigma segments to enhance details)
          lock_audio = True (reuse first sample complete audio)
          video_context_denoise = 1.0 (continuation area redraws together with new area, avoids screen tearing)
```

> 💡 **Second Sample `video_context_denoise`**: When connected to SplitSigmas, setting to 0 (precise continuation) may cause screen tearing at the boundary between continuation area and newly redrawn area; setting to 1.0 allows continuation area to redraw synchronously to avoid it. If the seam appears slightly discontinuous, it can be reduced to 0.3~0.5 for a compromise. First sample retains default 0.

## <a id="seam"></a> 🧵 Seam Correction Node (Minimax_H3_Seam_Correction)

| Parameter | Default Value | Description |
|------|--------|------|
| `fix_color_preset` | `"medium"` | **Color/Exposure Processing Level**<br>`off`：No processing; <br>`low`：Per-channel brightness gain, correction amount halved, most conservative, no color bias; <br>`medium`：Per-channel brightness gain, only corrects seam level jumps (recommended); <br>`high`：MKL linear color migration, longer statistical window, more stable during large motion; <br>`max`：Full frame brightness normalization across the entire clip, eliminates intra-segment gradient drift, but flattens the actual brightness variation in the frame (e.g., night scene/entering a tunnel), near black frames are ineffective (reported in logs). |
| `fix_motion_preset` | `"off"` | **Seam Continuity (Optical Flow Alignment+Blending) Level**<br>`off`：No processing (recommended to first observe effects with color level); <br>`low/medium/high/max`：Higher levels involve more frames and stronger blending, but may introduce slight blurring or breathing effects. |
| `fix_flash` | `false` | **Flash Processing** (instantaneous brightness jumps at boundaries). Independent switch, uses temporal fusion logic. If the scene has reasonable rapid brightness changes like lightning, explosions, suppression will flatten these effects. Even when `fix_motion_preset=off`, it can take effect independently. |
| `flash_threshold` | `0.30` | Transient correction selection threshold (percentage of anomalous pixels), smaller value is more aggressive (corrects more frames), recommended `0.20` ~ `0.40`. |
| `cut_threshold` | `15.0` | PySceneDetect's sensitivity threshold (range `5.0` ~ `50.0`), smaller value is more sensitive, recommended `10` ~ `20`. |
| `blend_frames` | `2` | Seam level gradient window (frames, 0~8): After exposure alignment, smooth the brightness transition of the `blend_frames` before and after the boundary as a smooth ramp; larger value results in smoother transition, more natural, but large motion scenes may cause slight blurring/breathing; `0` means disabled. |
| `use_gpu` | `true` | Use CUDA GPU for statistics, color transformation, and optical flow calculation (automatically fallback to CPU if unavailable). |

⚠️ Removed the scene detection model, using PySceneDetect (pure CPU, no potential contamination).

> Usage: `VAE Decode → H3_Seam_Correction → Save/Video`.

> ⚠️ Note: This node only performs visual seam correction, cannot fix artifacts generated by the upstream of binary sampling.

## <a id="prompt-examples"></a> ✍️ Prompt Writing Examples

### Timeline Mode (auto / timeline)

```text
0-5s: ...
5-10s: ...

integrated_multimodal_description
....

overall_soundscape
```

> Segments marked with `0-5s` are split by time, unmarked segments (styles/effects/prohibitions) are automatically combined into each window.

### Global Mode (global)

> The entire prompt applies to all segments, suitable for homogeneous actions in a single take throughout.

### Clip_Tag Mode (Segment by Tags)

> `clip_mode` set to `Clip_Tag`, `clip_tag` filled with tag template (must end with a numeric sequence number).

**Tag Template Examples**

| Template | Match |
|------|------|
| `Section 1` | `Section 1` / `Section 2` / `Section 3` (prefix "Section"+number) |
| `A01` | `A01` / `A02` / `A03` (prefix "A"+number) |
| `[Clip001]` | `[Clip001]` / `[Clip002]` (prefix "[Clip"+number+suffix "]") |

**Tag Writing**: Tags occupy a line as a separator, recommended to newline after the tag. Without newline, it can also be processed (skips separator to take segment content):

```text
段1:3s
Video:
...
Audio Design:
...


段2:3-8s
Video:
0-2 seconds:
...
2-5 seconds:
...
Audio Design:
0-5 seconds:...
```

**Segment Duration Rules** (three levels of priority):

1. Duration immediately following the tag line: `段1:0-5s` → 5 seconds; `段1:3-8s` → 5 seconds (duration markers will be removed from the prompt)
2. In-segment time markers 0-based: `【0-2秒】`+`【2-5秒】` → 5 seconds
3. None → `chunk_frames / fps` as fallback

**prompt_format Selection**

- `official` / `legacy`：Converts in-segment time markers to relative coordinates for rendering within the segment
- `raw`：Outputs the original prompt after removing tags, time markers remain unchanged (suitable for structured prompts generated by large models)
## <a id="limitations"></a> 📝 Prompt Precautions (Limitations of Nodes)

> The following precautions **do not apply** to simple, always-effective prompt scenarios (i.e., all segments share the same prompt, global mode),
> such as: voice-over digital humans (of course, lines need to be segmented), minimal changes in shots/construction in videos, or video character replacements, etc.

### 1️⃣ Core Principle: Temporal Exclusivity

> When using segmented reasoning (Chunks), please strictly adhere to the **temporal exclusivity** principle—each segment's prompt can only describe the **new changes** that are "occurring" in that segment relative to the end of the previous segment.

- **Segments as "Relays"**: When generating the Nth segment, its starting frame state (position, action posture, camera position) is fully implicitly provided by the "anchoring frames (Context Frames)" at the end of the previous segment. You don't need to repeat this starting state in the prompt.
- **Prohibited "Retrospection" and "Overlap"**: The prompt for the Nth segment absolutely cannot repeat actions or camera movements already completed in the N-1th segment. If repeated, the model will receive conflicting instructions with the anchoring frame's visual, leading to stuttered generation, illogical motion, or repeated actions.
- **Zeroing at Boundaries**: When switching segments, zero out the "ongoing actions" of the previous segment. The new segment's prompt should act as a "new command after taking a snapshot," targeting only the displacement, actions, or new elements that occur within the current new time period.

**❌ Incorrect Usage (Conflicting Overlap)**

```text
Segment 1: 3 seconds
"Object A moves to position B"
Segment 2: 3-6 seconds
"Object A moves to position B and then turns around at position B"
```

> Problem Analysis: When the 1st segment ends, the anchoring frame shows Object A has already reached position B and just stopped. However, the 2nd segment's prompt forcibly requires "Object A moving to position B," which conflicts with the anchoring frame's static result "already arrived." The model will attempt to "re-move" it, causing creepy effects or frame skipping.

**✅ Correct Usage (Seamless Progression)**

```text
Segment 1: 3 seconds
"Object A moves to position B and stops finally at position B" (emphasizing action closure)
Segment 2: 3-6 seconds
"Stand still, then Object A slowly turns direction" (directly describing the new action after the previous segment ends)
```

> Correct Logic: The 2nd segment completely discards the description of the "moving process," assuming "stopped at B point" is a given fact, and only describes the subsequent "turning" new action. The model can then perfectly continue using the anchoring frame.

> 🚀 **In a nutshell**: The end of the previous segment is the "result," and the start of the next segment is the "new action after the result." Don't put the "process that led to the result" into the next segment.

### 2️⃣ Core Principle: Per-Segment Reference Declaration

> When using segmented reasoning with reference images/videos (image1, video1, etc.), please strictly adhere to the **per-segment reference declaration** principle—each segment's prompt must independently and completely declare all the reference materials required for that segment. References are not "memorized" or "inherited" to the next segment (only the referenced references participate in reasoning for the current segment).

- **No Global Memory**: The node parses the reference labels explicitly mentioned within the current segment's prompt to accurately determine which materials are needed for that segment. Writing image1 in the previous segment only means it was used there; the next segment will re-scan.
- **If Not Written, Not Passed**: If the Nth segment doesn't re-write image1, that reference image won't be passed, leading to inconsistent characters/objects.

**❌ Incorrect Usage (Implicit Inheritance)**

```text
Segment 1[3s]: image1 is Object A, Object A is moving forward.
Segment 2[3-6s]: Object A stops, turns to look at the camera. (No image1 written)
```

**✅ Correct Usage (Explicit Per-Segment)**

```text
Segment 1[3s]: image1 is Object A, Object A is moving forward.
Segment 2[3-6s]: image1 is Object A, Object A stops, turns to look at the camera.
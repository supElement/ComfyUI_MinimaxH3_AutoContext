<div align="center">

[![Chinese](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

One-click MiniMax H3 long video automated generation node: **Segmented Reasoning + Inter-segment Anchor + Prompt Timeline Slicing + Secondary Sampling (2-Sample) + Seam Correction**.
In limited GPU memory, split long videos into multiple independent reasoning segments, achieve seamless inter-segment connection through overlay enhancement methods, and automatically slice prompts along the timeline, aligning generated content with prompt rhythm; perform the same slicing and alignment on audio/video references; only the referenced references in the current segment participate in reasoning. Supports secondary sampling. Video continuation, video forward, dual video connection.
Supports latent cache storage and retrieval, facilitating quick skipping of already reasoned segments if reasoning is interrupted for some reason, with cache files stored per segment. When upstream parameters of the sampling node remain unchanged, existing latent cache files can be read.

Note: Changing the model, including lora, sageattention, and other acceleration nodes, will not detect latent changes, so latent cache must be deleted. Two methods to delete latent cache:
- Enable the `clear_cache` parameter on the `Minimax_H3_AutoContext_Sampler` node, which forces the re-establishment of this node's cache file at the start of sampling.
- Manually delete the corresponding folder in the cache directory (`\ComfyUI\output\cache`), with the folder name being "node_" + "node ID".

<img width="2230" height="976" alt="image" src="https://github.com/user-attachments/assets/5634914a-6f98-4d4f-b573-2c8b41e0c57e" />


<img width="2209" height="1030" alt="image" src="https://github.com/user-attachments/assets/9bbdda2a-d4ce-4836-b108-e359e72e31de" />

## BUG Fixes and Optimizations

V0.7.2

- Fixed a bug where the expected segmentation request would enter an infinite loop when the input endpoint of the H3Parameter parameter node (total_frames / chunk_frames / context_frames) is connected to a similar Math Expression node, causing ComfyUI web interface to freeze.

V0.7.1

- Added `video_guide` parameter, used to optimize video continuation, video forward, and dual video connection (generating intermediate segments), supporting segmentation. Note: When non-none, the reference at the corresponding reference port of the sampling node will be forcibly cropped to the value set in the `context_frames` parameter. Reference logic is the same as normal references (only referenced if declared in prompts).

V0.6.5

- Optimized latent cache handling logic, removed manual cache directory specification, and automatically assigns a unique cache directory for each node ("node + node ID") to prevent accidental overlap of sampling node latent cache logic due to misoperation.
- Established and verified cache logic in a segmented manner. If the upstream node only adds prompts or increases segmentation without changing other prompts submitted to sampling, and other parameters associated with the sampling node remain unchanged, the existing corresponding cache is still considered valid and called, while new segments will automatically establish latent cache. Downstream sampling nodes (2-sample) will also retain and call the existing latent cache, only creating new cache for added segments.
- The position where prompts are changed determines which latent caches can be reused. Segments after prompts that are changed will be forcibly rebuilt, and downstream nodes adopt the same handling logic.
- `ignore_latent_hash`, ignores the hash value check of the input port `input_latent`. Practical scenario: Some latent processing nodes may change latent judgment information (e.g., `Minimax H3 Latent Upscaler (3D)` node), causing minor latent changes to make latent cache unusable and waste reasoning time. It is recommended to set this to true. I only tested the `Minimax_H3-LatentUpscaler_Adv` node in my other repository `github.com/supElement/ComfyUI_Element_easy` extension, similar nodes were not tested. When using latent processing nodes that do not change latent noise characteristics, you can set `ignore_latent_hash` parameter to false.

V0.5.8
- Improved hash value detection parameter to resolve tensor mismatch errors caused by changes in parameter parameters of upstream nodes of the sampler.
- Removed the shot detection model from the `Minimax_H3_Seam_Correction` node, as the detection model would cause the sampler node preview to show a "white screen". Replaced with PySceneDetect method (pure CPU, no potential contamination).

## 📖 Table of Contents

- [Node List](#nodes)
- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [2-Sample and SplitSigmas High/Low Frequency](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Precautions (Node Limitations)](#limitations)

## <a id="nodes"></a> 🧩 Node List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter group node: Centralizes management of prompts/segmentation/resolution/audio, outputs `parameter`, and provides real-time preview of "Expected Segments" |
| **Minimax_H3_AutoContext_Sampler** | Main node: Segmented reasoning + Anchor + Sampling (1-sample/2-sample shared) |
| **Minimax_H3_Seam_Correction** | Seam correction node: Performs pixel-domain seam correction on decoded video segments |

> Usage: `parameter node --parameter--> Main node`. Prompts are filled in the parameter node, and the main node receives `parameter` (required).

## <a id="features"></a> ✨ Core Features

### 🧩 Segmented Reasoning

- Split into multiple segments based on `total_frames` / `chunk_frames` (frame unit). Recommended frame counts: 5, 22, 39, 56, 73, 90…
- Automatically pads the last segment to avoid excessively short tail segments
- `fps` is only used for audio synchronization and prompt time conversion within seconds

### 🔗 Inter-segment Continuation

- **Overlay Enhancement**: Non-first segments automatically "take over" the ending frame of the previous segment, with new content naturally continuing from where the previous segment ended, eliminating pauses or position jumps at the seams
- The ending of the previous segment is used as motion reference for the current segment, helping to continue motion direction and speed
- Previous audio is also passed as "previous content" to help the sound continue naturally
- Inter-segment audio fades smoothly, aligned with the number of video frames

> Frame count rule: `total_frames` / `chunk_frames` / `context_frames` all take 5, 22, 39, 56, 73, 90… (17 times plus 5), the node will automatically align, generally no need for manual calculation.

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **Clip_Tag** | Slices prompts based on user-defined tags (e.g., `段1`/`段2`), each tag corresponds to an independent video segment; segment duration is determined by the prompt content (duration after tag > segment time markers > default `total_frames/fps`). |
| **timeline** | Slices prompts based on explicit time markers (e.g., `0-2s`/`2-6s`), each time interval corresponds to a video segment; segment duration = interval length × `fps` and automatically snaps to legal grid; **ignores `total_frames` and `chunk_frames`**, entirely determined by prompts for total duration. Global segments (`【Global】`) remain in their original positions and are not extracted together. |
| **sequential** | Distributes prompts in sentence order uniformly along the entire video timeline without splitting the prompts themselves; video segmentation still follows `chunk_frames`. | 
| **global** | Entire prompt is used for all video segments (after stripping `【Global】` tag), video segmentation follows `chunk_frames`. |

> In `Clip_Tag` and `timeline` modes, `total_frames` and `chunk_frames` parameters are ignored (segment length determined by prompts), only used when degraded (e.g., no tags/time markers detected) to fall back to these values.

### 🏷️ Clip_Tag Tag Segmentation Mode

- Slices prompts based on user-defined tags (e.g., `段1`/`段2`/`Segment 3`), each segment = one chunk = all prompts for that segment
- Segment duration determined by prompt content (three layers of priority):
  1. Duration immediately following the tag line (e.g., `段1:0-5s` → 5s; `段1:3-8s` → 5s)
  2. Maximum end value of time markers within the segment (e.g., `【0-2s】`+`【2-5s】` → 5s)
3. Default `total_frames / fps` fallback (matches `total_frames` for single segments)
- Segment time markers are **relative time** (starting from 0 for each segment), not global absolute time
- An overlap frame is automatically generated for non-first segments to ensure smooth connection, and is cropped after generation
- Total duration is automatically aligned to the target total frame count, as close as possible to the expected duration
- Labels themselves are removed during reasoning, and the rest of the prompt content is output based on `prompt_format`

### 🎯 Intelligent Reference Filtering (Image / Video / Audio)

- Automatically identifies reference images/videos/audios used in each segment's prompts and only passes referenced materials to that segment
- Video and its paired audio track are bound together to avoid visual/audio cross-talk

### 🖌️ Secondary Sampling (2-Sample)

- Main node `latent_input` receives 1-sample latent (or via latent amplification node) to enter 2-sample mode
- 2-sample resolution **takes input latent as reference** (ignores width/height), achieving low-quality 1-sample → high-quality 2-sample
- `denoise` controls redraw intensity; `sigmas` supports custom sigma sequences (same as `SamplerCustomAdvanced`)
- `lock_audio`: 2-sample only redraws video, reuses 1-sample audio

### 🎵 Audio Drive

- `drive_audio` (AUDIO, optional) + `audio_drive` switch
- Enabled, video follows this audio for generation, output audio = source audio itself (lip sync/rhythm driven by it)
## <a id="install"></a> 📦 Installation

### Method 1: Manual Installation

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
```

### Method 2: Install using Manager

Search for `ComfyUI_MinimaxH3_AutoContext` in ComfyUI Manager and click Install.


## <a id="params"></a> ⚙️ Node Parameters

### Minimax_H3_AutoContext_parameter（Parameter Group Node）

| Parameter | Default Value | Description |
|----------|---------------|------------|
| long_prompt | — | Prompt (passed to the main node for reasoning, also used for "Estimated Segmentation" preview) |
| **clip_mode** | `Clip_Tag` | How prompts map to video segments: `Clip_Tag` / `timeline` / `sequential` / `global`. In `Clip_Tag` and `timeline` modes, `total_frames` and `chunk_frames` are ignored. |
| clip_tag | `段1` | Clip_Tag segmentation tag template (must end with a numeric sequence number), only effective when `clip_mode=Clip_Tag` |
| prompt_format | `official` | Prompt output format: `official` / `legacy` / `raw`. `official` uses the official [Shot] format of MiniMax H3, `legacy` is the old-style time tag, `raw` outputs as-is (used for Clip_Tag mode) |
| crop_mode | `stretch` | Reference image/first/last frame/reference video scaling and cropping: `center` / `stretch` / `none` |
| ref_sync_mode | `segmented` | Whether reference video/audio is sliced per segment: `global` (uses the full material per segment) / `segmented` (slices by the time ratio of each segment) |
| width × height | 960×544 | Resolution of one sample (latent_input covers it when two-sampling) |
| total_frames | 362 | Total number of frames to generate (17n+5); only serves as a fallback in `Clip_Tag`/`timeline` modes, ultimately overridden by the sum of each segment |
| fps | 24 | Frame rate, used for audio synchronization and prompt second conversion |
| chunk_frames | 90 | Number of frames generated per segment (17n+5), only effective in `sequential` / `global` modes |
| context_frames | 22 | Frames for segment continuation (17n+5: 5/22/39/56…), recommended to be 22 or higher |
| lock_audio | `true` | Lock audio area during two-sampling (noise_mask audio=0): resamples video only, keeps one-sampling audio unchanged |
| audio_drive | `false` | Audio drive switch, after enabling, video follows drive_audio generation |
| video_guide | `none` | Video extension parameter, supports segmentation. none: disabled (does not modify video reference logic); pre_guide: video continuation (sample node ref_video_0 or + ref_video_audio_0 port); post_guide: video push forward (sample node ref_video_0 or + ref_video_audio_0 port); pre_post_guide: dual video middle connection (sample node ref_video_0 or + ref_video_audio_0 port, ref_video_1 or + ref_video_audio_1 port). The anchored frame count is determined by context_frames. Note: When not none, the reference of the sampling node's corresponding reference port will be forcibly cut to the value set in the context_frames parameter. The reference logic is the same as normal references (only reference if declared in the prompt). |

> The node displays "Estimated Segmentation" preview in real-time (calculated by frontend JS, does not participate in reasoning).

### Minimax_H3_AutoContext_Sampler（Main Node）

| Parameter | Default Value | Description |
|----------|---------------|------------|
| model / vae / audio_vae / clip | — | MiniMax H3 model components |
| parameter | Required | Parameter group input (from parameter node) |
| sampler | Optional | External sampler object (SAMPLER), overrides built-in sampler_name/scheduler |
| sigmas | Optional | Custom sigma sequence (SIGMAS), highest priority |
| latent_input | Optional | Two-sampling input latent (enables two-sampling upon connection) |
| info | Optional | Parameter inheritance input (multi-sampling chaining, ensures segment consistency) |
| first_frame / last_frame | Optional | First/last frame anchoring (FL2VA) |
| video_context_denoise | 0.0 | Segment continuation strength (only non-first segment): 0=exact continuation of the previous segment's end, 1=regenerate, intermediate values=soft blend. When connected to SplitSigmas for two-sampling, it is recommended to set to 1 to avoid screen artifacts |
| seed | 0 | Random seed (control_after_generate) |
| steps / cfg | 30 / 1.0 | Sampling steps / CFG |
| sampler_name / scheduler | euler / simple | Built-in sampler / scheduler |
| denoise | 1.0 | Redraw strength (1=full resampling, smaller value retains more original structure) |
| enable_cache | true | Store/read latent cache, automatically creates a folder named "node+nodeID" in the "\ComfyUI\output\cache" directory, and overrides existing latent cache files when upstream nodes or parameters change |
| clear_cache | false | Force rebuild latent cache file |
| ignore_latent_hash | false | Ignore hash validation of input port input_latent. Practical scenario: Some latent processing nodes change latent judgment information, causing minor latent changes to result in cache incompatibility and wasting reasoning time. It is recommended to set it to true in this case |
| ref_image_N / ref_video_N / ref_video_audio_N / ref_audio_N | Optional | Reference materials (Autogrow dynamic ports) |
| drive_audio | Optional | Audio drive source |
## <a id="output"></a> 📤 Output

| Output | Description |
|------|------|
| **latent** | Concatenated audio-video latent, followed by VAE Decode, or upscaled then followed by binary sampling |
| **denoised_latent** | Clean latent output, used for binary sampling continuation / preview |
| **info** | Segment parameters (Dict), passed to the next main node's info input, ensuring consistent multi-sampling segmentation |



## <a id="second-pass"></a> 🔄 Binary Sampling and SplitSigmas High/Low Frequencies

### Basic Binary Sampling (Low-Resolution First Sample → High-Resolution Second Sample)

```
parameter node ──parameter──> Main node (First Sample, 864×480)
    └─ latent / denoised_latent ──> [Separate AV] ──> video_latent ──> latent upscaled ──> [Combine AV] ──> Main node (Second Sample).latent_input
Binary Sampling node: parameter shared (or info inherited), optional denoise 0.4~0.6
```

- Binary sampling resolution is based on `latent_input`, ignoring parameter's width/height

### SplitSigmas High/Low Frequencies (Save Time, Enhance Clarity)

> ⚠️ **Audio Constraint**: High/Low frequencies only take effect on video (audio segments need complete sampling), keep audio as complete sampling.

```
First Sample node: Complete sampling (no connection to high_sigmas, audio complete denoising)
          → denoised_latent → Separate amplify video (audio unchanged) → Combine → Second Sample.latent_input
Second Sample node: sigmas ← low_sigmas (only run low sigma segments to enhance details)
          lock_audio = True (reuse first sample complete audio)
          video_context_denoise = 1.0 (continuation area redraws together with new area, avoid screen tearing)
```

> 💡 **Second Sample `video_context_denoise`**: When connected to SplitSigmas, setting to 0 (precise continuation) may cause screen tearing at the boundary between continuation area and newly redrawn area; setting to 1.0 allows continuation area to redraw synchronously to avoid it. If the seam appears slightly discontinuous, it can be reduced to 0.3~0.5 for a compromise. First sample keeps default 0.

## <a id="seam"></a> 🧵 Seam Correction Node (Minimax_H3_Seam_Correction)

| Parameter | Default Value | Description |
|------|--------|------|
| `fix_color_preset` | `"medium"` | **Color/Exposure Processing Level**<br>`off`：No processing; <br>`low`：Per-channel brightness gain, correction amount halved, most conservative, no color bias; <br>`medium`：Per-channel brightness gain, only corrects seam level jumps (recommended); <br>`high`：MKL linear color migration, longer statistical window, more stable during large motion; <br>`max`：Full frame brightness normalization across the entire video, eliminates intra-segment gradient drift, but flattens the actual brightness changes in the frame (e.g., night scene/entering a tunnel), near black frames are ineffective (reported in logs). |
| `fix_motion_preset` | `"off"` | **Seam Continuity (Optical Flow Alignment+Blending) Level**<br>`off`：No processing (recommended to first observe the effect with color level); <br>`low/medium/high/max`：Higher levels involve more frames and stronger blending, but may introduce slight blurring or breathing effects. |
| `fix_flash` | `false` | **Flash Processing** (instant brightness jumps at boundaries). Independent switch, uses temporal fusion logic. If the scene has reasonable rapid brightness changes like lightning, explosions, suppression may flatten these effects. Still effective even when `fix_motion_preset=off`. |
| `flash_threshold` | `0.30` | Transient correction selection threshold (percentage of abnormal pixels), smaller value is more aggressive (corrects more frames), recommended `0.20` ~ `0.40`. |
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

> Set `clip_mode` to `Clip_Tag`, fill `clip_tag` with the tag template (must end with a numeric sequence number).

**Tag Template Examples**

| Template | Match |
|------|------|
| `Section 1` | `Section 1` / `Section 2` / `Section 3` (prefix "Section"+number) |
| `A01` | `A01` / `A02` / `A03` (prefix "A"+number) |
| `[Clip001]` | `[Clip001]` / `[Clip002]` (prefix "[Clip"+number+suffix "]") |

**Tag Writing**: Tags occupy a line as a separator, recommended to add a newline after the tag. Without newline, it can still process (skips the separator to take segment content):

```text
段1:3s
Video:
...
Audio Design:
...


段2:3-8s
Video:
0-2s:
...
2-5s:
...
Audio Design:
0-5s:...
```

**Segment Duration Rules** (three levels of priority):

1. Duration immediately following the tag line: `段1:0-5s` → 5s; `段1:3-8s` → 5s (duration markers will be removed from the prompt)
2. In-segment time markers 0-based: `【0-2秒】`+`【2-5秒】` → 5s
3. None → `chunk_frames / fps` as fallback

**prompt_format Selection**

- `official` / `legacy`：Convert in-segment time markers to relative coordinates for rendering within the segment
- `raw`：Output the original untagged content, time markers remain unchanged (suitable for structured prompts generated by large models)
## <a id="limitations"></a> 📝 Prompt Precautions (Limitations of Nodes)

> The following precautions **do not apply** to simple, always-effective prompt scenarios (i.e., all segments share the same prompt, global mode),
> such as: voice-over digital humans (of course, lines need to be segmented), minimal changes in shots/construction in videos, or video character replacements, etc., common prompt scenarios.

### 1️⃣ Core Principle: Temporal Exclusivity

> When using segmented reasoning (Chunks), please strictly adhere to the **temporal exclusivity** principle—each segment's prompt can only describe the **new changes** that are "occurring" in that segment relative to the end of the previous segment.

- **Segments are "Relays"**: When generating the Nth segment, its starting visual state (position, action posture, camera position) is fully implicitly provided by the "anchoring frames (Context Frames)" at the end of the previous segment. You don't need to repeat this starting state in the prompt.
- **Prohibited "Retrospection" and "Overlap"**: The prompt for the Nth segment must absolutely not repeat the actions or camera movements already completed in the N-1th segment. If repeated, the model will receive conflicting instructions with the anchoring frame's visual (instruction conflict), leading to jerky generation, incorrect motion logic, or repeated actions.
- **Zeroing at Boundaries**: When switching segments, zero out the "ongoing actions" of the previous segment. The new segment's prompt should act like a "new instruction after taking a snapshot," targeting only the displacement, actions, or new elements that appear within the current new time period.

**❌ Incorrect Style (Conflicting Overlap)**

```text
段1: 3s
"Object A moves to position B"
段2: 3-6s
"Object A moves to position B and then turns around at position B"
```

> Problem Analysis: When the 1st segment ends, the anchoring frame shows Object A has already reached position B and just stopped. However, the 2nd segment's prompt forcibly requires "Object A moving to position B," which conflicts with the anchoring frame's static result "already arrived." The model will attempt to "re-move" it, causing creepy or frame-skipping effects.

**✅ Correct Style (Seamless Progression)**

```text
段1: 3s
"Object A moves to position B and finally stops at position B" (emphasizing action closure)
段2: 3-6s
"After standing firm, Object A slowly turns direction" (directly describing the new action after the previous segment ends)
```

> Correct Logic: The 2nd segment completely discards the description of the "movement process," assuming "stopped at B point" is a given fact, and only describes the subsequent "turning" new action. The model can then perfectly continue using the anchoring frame.

> 🚀 **In one sentence**: The end of the previous segment is the "result," and the beginning of the next segment is the "new action after the result." Don't put the "process that led to the result" into the next segment.

### 2️⃣ Core Principle: Per-Segment Reference Declaration

> When using segmented reasoning with reference images/videos (image1, video1, etc.), please strictly adhere to the **per-segment reference declaration** principle—each segment's prompt must independently and completely declare all the reference materials required for that segment. References are not "memorized" or "inherited" to the next segment (only the referenced references participate in reasoning for the current segment).

- **No Global Memory**: The node parses the reference labels explicitly mentioned in the current segment's prompt to accurately determine which materials are needed for that segment. Writing image1 in the previous segment only means it was used in the previous segment; the next segment will re-scan.
- **If Not Written, Not Passed**: If the Nth segment doesn't write image1 again, that reference image won't be passed to this segment, causing inconsistencies in characters/objects.

**❌ Incorrect Style (Implicit Inheritance)**

```text
段1[3s]: image1 is Object A, Object A is moving forward.
段2[3-6s]: Object A stops, turns to look at the camera. (No image1 written)
```

**✅ Correct Style (Explicit Per-Segment)**

```text
段1[3s]: image1 is Object A, Object A is moving forward.
段2[3-6s]: image1 is Object A, Object A stops, turns to look at the camera.

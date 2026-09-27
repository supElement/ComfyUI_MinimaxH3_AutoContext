<div align="center">

[![Chinese](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

One-click MiniMax H3 long video automated generation node: **segmented reasoning + inter-segment continuation anchoring + prompt timeline slicing + secondary sampling (2-sampling) + seam correction**.
In limited GPU memory, long videos are split into multiple independent reasoning segments. Seamless inter-segment connections are achieved through overlay enhancement methods, while prompts are automatically sliced along the timeline to align the generated content with the prompt rhythm for each segment. Audio and video references are similarly sliced and aligned. Only the referenced references in the current segment participate in reasoning. Supports secondary sampling. Video continuation, video forward, dual video connection.
Supports latent cache storage and retrieval, making it convenient to quickly skip already reasoned segments if reasoning is interrupted for some reason. Latent cache files are stored per segment. If the upstream parameters of the sampling node remain unchanged, existing latent cache files can be read.

Note: Changing the model, including LoRA, SageAttention, and other acceleration nodes, will not detect the changes in latent detection, so latent cache must be deleted. There are two ways to delete the latent cache:
- Enable the `clear_cache` parameter on the Minimax_H3_AutoContext_Sampler node, which will force the re-establishment of this node's cache file when sampling starts.
- Manually delete the corresponding folder in the cache directory (`\ComfyUI\output\cache`), with the folder name being "node_" + "node ID".

<img width="2230" height="976" alt="image" src="https://github.com/user-attachments/assets/5634914a-6f98-4d4f-b573-2c8b41e0c57e" />


<img width="2209" height="1030" alt="image" src="https://github.com/user-attachments/assets/9bbdda2a-d4ce-4836-b108-e359e72e31de" />

## BUG Fixes and Optimizations

v0.8.5

<img width="2156" height="629" alt="image" src="https://github.com/user-attachments/assets/b0b9373a-258b-485e-b5e8-0b3778f744e3" />

- Added and optimized the face repair node. Detailed instructions: [Chinese](h3_fix_zh.md) | [English](h3_fix_en.md)</sub>
- Minimax_H3_Face_Cut: Detection and cropping, storyboard + YOLO detection + optional SeC-4B tracking.
- Minimax_H3_Face_Resample: Refinement, uses the same model as primary sampling for block-level img2img resampling (block structure mirrors primary sampling segments + block-to-block anchoring).
- Minimax_H3_Face_Blend: Reattachment, refined faces are reattached to the original image pixel by pixel using geometric accounting and masks.

V0.7.2

- Fixed a bug where the request for segmented reasoning would enter an infinite loop when the input endpoint (total_frames / chunk_frames / context_frames) of the H3Parameter parameter node is connected to a node like Math Expression, causing ComfyUI to freeze.

V0.7.1

- Added `video_guide` parameter to optimize video continuation, video forward, and dual video connection (generating intermediate segments), supporting segmentation. Note: When non-None, the reference at the corresponding reference port of the sampling node will be forcibly cropped to the value set in the `context_frames` parameter. Reference logic is the same as normal references (only if declared in the prompt will it be referenced).

V0.6.5

- Optimized latent cache handling logic, removing manual cache directory specification, and automatically assigning a unique cache directory to each node ("node + node ID") to prevent accidental overlaps in sampling node latent cache logic.
- Established and verified cache logic in a segmented manner. If the upstream node only adds prompts or adds segmentation without changing other prompts submitted to sampling, and the other parameters associated with the sampling node remain unchanged, the existing corresponding cache is still considered valid and called. New segments will automatically establish latent cache. Downstream sampling nodes (2-sampling) will also retain existing latent cache and call it, only new added segment cache will be created.
- The position where the prompt is changed determines which latent caches can be reused. Segments after the changed prompt will be forcibly rebuilt. Downstream nodes adopt the same handling logic.
- `ignore_latent_hash`, ignores the hash value check of the input port `input_latent`. Practical scenarios: Some latent processing nodes change the latent judgment information (e.g., Minimax H3 Latent Upscaler (3D) node), causing minor latent changes to result in latent cache being invalid and wasting reasoning time. In such cases, it is recommended to set it to true. I only tested the Minimax_H3-LatentUpscaler_Adv node in my other repository github.com/supElement/ComfyUI_Element_easy extension, similar nodes were not tested. When using latent processing nodes that do not change latent noise characteristics, `ignore_latent_hash` parameter can be set to false.

V0.5.8
- Improved hash value detection parameter to resolve tensor mismatch errors caused by parameter changes in upstream nodes of the sampler.
- Minimax_H3_Seam_Correction node, removed the shot detection model, as the detection model would cause the sampler node preview to show a "white screen". Replaced with PySceneDetect method (pure CPU, no potential contamination).

## 📖 Table of Contents

- [Nodes List](#nodes)
- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [2-Sampling and SplitSigmas High/Low Frequency](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Considerations (Node Limitations)](#limitations)

## <a id="nodes"></a> 🧩 Nodes List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter group node: Centralizes management of prompts/segments/resolution/audio, outputs `parameter`, and provides real-time preview of "Expected Segments" |
| **Minimax_H3_AutoContext_Sampler** | Main node: Segmented reasoning + continuation anchoring + sampling (primary/secondary sampling shared) |
| **Minimax_H3_Seam_Correction** | Seam correction node: Performs pixel-domain correction on inter-segment seams of decoded video |

> Usage: `parameter node --parameter--> main node`. Prompts are filled in the parameter node, and the main node receives `parameter` (required).
## <a id="features"></a> ✨ Core Features

### 🧩 Segmented Reasoning

- Split into multiple segments based on `total_frames` / `chunk_frames` (in frames). Recommended frame counts are 5, 22, 39, 56, 73, 90…
- Automatically pad the last segment to avoid short tail segments
- `fps` is only used for audio synchronization and second conversion within prompts

### 🔗 Inter-Segment Continuation

- **Overlay Enhancement**: Non-first segments automatically "take over" the ending scene of the previous segment. New content naturally continues from where the previous segment ended, eliminating pauses or position jumps at the join
- The ending of the previous segment is passed as motion reference to the current segment, helping to maintain motion direction and speed
- The audio of the previous segment is also passed as "previous content" to help the sound flow naturally
- Audio between segments fades smoothly, aligned with the video frame count

> Frame count rule: `total_frames` / `chunk_frames` / `context_frames` all take 5, 22, 39, 56, 73, 90… (17n+5), and the node will automatically align, generally no need for manual calculation.

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **Clip_Tag** | Splits prompts based on user-defined tags (e.g., `段1`/`段2`). Each tag corresponds to an independent video segment; segment duration is determined by the prompt content (duration after tag > segment time markers > `total_frames/fps` fallback). |
| **timeline** | Splits prompts based on explicit time markers (e.g., `0-2s`/`2-6s`). Each time range corresponds to a video segment; segment duration = range length × `fps` and automatically snaps to legal grids; **ignores `total_frames` and `chunk_frames`**, completely determined by the prompt. Global segments (`【全局】`) remain in their original positions and are not extracted together. |
| **sequential** | Distributes prompts in sentence order evenly across the entire video timeline without splitting the prompts themselves; video segmentation still follows `chunk_frames`. | 
| **global** | The entire prompt is used for all video segments (after stripping `【全局】` tags), and video segmentation follows `chunk_frames`. |

> Under `Clip_Tag` and `timeline` modes, `total_frames` and `chunk_frames` parameters are ignored (segment length determined by prompts), only fallback to these values when the mode degrades (e.g., no tags/time markers detected).

### 🏷️ Clip_Tag Tagging Mode

- Splits prompts based on user-defined tags (e.g., `段1`/`段2`/`段3`), each segment = one chunk = all prompts in that segment
- Segment duration is determined by the prompt content (three priority levels):
  1. Duration immediately following the tag line (e.g., `段1:0-5秒` → 5 seconds; `段1:3-8秒` → 5 seconds)
  2. The maximum end value of time markers within the segment (e.g., `【0-2秒】`+`【2-5秒】` → 5 seconds)
  3. `total_frames / fps` as a fallback (for single segments, total frames fit `total_frames`)
- Segment time markers are **relative time** (starting from 0 for each segment), not global absolute time
- Overlapping frames are generated for non-first segments to ensure smooth transitions, and are automatically cropped after generation
- Total duration automatically aligns to the target total frames, trying to match the expected duration
- Tags themselves are removed during reasoning, and the rest of the prompt content is output based on `prompt_format`

### 🎯 Reference Intelligent Filtering (Image / Video / Audio)

- Automatically identifies reference images/videos/audios used in each segment's prompt, only passing the referenced materials to that segment
- Video and its paired audio track are bound together to avoid visual/audio cross-talk

### 🖌️ Secondary Sampling (Resampling)

- Main node `latent_input` receives one-sampled latent (or latent amplified node) to enter resampling mode
- Resampling resolution is based on the input latent (ignores width/height), achieving low-quality one-sample → high-quality resampling
- `denoise` controls redraw intensity; `sigmas` supports custom sigma sequences (same as `SamplerCustomAdvanced`)
- `lock_audio`: Resampling only redraws video, reuses one-sample audio

### 🎵 Audio-Driven (Audio Drive)

- `drive_audio` (AUDIO, optional) + `audio_drive` switch
- When enabled, video is generated following this audio, output audio = source audio itself (lip sync/rhythm driven by it)

## <a id="install"></a> 📦 Installation

### Method 1: Manual Installation (Manual Installation)

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
```

### Method 2: Install via Manager (Install using Manager)

Search for `ComfyUI_MinimaxH3_AutoContext` in ComfyUI Manager and click Install.


## <a id="params"></a> ⚙️ Node Parameters

### Minimax_H3_AutoContext_parameter (Parameter Group Node)

| Parameter | Default Value | Description |
|------|--------|------|
| long_prompt | — | Prompt (passed to the main node for reasoning, also used for "Estimated Segmentation" preview) |
| **clip_mode** | `Clip_Tag` | How prompts map to video segments: `Clip_Tag` / `timeline` / `sequential` / `global`. `Clip_Tag` and `timeline` modes ignore `total_frames` and `chunk_frames`. |
| clip_tag | `段1` | Clip_Tag segmentation tag template (must end with a numeric sequence number), only effective when `clip_mode=Clip_Tag` |
| prompt_format | `official` | Prompt output format: `official` / `legacy` / `raw`. `official` uses MiniMax H3's official [Shot] format, `legacy` is the old-style time tag, `raw` outputs as-is (used for Clip_Tag mode) |
| crop_mode | `stretch` | Reference image/first/last frame/reference video scaling/cropping: `center` / `stretch` / `none` |
| ref_sync_mode | `segmented` | Whether reference video/audio is sliced per segment: `global` (uses full material per segment) / `segmented` (slices based on segment time ratio) |
| width × height | 960×544 | One-sample resolution (overridden by latent_input during resampling) |
| total_frames | 362 | Total frames to generate (17n+5); only used as a fallback in `Clip_Tag`/`timeline` modes (when no tags/time markers), ultimately overridden by the sum of all segments |
| fps | 24 | Frame rate, used for audio synchronization and prompt second conversion |
| chunk_frames | 90 | Frames per segment generation (17n+5), only effective in `sequential` / `global` modes |
| context_frames | 22 | Inter-segment continuation frames (17n+5: 5/22/39/56…), recommended 22 or higher |
| lock_audio | `true` | Lock audio area during resampling (noise_mask audio=0): resample video only, keep one-sample audio unchanged |
| audio_drive | `false` | Audio drive switch, when enabled, video is generated following `drive_audio` |
| video_guide | `none` | Video extension parameter, supports per-segment. none: disabled; pre_guide: video continuation (sample node ref_video_0 or + ref_video_audio_0 port); post_guide: video push forward (sample node ref_video_0 or + ref_video_audio_0 port); pre_post_guide: dual video middle connection (sample node ref_video_0 or + ref_video_audio_0 port, ref_video_1 or + ref_video_audio_1 port). Anchored frame count determined by `context_frames`. Note: When not none, the corresponding reference port of the sampling node is forcibly cropped to the value set in the `context_frames` parameter. Reference logic is the same as normal references (only referenced if declared in the prompt) |

> The node displays "Estimated Segmentation" preview in real-time (calculated by frontend JS, does not participate in reasoning).

### Minimax_H3_AutoContext_Sampler (Main Node)

| Parameter | Default Value | Description |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 model components |
| parameter | Required | Parameter group input (from parameter node) |
| sampler | Optional | External sampler object (SAMPLER), overrides built-in sampler_name/scheduler |
| sigmas | Optional | Custom sigma sequence (SIGMAS), highest priority |
| latent_input | Optional | Resampling input latent (enables resampling upon connection) |
| info | Optional | Parameter inheritance input (for multi-sampling chaining, ensures segment consistency) |
| first_frame / last_frame | Optional | First/last frame anchoring (FL2VA) |
| video_context_denoise | 0.0 | Inter-segment continuation strength (only for non-first segments): 0=exact continuation of the previous segment's ending, 1=regenerate, intermediate values=soft mix. Recommended to set 1 when resampling with SplitSigmas to avoid screen artifacts |
| seed | 0 | Random seed (control_after_generate) |
| steps / cfg | 30 / 1.0 | Sampling steps / CFG |
| sampler_name / scheduler | euler / simple | Built-in sampler / scheduler |
| denoise | 1.0 | Redraw intensity (1=full resampling, smaller value preserves more original structure) |
| enable_cache | true | Store/read latent cache, automatically creates a folder named "node+node ID" in "\ComfyUI\output\cache" directory, and overrides existing latent cache files when upstream nodes or parameters change |
| clear_cache | false | Forcefully rebuild latent cache files|
| ignore_latent_hash | false | Ignore hash validation of input port input_latent. Useful scenarios: Some latent processing nodes alter latent judgment information, causing minor latent changes to make caching unavailable, wasting reasoning time. In such cases, it is recommended to set this to true |
| ref_image_N / ref_video_N / ref_video_audio_N / ref_audio_N | Optional | Reference materials (Autogrow dynamic ports) |
| drive_audio | Optional | Audio drive source |
## <a id="output"></a> 📤 Output

| Output | Description |
|------|------|
| **latent** | Latent of concatenated audio and video, connected to VAE Decode, or upscaled and then connected to binary sampling |
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

### SplitSigmas High/Low Frequency (Save Time, Improve Clarity)

> ⚠️ **Audio Constraint**: High/Low frequency **only takes effect on video** (audio segments need complete sampling), audio should maintain complete sampling.

```
First Sample node: Complete sampling (not connected to high_sigmas, audio complete denoising)
          → denoised_latent → Separate amplify video (audio unchanged) → Combine → Second Sample.latent_input
Second Sample node: sigmas ← low_sigmas (only run low sigma segments to enhance details)
          lock_audio = True (reuse first sample complete audio)
          video_context_denoise = 1.0 (continuation area redraws together with new area, avoid screen tearing)
```

> 💡 **Second Sample `video_context_denoise`**: When connected to SplitSigmas, setting to 0 (precise continuation) may cause screen tearing at the boundary between continuation area and newly redrawn area; setting to 1.0 allows continuation area to redraw synchronously to avoid it. If the seam appears slightly discontinuous, it can be reduced to 0.3~0.5 for a compromise. First sample retains default 0.

## <a id="seam"></a> 🧵 Seam Correction Node (Minimax_H3_Seam_Correction)

| Parameter | Default Value | Description |
|------|--------|------|
| `fix_color_preset` | `"medium"` | **Color/Exposure Processing Level**<br>`off`：No processing; <br>`low`：Per-channel brightness gain, correction amount halved, most conservative, no color bias; <br>`medium`：Per-channel brightness gain, only corrects seam level jumps (recommended); <br>`high`：MKL linear color migration, longer statistical window, more stable during large motion; <br>`max`：Full frame brightness normalization across the entire video, eliminates intra-segment gradient drift, but will flatten the actual brightness changes in the frame (e.g., night scene/entering a tunnel), near black frames are ineffective (reported in logs). |
| `fix_motion_preset` | `"off"` | **Seam Continuity (Optical Flow Alignment+Blending) Level**<br>`off`：No processing (recommended to first observe the effect with color level); <br>`low/medium/high/max`：Higher levels involve more frames and stronger blending, but may introduce slight blurring or breathing effects. |
| `fix_flash` | `false` | **Flash Processing** (instant brightness jumps at boundaries). Independent switch, uses temporal fusion logic. If the scene has reasonable rapid brightness changes like lightning, explosions, suppression will flatten these effects. Even when `fix_motion_preset=off`, it can still take effect independently. |
| `flash_threshold` | `0.30` | Transient correction selection threshold (percentage of abnormal pixels), smaller value is more aggressive (corrects more frames), recommended `0.20` ~ `0.40`. |
| `cut_threshold` | `15.0` | PySceneDetect's sensitivity threshold (range `5.0` ~ `50.0`), smaller value is more sensitive, recommended `10` ~ `20`. |
| `blend_frames` | `2` | Seam level gradient window (frames, 0~8): After exposure alignment, smooth the brightness transition of the `blend_frames` before and after the boundary as a smooth ramp; larger value results in smoother transition, more natural, but may cause slight blurring/breathing in highly dynamic shots; `0` means disabled. |
| `use_gpu` | `true` | Use CUDA GPU for statistics, color transformation, and optical flow calculation (automatically fallback to CPU if unavailable). |

⚠️ Removed the scene cut detection model, using PySceneDetect (pure CPU, no potential contamination).

> Usage: `VAE Decode → H3_Seam_Correction → Save/Video`.

> ⚠️ Note: This node only performs visual seam correction and cannot fix artifacts generated by the upstream binary sampling.

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

> The entire prompt is used for all segments, suitable for homogeneous actions in a single shot throughout.

### Clip_Tag Mode (Segment by Tags)

> Set `clip_mode` to `Clip_Tag`, fill `clip_tag` with a tag template (must end with a numeric sequence number).

**Tag Template Examples**

| Template | Match |
|------|------|
| `Section 1` | `Section 1` / `Section 2` / `Section 3` (prefix "Section"+number) |
| `A01` | `A01` / `A02` / `A03` (prefix "A"+number) |
| `[Clip001]` | `[Clip001]` / `[Clip002]` (prefix "[Clip"+number+suffix"]`) |

**Tag Writing**: Tags occupy a line as a separator, recommended to newline after the tag. Without newline, it can still be processed (skips separator to take segment content):

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

- `official` / `legacy`：Convert in-segment time markers to relative coordinates for rendering within the segment
- `raw`：Output the original prompt after removing tags, time markers remain unchanged (suitable for structured prompts generated by large models)
## <a id="limitations"></a> 📝 Prompt Considerations (Limitations of Nodes)

> The following considerations **do not apply** to simple, always-effective prompt scenarios (i.e., all segments share the same prompt, global mode),
> such as: voice-over digital humans (of course, lines need to be segmented), minimal changes in shots/construction in videos, or video character replacements, etc., common prompt scenarios.

### 1️⃣ Core Principle: Temporal Exclusivity

> When using segmented reasoning (Chunks), please strictly adhere to the **temporal exclusivity** principle—each segment's prompt can only describe the **new changes** occurring in that segment relative to the end of the previous segment.

- **Segments are "Relays"**: When generating the Nth segment, its starting visual state (position, action posture, camera position) is completely implicitly provided by the "anchored frames (Context Frames)" at the end of the previous segment. You don't need to repeat this starting state in the prompt.
- **Forbidden "Retrospection" and "Overlap"**: The prompt for the Nth segment absolutely cannot repeat actions or camera movements already completed in the N-1th segment. If repeated, the model receives conflicting instructions with the anchored frame's visual, causing jerky or logic errors in the generated visuals or repeated actions.
- **Zeroing at Boundaries**: When switching segments, zero out the "ongoing actions" of the previous segment. The new segment's prompt should act like a "new instruction after taking a snapshot," targeting only the displacement, actions, or new elements appearing within the current new time segment.

**❌ Incorrect Style (Conflict Overlap)**

```text
Segment 1: 3 seconds
"Object A moves to position B"
Segment 2: 3-6 seconds
"Object A moves to position B and then turns at position B"
```

> Problem Analysis: At the end of the 1st segment, the anchored frame shows Object A has arrived at position B and just stopped. However, the 2nd segment's prompt forcibly demands "Object A moves to position B," conflicting with the anchored frame's static result "already arrived." The model attempts to "re-move," causing glitches or frame skipping.

**✅ Correct Style (Seamless Progression)**

```text
Segment 1: 3 seconds
"Object A moves to position B and finally stops at position B" (emphasizing action closure)
Segment 2: 3-6 seconds
"Stand firm, then Object A slowly turns direction" (directly describing the new action after the previous segment ends)
```

> Correct Logic: The 2nd segment completely discards the description of the "movement process," assuming "stopped at B point" is a given fact, and only describes the subsequent "turning" new action. The model can then perfectly continue using the anchored frame.

> 🚀 **In a nutshell**: The end of the previous segment is the "result," and the start of the next segment is the "new action after the result." Don't put the "process that led to the result" into the next segment.

### 2️⃣ Core Principle: Per-Segment Reference Declaration

> When using segmented reasoning with reference images/videos (image1, video1, etc.), please strictly adhere to the **per-segment reference declaration** principle—each segment's prompt must independently and completely declare all required reference materials for that segment. References are not "memorized" or "inherited" to the next segment (only the referenced references participate in reasoning for the current segment).

- **No Global Memory**: The node parses the reference labels explicitly mentioned in the current segment's prompt to accurately determine which materials are needed for that segment. Writing image1 in the previous segment only means it was used there; the next segment will re-scan.
- **If Not Written, Not Passed**: If the Nth segment doesn't write image1 again, that reference image won't be passed, causing inconsistencies in characters/objects.

**❌ Incorrect Style (Implicit Inheritance)**

```text
Segment 1[3 seconds]: image1 is Object A, Object A is moving forward.
Segment 2[3-6 seconds]: Object A stops, turns to look at the camera. (No image1 written)
```

**✅ Correct Style (Explicit Per-Segment)**

```text
Segment 1[3 seconds]: image1 is Object A, Object A is moving forward.
Segment 2[3-6]: image1 is Object A, Object A stops, turns to look at the camera.
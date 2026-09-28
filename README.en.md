<div align="center">

[![Chinese](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

One-click MiniMax H3 long video automated generation node: **segmented reasoning + inter-segment continuation anchoring + prompt timeline slicing + secondary sampling (2-sampling) + seam correction**.
Under limited GPU memory, split long videos into multiple independent reasoning segments, achieve seamless inter-segment connection through overlay enhancement methods, and automatically slice prompts along the timeline to align each segment's generated content with the prompt rhythm; perform the same slicing and alignment on audio-video references; only the referenced references in the current segment participate in reasoning. Supports secondary sampling. Video continuation, video forward, dual video connection.
Supports latent cache storage and retrieval, making it convenient to quickly skip already reasoned segments if reasoning is interrupted for some reason. Cache files are stored per segment. If the upstream sampling node parameters remain unchanged, existing latent cache files can be read.

Note: Changing the model, including LoRA, SageAttention, and other acceleration nodes, will not detect the change in latent detection, so latent cache must be deleted. There are two ways to delete the latent cache:
- Enable the `clear_cache` parameter on the Minimax_H3_AutoContext_Sampler node, which will force the re-establishment of this node's cache file when sampling starts.
- Manually delete the corresponding folder in the cache directory (`\ComfyUI\output\cache`), with the folder name being "node_" + "node ID".

<img width="2230" height="976" alt="image" src="https://github.com/user-attachments/assets/5634914a-6f98-4d4f-b573-2c8b41e0c57e" />


<img width="2209" height="1030" alt="image" src="https://github.com/user-attachments/assets/9bbdda2a-d4ce-4836-b108-e359e72e31de" />

## BUG Fixes and Optimizations

v0.8.5 

> **New node in `test` branch**
>
> First installation (to test branch):  
>   `git clone -b test https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git`  
> Already installed, switching from main to test:  
>   `git fetch origin` → `git checkout test` → `git pull`  
> Manager users: Switch the branch to `test` in Manager, then click Update.

<img width="2156" height="629" alt="image" src="https://github.com/user-attachments/assets/b0b9373a-258b-485e-b5e8-0b3778f744e3" />  <br>
  
Added Minimax_H3_TST_AttentionPatch attention correction node; H3 TST attention correction — spectral tension diagnosis + video line query adaptive scaling, suppress temporal flicker/smashed face. tau intensity (0.2), needs to be placed downstream of other attention patches.  

Added and optimized facial repair node. Detailed instructions: [Chinese](https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext/blob/test/h3_fix_zh.md) | [English](https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext/blob/test/h3_fix_en.md)</sub>
- Minimax_H3_Face_Cut: Detection and cropping, storyboard + YOLO detection + optional SeC-4B tracking.
- Minimax_H3_Face_Resample: Refinement, uses the same sampling model for block-level img2img resampling (block structure mirrors the main sampling segments + block-to-block anchoring).
- Minimax_H3_Face_Blend: Reintegration, refined faces are pasted back into the original scene pixel by pixel using geometric accounting and masks.

V0.7.2

- Fixed a bug where the expected segment requests would enter an infinite loop when the input endpoints (total_frames / chunk_frames / context_frames) of the H3Parameter parameter node are connected to nodes like Math Expression, causing ComfyUI to freeze.

V0.7.1

- Added `video_guide` parameter, for optimizing video continuation, video forward, and dual video connection (generating intermediate segments), supports segmentation. Note: When non-none, the reference at the corresponding reference port of the sampling node will be forcibly cut to the value set in the `context_frames` parameter. Reference reference logic is the same as normal references (only references if declared in the prompt).

V0.6.5

- Optimized latent cache processing logic, removed manual cache directory specification, and changed to automatically assign a unique cache directory to each node ("node + node ID"), preventing accidental overlap of sampling node latent cache logic.
- Established cache and validation logic in a segmented manner. If the upstream node only adds prompts or increases segmentation without changing other prompts submitted to sampling, and the other parameters associated with the sampling node remain unchanged, the existing corresponding cache is still considered valid and called, and new segments will automatically establish latent cache. Downstream sampling nodes (2-sampling) will also retain existing latent cache and call it, only new added segment cache will be created.
- The position of prompt changes determines which latent caches can be reused. Segments after the changed prompts will be forcibly rebuilt, and downstream nodes adopt the same processing logic.
- `ignore_latent_hash`, ignores the hash value check of the input port `input_latent`. Practical scenario: Some latent processing nodes change latent judgment information (e.g., Minimax H3 Latent Upscaler (3D) node), making tiny changes in latent cause latent cache to become invalid, wasting reasoning time. In such cases, it is recommended to set it to true. I only tested another repository of mine `github.com/supElement/ComfyUI_Element_easy` extension's Minimax_H3-LatentUpscaler_Adv node, similar nodes were not tested. When using latent processing nodes that do not change latent noise characteristics, you can set `ignore_latent_hash` parameter to false.

V0.5.8
- Improved hash value detection parameter to resolve tensor mismatch errors caused by parameter changes in upstream nodes of the sampler.
- Minimax_H3_Seam_Correction node, removed the shot detection model, as the detection model would cause the sampler node preview to show a "white screen", replaced with PySceneDetect method (pure CPU, no potential contamination).
## 📖 Table of Contents

- [Node List](#nodes)
- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [Two-Stage Sampling and SplitSigmas Frequency](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Precautions (Node Limitations)](#limitations)

## <a id="nodes"></a> 🧩 Node List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter group node: Centralizes prompts/splitting/resolution/audio parameters, outputs `parameter`, and provides real-time preview of "Estimated Splitting" |
| **Minimax_H3_AutoContext_Sampler** | Main node: Splitting inference + anchor continuation + sampling (shared by one-stage and two-stage sampling) |
| **Minimax_H3_Seam_Correction** | Seam correction node: Performs pixel-domain correction on inter-segment seams of decoded video |

> Usage: `parameter node --parameter--> main node`. Prompts are filled in the parameter node, and the main node receives `parameter` (required) through.

## <a id="features"></a> ✨ Core Features

### 🧩 Splitting Inference

- Splits into multiple segments based on `total_frames` / `chunk_frames` (frame units), frame count recommended: 5, 22, 39, 56, 73, 90…
- Automatically pads the last segment to avoid excessively short tail segments
- `fps` is only used for audio synchronization and prompt second conversion

### 🔗 Inter-segment Continuation

- **Overlay Enhancement**: Non-first segments automatically "take over" the ending frame of the previous segment, with new content naturally continuing from where the previous segment ended, eliminating pauses or position jumps at the seams
- The ending frame of the previous segment is used as a motion reference for the current segment, helping to maintain the direction and speed of motion
- The audio of the previous segment is also passed in as "previous content" to help the sound continue naturally
- Inter-segment audio fades smoothly and aligns with the video frame count

> Frame count rule: `total_frames` / `chunk_frames` / `context_frames` all take 5, 22, 39, 56, 73, 90… (17n+5), the node automatically aligns, generally no need for manual calculation.

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **Clip_Tag** | Splits prompts based on user-defined tags (e.g., `Segment1`/`Segment2`), each tag corresponds to an independent video segment; segment duration is determined by the prompt content (duration after tag > segment time markers > `total_frames/fps` fallback). |
| **timeline** | Splits prompts based on explicit time markers (e.g., `0-2s`/`2-6s`), each time interval corresponds to a video segment; segment duration = interval length × `fps` and automatically snaps to legal grids; **ignores `total_frames` and `chunk_frames`**, completely determined by the prompt. Global segments (`【Global】`) remain in their original positions and are not extracted together. |
| **sequential** | Distributes the prompt in order of sentence reading across the entire video timeline without splitting the prompt itself; video segmentation still follows `chunk_frames`. | 
| **global** | The entire prompt is used for all video segments (after stripping `【Global】` tags), video segmentation follows `chunk_frames`. |

> In `Clip_Tag` and `timeline` modes, `total_frames` and `chunk_frames` parameters are ignored (segment length determined by the prompt), only fallback to these values when the mode degrades (e.g., no tags/time markers detected).

### 🏷️ Clip_Tag Tag Splitting Mode

- Splits prompts based on user-defined tags (e.g., `Segment1`/`Segment2`/`Segment3`), each segment = one chunk = all prompts for that segment
- Segment duration is determined by the prompt content (three levels of priority):
  1. Duration immediately following the tag line (e.g., `Segment1:0-5s` → 5s; `Segment1:3-8s` → 5s)
  2. Maximum end value of time markers within the segment (e.g., `【0-2s】`+`【2-5s】` → 5s)
  3. `total_frames / fps` default value fallback (single segment total frames fit `total_frames`)
- Time markers within the segment are **relative time** (starting from 0 for each segment), not global absolute time
- Overlapping frames are automatically generated for transitions in non-first segments and cropped after generation
- Total duration automatically aligns to the target total frame count, as close as possible to the expected duration
- Labels are removed during inference, and the remaining prompt content is output based on `prompt_format`

### 🎯 Reference Intelligent Filtering (Image / Video / Audio)

- Automatically identifies reference images/videos/audios used in each segment of the prompt and only passes the referenced materials to that segment
- Video and its paired audio track are bound together to avoid cross-talk between visuals and sound

### 🖌️ Two-Stage Sampling (Two-Stage Sampling)

- Main node `latent_input` inputs one-stage latent (or latent amplified node) to enter two-stage sampling mode
- Two-stage resolution **takes input latent as the reference** (ignores width/height), achieving low-quality one-stage → high-quality two-stage sampling
- `denoise` controls redraw intensity; `sigmas` supports custom sigma sequences (same as `SamplerCustomAdvanced`)
- `lock_audio`: Two-stage sampling only redraws video, reuses one-stage audio

### 🎵 Audio Drive

- `drive_audio` (AUDIO, optional) + `audio_drive` switch
- Enabled, video follows this audio for generation, output audio = source audio itself (lip sync/rhythm driven by it)

## <a id="install"></a> 📦 Installation

### Method 1: Manual Installation

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
```

### Method 2: Install via Manager

Search `ComfyUI_MinimaxH3_AutoContext` in ComfyUI Manager and click Install.

## <a id="params"></a> ⚙️ Node Parameters

### Minimax_H3_AutoContext_parameter (Parameter Group Node)

| Parameter | Default Value | Description |
|------|--------|------|
| long_prompt | — | Prompt (passed to main node for inference, also used for "Estimated Splitting" preview) |
| **clip_mode** | `Clip_Tag` | How prompts map to video segments: `Clip_Tag` / `timeline` / `sequential` / `global`. `Clip_Tag` and `timeline` modes ignore `total_frames` and `chunk_frames`. |
| clip_tag | `Segment1` | Clip_Tag splitting tag template (must end with a numeric sequence number), only effective when `clip_mode=Clip_Tag` |
| prompt_format | `official` | Prompt output format: `official` / `legacy` / `raw`. `official` uses the official MiniMax H3 [Shot] format, `legacy` is the old-style time tag, `raw` outputs as-is (used in Clip_Tag mode) |
| crop_mode | `stretch` | Reference image/first/last frames/reference video scaling/cropping: `center` / `stretch` / `none` |
| ref_sync_mode | `segmented` | Whether reference video/audio is sliced per segment: `global` (uses full material per segment) / `segmented` (slices by segment time ratio) |
| width × height | 960×544 | One-stage resolution (latent_input overrides when two-stage) |
| total_frames | 362 | Total frames to generate (17n+5); only used as a fallback in `Clip_Tag`/`timeline` modes (when no tags/time markers), ultimately covered by the sum of each segment |
| fps | 24 | Frame rate, used for audio synchronization and prompt second conversion |
| chunk_frames | 90 | Frames generated per segment (17n+5), only effective in `sequential` / `global` modes |
| context_frames | 22 | Inter-segment continuation frames (17n+5: 5/22/39/56…), recommended 22 or higher |
| lock_audio | `true` | Lock audio region during two-stage sampling (noise_mask audio=0): resamples video only, keeps one-stage audio unchanged |
| audio_drive | `false` | Audio drive switch, enables video to follow drive_audio for generation |
| video_guide | `none` | Video extension parameter, supports per-segment. none: disabled (does not modify video reference logic); pre_guide: video continuation (ref_video_0 or + ref_video_audio_0 port on sampling node); post_guide: video pre-push (ref_video_0 or + ref_video_audio_0 port on sampling node); pre_post_guide: dual-video middle connection (ref_video_0 or + ref_video_audio_0 port on sampling node, ref_video_1 or + ref_video_audio_1 port on sampling node). Anchor frame count determined by context_frames. Note: When not none, the reference of the corresponding port on the sampling node is forcibly cropped to the value set in the context_frames parameter. Reference logic is the same as normal references (only referenced if declared in the prompt) |

> The node dynamically displays "Estimated Splitting" preview (calculated by frontend JS, does not participate in inference).

### Minimax_H3_AutoContext_Sampler (Main Node)

| Parameter | Default Value | Description |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 model components |
| parameter | Required | Parameter group input (from parameter node) |
| sampler | Optional | External sampler object (SAMPLER), overrides built-in sampler_name/scheduler |
| sigmas | Optional | Custom sigma sequence (SIGMAS), highest priority |
| latent_input | Optional | Two-stage sampling input latent (enables two-stage sampling upon connection) |
| info | Optional | Parameter inheritance input (multi-sampling chaining, ensures segment consistency) |
| first_frame / last_frame | Optional | First/last frame anchoring (FL2VA) |
| video_context_denoise | 0.0 | Inter-segment continuation strength (only non-first segments): 0=exact continuation of previous segment ending, 1=regenerate, intermediate values=soft blend. When connected to SplitSigmas in two-stage sampling, set to 1 to avoid screen artifacts |
| seed | 0 | Random seed (control_after_generate) |
| steps / cfg | 30 / 1.0 | Sampling steps / CFG |
| sampler_name / scheduler | euler / simple | Built-in sampler / scheduler |
| denoise | 1.0 | Redraw strength (1=full resampling, smaller value retains more original structure) |
| enable_cache | true | Store/read latent cache, automatically creates a folder named "node+nodeID" in "\ComfyUI\output\cache" directory, latent cache files are overwritten when upstream nodes or parameters change |
| clear_cache | false | Forcefully rebuild latent cache files |
| ignore_latent_hash | false | Ignore hash validation of input port input_latent. Useful scenario: Some latent processing nodes alter latent judgment information, causing minor latent changes to result in cache incompatibility and waste inference time, in which case it is recommended to set to true |
| ref_image_N / ref_video_N / ref_video_audio_N / ref_audio_N | Optional | Reference materials (Autogrow dynamic ports) |
| drive_audio | Optional | Audio drive source |
## <a id="output"></a> Output

| Output | Description |
|------|------|
| **latent** | Latent of audio and video after concatenation, connected to VAE Decode, or upscaled and then followed by binary sampling |
| **denoised_latent** | Clean latent output, used for binary sampling continuation / preview |
| **info** | Segmentation parameters (Dict), passed to the next main node's info input, ensuring consistent segmentation across multiple samples |



## <a id="second-pass"></a> 🔄 Second Pass and SplitSigmas High/Low Frequencies

### Basic Binary Sampling (Low-Resolution First Sample → High-Resolution Second Sample)

```
parameter 节点 ──parameter──> 主节点(一采, 864×480)
    └─ latent / denoised_latent ──> [分离 AV] ──> video_latent ──> latent 放大 ──> [合并 AV] ──> 主节点(二采).latent_input
二采节点: parameter 共用 (或 info 继承)，可选 denoise 0.4~0.6
```

- Second sample resolution is based on `latent_input`, ignores parameter's width/height

### SplitSigmas High/Low Frequencies (Save Time, Increase Clarity)

> ⚠️ **Audio Constraint**: High and low frequencies only take effect on video (audio segments need complete sampling), audio should maintain complete sampling.

```
一采节点: 完整采样 (不接 high_sigmas，audio 完整去噪)
          → denoised_latent → 分离放大 video (audio 不动) → 合并 → 二采.latent_input
二采节点: sigmas ← low_sigmas (只跑低 sigma 段提细节)
          lock_audio = True (复用一采完整音频)
          video_context_denoise = 1.0 (续接区随新增区一起重绘，避免花屏)
```

> 💡 **Second sample `video_context_denoise`**: When connected to SplitSigmas, if set to 0 (precise continuation), the continuation area and the newly redrawn area may flicker at the boundary; set to 1.0 to let the continuation area redraw synchronously to avoid it. If the seam is slightly discontinuous, it can be reduced to 0.3~0.5 for a compromise. First sample remains default 0.

## <a id="seam"></a> 🧵 Seam Correction Node (Minimax_H3_Seam_Correction)

| Parameter | Default Value | Description |
|------|--------|------|
| `fix_color_preset` | `"medium"` | **Color/Exposure Processing Level**<br>`off`: No processing;<br>`low`: Per-channel brightness gain, correction amount halved, most conservative, no color bias;<br>`medium`: Per-channel brightness gain, only corrects seam level jumps (recommended);<br>`high`: MKL linear color migration, longer statistical window, more stable during large motion;<br>`max`: Frame-by-frame brightness normalization across the entire clip, eliminates intra-segment gradient drift, but will flatten the actual brightness changes in the frame (e.g., sky darkening/entering a tunnel), invalid for near-black frames (reported in logs). |
| `fix_motion_preset` | `"off"` | **Seam Continuity (Optical Flow Alignment+Blending) Level**<br>`off`: No processing (recommended to first observe the effect with color level);<br>`low/medium/high/max`: The higher the level, the more frames and intensity participate in blending, but it may introduce slight blurring or breathing effects. |
| `fix_flash` | `false` | **Flash Frame Processing** (instant brightness jumps at boundaries). Independent switch, uses temporal fusion logic. If the scene has reasonable rapid brightness changes like lightning, explosions, etc., suppression will flatten these effects. Still effective independently even when `fix_motion_preset=off` is active. |
| `flash_threshold` | `0.30` | Transient correction selection threshold (percentage of abnormal pixels), smaller value is more aggressive (corrects more frames), recommended `0.20` ~ `0.40`. |
| `cut_threshold` | `15.0` | PySceneDetect's sensitivity threshold (range `5.0` ~ `50.0`), smaller value is more sensitive, recommended `10` ~ `20`. |
| `blend_frames` | `2` | Seam level transition window (frames, 0~8): After exposure alignment, smooth the brightness transition of `blend_frames` frames before and after the boundary as a smooth ramp; larger value results in smoother, more natural transitions, but may cause slight blurring/breathing effects during large motion scenes; `0` means disabled. |
| `use_gpu` | `true` | Use CUDA GPU for statistics, color transformation, and optical flow calculation (automatically fallback to CPU if unavailable). |

⚠️ Removed the scene detection model, using PySceneDetect (pure CPU, no potential contamination).

> Usage: `VAE Decode → H3_Seam_Correction → Save/Video`.

> ⚠️ Note: This node only performs visual seam correction and cannot fix artifacts generated by the upstream second sample.

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

> The entire prompt is used for all segments, suitable for homogeneous actions throughout the entire shot.

### Clip_Tag Mode (Segment by Tags)

> `clip_mode` set to `Clip_Tag`, `clip_tag` fill in the tag template (must end with a numeric sequence number).

**Tag Template Examples**

| Template | Match |
|------|------|
| `Section 1` | `Section 1` / `Section 2` / `Section 3` (prefix "Section"+number) |
| `A01` | `A01` / `A02` / `A03` (prefix "A"+number) |
| `[Clip001]` | `[Clip001]` / `[Clip002]` (prefix "[Clip"+number+suffix"]) |

**Tag Writing**: Tags occupy a line as a separator, recommended to newline after the tag. Without newline, it can also be processed (skips separator to take segment content):

```text
段1:3s
视频：
...
音频设计：
...


段2:3-8s
视频：
0-2秒：
...
2-5秒：
...
音频设计：
0-5秒：...
```

**Segment Duration Rules** (three levels of priority):

1. Duration immediately following the tag line: `段1:0-5s` → 5 seconds; `段1:3-8s` → 5 seconds (duration markers will be removed from the prompt)
2. In-segment time markers 0-based: `【0-2s】`+`【2-5s】` → 5 seconds
3. None → `chunk_frames / fps` fallback

**prompt_format Selection**

- `official` / `legacy`: In-segment time markers automatically converted to relative coordinates for rendering within the segment
- `raw`: Output as is after removing tags, time markers remain unchanged (suitable for structured prompts generated by large models)
## <a id="limitations"></a> 📝 Prompt Precautions (Limitations of Nodes)

> The following precautions **do not apply** to simple, always-effective prompt scenarios (i.e., all segments share the same prompt, global mode),
> such as: voice-over digital humans (of course, the lines need to be segmented), minimal changes in shots/construction in videos, or video character replacements, etc., common prompt scenarios.

### 1️⃣ Core Principle: Temporal Exclusivity

> When using segmented reasoning (Chunks), please strictly adhere to the **temporal exclusivity** principle—each segment's prompt can only describe the **new changes** that are "occurring" in that segment relative to the end of the previous segment.

- **Segments as "Relays"**: When generating the Nth segment, its starting frame state (position, action posture, camera position) is completely implicitly provided by the "anchoring frames (Context Frames)" at the end of the previous segment. You don't need to repeat this starting state in the prompt.
- **Prohibited "Retrospection" and "Overlap"**: The prompt for the Nth segment absolutely cannot repeat actions or camera movements already completed in the N-1th segment. If repeated, the instructions received by the model will conflict with the anchoring frame's visuals (instruction conflict), causing the generated visuals to stutter, logical movement errors, or action repetition.
- **Zeroing at Boundaries**: When switching segments, zero out the "ongoing actions" of the previous segment. The new segment's prompt should act like a "new instruction after pressing the shutter," targeting only the displacement, actions, or new elements that appear within the current new time segment.

**❌ Incorrect Writing (Conflict Overlap)**

```text
段1：3秒
"物体 A 向位置 B 移动"
段2：3-6秒
"物体 A 移动到位置 B 后，正在位置 B 转身"
```

> Problem Analysis: At the end of the 1st segment, the anchoring frame shows Object A has arrived at position B and just stopped. However, the 2nd segment's prompt forcibly requires "Object A moving to position B," which conflicts with the anchoring frame's static result "already arrived." The model will attempt to "re-move" it, causing uncanny valley or frame skipping.

**✅ Correct Writing (Seamless Progression)**

```text
段1：3秒
"物体 A 向位置 B 移动，并最终停在位置 B"（强调动作闭环）
段2：3-6秒
"站稳后，物体 A 缓慢转动方向"（直接描述上一段结束后的新动作）
```

> Correct Logic: The 2nd segment completely discards the description of the "movement process," assuming "stopped at B point" is a given fact, and only describes the subsequent "turning" new action. The model can then perfectly continue using the anchoring frame.

> 🚀 **In a nutshell**: The end of the previous segment is the "result," and the beginning of the next segment is the "new action after the result." Don't put the "process that led to the result" into the next segment.

### 2️⃣ Core Principle: Per-Segment Reference Declaration

> When using segmented reasoning with reference images/videos (image1, video1, etc.), please strictly adhere to the **per-segment reference declaration** principle—each segment's prompt must independently and completely declare all the reference materials required for that segment. References are not "memorized" or "inherited" to the next segment (only the referenced references participate in reasoning for the current segment).

- **No Global Memory**: The node parses the reference labels explicitly mentioned within the current segment's prompt to accurately determine which materials are needed for that segment. Writing image1 in the 1st segment only means it was used in the 1st segment; the next segment will re-scan.
- **No Write, No Transfer**: If the Nth segment doesn't write image1 again, that reference image won't be passed in, causing inconsistency in characters/objects.

**❌ Incorrect Writing (Implicit Inheritance)**

```text
段1[3秒]：image1 是物体A，物体A正在向前移动。
段2[3-6秒]：物体A停下，转身看向镜头。（没写 image1）
```

**✅ Correct Writing (Explicit Per-Segment)**

```text
段1[3秒]：image1 是物体A，物体A正在向前移动。
段2[3-6]：image1 是物体A，物体A停下，转身看向镜头。
```
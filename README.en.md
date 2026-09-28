<div align="center">

[![Chinese](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

One-click MiniMax H3 long video automated generation node: **segmented reasoning + inter-segment continuation anchoring + prompt timeline slicing + secondary sampling (two-sampling) + seam correction**.
In limited GPU memory, split long videos into multiple independent reasoning segments, achieve seamless inter-segment connection through overlay enhancement methods, and automatically slice prompts along the timeline to align each segment's generated content with the prompt rhythm; perform the same slicing and alignment on audio-video references; only the referenced references in the current segment participate in reasoning. Supports secondary sampling. Video continuation, video forward, dual video connection.
Supports latent cache storage and retrieval, making it convenient to quickly skip already reasoned segments if reasoning is interrupted for some reason, with cache files stored per segment. When upstream parameters of the sampling node remain unchanged, existing latent cache files can be read.

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
> Already installed, switch from main to test:
>   `git fetch origin` → `git checkout test` → `git pull`  
> Manager users: Switch branch to `test` in Manager, then click Update.

<img width="2156" height="629" alt="image" src="https://github.com/user-attachments/assets/b0b9373a-258b-485e-b5e8-0b3778f744e3" />  <br>
  
Added Minimax_H3_TST_AttentionPatch attention correction node; H3 TST attention correction — spectral tension diagnosis + video line query adaptive scaling, suppress temporal flicker/smashed face. tau intensity (0.2), needs to be placed downstream of other attention patches.

Added and optimized facial repair node. Detailed instructions: [Chinese](h3_fix_zh.md) | [English](h3_fix_en.md)</sub>
- Minimax_H3_Face_Cut: Detection and cropping, storyboard + YOLO detection + optional SeC-4B tracking.
- Minimax_H3_Face_Resample: Refinement, uses the same main sampling model for block-level img2img resampling (block structure mirrors main sampling segments + block-to-block anchoring) .
- Minimax_H3_Face_Blend: Reintegration, refined faces are pasted back into the original scene according to geometric ledger and mask per pixel.

V0.7.2

- Fixed a bug where the expected segment requests would enter an infinite loop when the input endpoint (total_frames / chunk_frames / context_frames) of the H3Parameter parameter node is connected to a node like Math Expression, causing ComfyUI web interface to freeze.

V0.7.1

- Added `video_guide` parameter, used to optimize video continuation, video forward, and dual video connection (generating intermediate segments), supporting segmentation. Note: When non-none, the reference at the corresponding reference port of the sampling node will be forcibly cut to the value set in the `context_frames` parameter. Reference logic is the same as normal references (only if declared in the prompt will it be referenced).

V0.6.5

- Optimized latent cache processing logic, removed manual cache directory specification, changed to automatically assign a unique cache directory to each node ("node + node ID"), preventing accidental overlap of sampling node latent cache logic.
- Establish cache and validation logic in a segmented manner. If the upstream node only adds prompts or adds segmentation without changing other prompts submitted to sampling, and the other parameters associated with the sampling node remain unchanged, then the existing corresponding cache is still considered valid and called, and new segments will automatically establish latent cache. Downstream sampling nodes (two-sampling) will also retain existing latent cache and call it, only new added segment cache will be created.
- The position of prompt changes determines which latent caches can be reused. Segments after the prompts that are changed will be forcibly rebuilt, and downstream nodes adopt the same processing logic.
- `ignore_latent_hash`, ignores the hash value check of the input port `input_latent`. Practical scenario: Some latent processing nodes change latent judgment information (e.g., Minimax H3 Latent Upscaler (3D) node), making slight changes in latent cause latent cache to be invalid, wasting reasoning time. In this case, it is recommended to set it to true. I only tested the Minimax_H3-LatentUpscaler_Adv node in my other repository github.com/supElement/ComfyUI_Element_easy extension, similar nodes have not been tested. When using latent processing nodes that do not change latent noise characteristics, you can set `ignore_latent_hash` parameter to false.

V0.5.8
- Improved hash value detection parameter to resolve tensor mismatch errors caused by changes in parameter parameters of upstream nodes of the sampler.
- Minimax_H3_Seam_Correction node, removed shot detection model, as the detection model would cause the sampler node preview to show "white screen", replaced with PySceneDetect method (pure CPU, no potential contamination).

## 📖 Directory

- [Node List](#nodes)
- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [Two-Sampling and SplitSigmas High/Low Frequency](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Precautions (Node Limitations)](#limitations)
## <a id="nodes"></a> 🧩 Node List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter group node: Centralizes prompts/splitting/resolution/audio parameters, outputs `parameter`, and provides real-time preview of "Estimated Splitting" |
| **Minimax_H3_AutoContext_Sampler** | Main node: Splitting inference + Anchor continuation + Sampling (one-sampler/two-sampler shared) |
| **Minimax_H3_Seam_Correction** | Seam correction node: Performs pixel-domain correction on inter-segment seams of decoded video |

> Usage: `parameter node --parameter--> Main node`. Prompts are filled in the parameter node, and the main node receives `parameter` (required).

## <a id="features"></a> ✨ Core Features

### 🧩 Splitting Inference

- Splits into multiple segments based on `total_frames` / `chunk_frames` (frame units), recommended frame counts are 5, 22, 39, 56, 73, 90…
- Automatically pads the last segment to avoid overly short tail segments
- `fps` is only used for audio synchronization and prompt second conversion

### 🔗 Inter-segment Continuation

- **Overlay Enhancement**: Non-first segments automatically "take over" the ending frame of the previous segment, with new content naturally continuing from the end of the previous segment and eliminating pauses or position jumps at the seam
- The ending of the previous segment is used as motion reference for the current segment, helping to continue motion direction and speed
- The audio of the previous segment is also passed in as "previous content" to help the sound continue naturally
- Inter-segment audio fades smoothly and aligns with the video frame count

> Frame count rules: `total_frames` / `chunk_frames` / `context_frames` all take 5, 22, 39, 56, 73, 90… (17n+5), the node aligns automatically, generally no manual calculation is needed.

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **Clip_Tag** | Splits prompts based on user-defined tags (e.g., `Segment1`/`Segment2`), each tag corresponds to an independent video segment; segment duration is determined by the prompt content (duration after tag > segment time markers > `total_frames/fps` fallback). |
| **timeline** | Splits prompts based on explicit time markers (e.g., `0-2s`/`2-6s`), each time range corresponds to a video segment; segment duration = range length × `fps` and automatically snaps to legal grid; **ignores `total_frames` and `chunk_frames`**, completely determined by the prompt. Global segments (`【Global】`) remain in their original positions and are not centrally extracted. |
| **sequential** | Distributes prompts in sentence order evenly across the entire video timeline without splitting the prompt itself; video segmentation still follows `chunk_frames`. | 
| **global** | The entire prompt is used for all video segments (after stripping `【Global】` markers), video segmentation follows `chunk_frames`. |

> In `Clip_Tag` and `timeline` modes, `total_frames` and `chunk_frames` parameters are ignored (segment length determined by prompt), only fallback to these values when the mode degrades (e.g., no tags/time markers detected).

### 🏷️ Clip_Tag Tag Splitting Mode

- Splits prompts based on user-defined tags (e.g., `Segment1`/`Segment2`/`Segment3`), each segment = one chunk = all prompts for that segment
- Segment duration is determined by the prompt content (three priority levels):
  1. Duration immediately following the tag line (e.g., `Segment1:0-5s` → 5 seconds; `Segment1:3-8s` → 5 seconds)
  2. Maximum end value of time markers within the segment (e.g., `【0-2秒】`+`【2-5秒】` → 5 seconds)
3. `total_frames / fps` default fallback (matches `total_frames` for single segments)
- Segment time markers are **relative time** (starting from 0 for each segment), not global absolute time
- Overlapping frames are automatically generated for non-first segments to ensure smooth transitions, and are cropped after generation
- Total duration automatically aligns to the target total frames, trying to match the expected duration
- Labels are removed during inference, while the rest of the prompt content is output based on `prompt_format`

### 🎯 Reference Intelligent Filtering (Image / Video / Audio)

- Automatically identifies reference images/videos/audios used in each segment prompt, only passing referenced materials to that segment
- Video is bound to its paired audio track to avoid visual/audio cross-talk

### 🖌️ Secondary Sampling (Two-Sample)

- Main node `latent_input` receives one-sampled latent (or latent amplified node) to enter two-sample mode
- Two-sample resolution **takes input latent as reference** (ignores width/height), achieving low-quality one-sample → high-quality two-sample
- `denoise` controls redraw intensity; `sigmas` supports custom sigma sequences (same as `SamplerCustomAdvanced`)
- `lock_audio`: Two-sample only redraws video, reuses one-sample audio

### 🎵 Audio Drive

- `drive_audio` (AUDIO, optional) + `audio_drive` switch
- Enabled, video is generated following this audio, output audio = source audio itself (lip sync/rhythm driven by it)

## <a id="install"></a> 📦 Installation

### Method 1: Manual Installation

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
```

### Method 2: Install via Manager

Search for `ComfyUI_MinimaxH3_AutoContext` in ComfyUI Manager and click Install.

## <a id="params"></a> ⚙️ Node Parameters

### Minimax_H3_AutoContext_parameter (Parameter Group Node)

| Parameter | Default Value | Description |
|------|--------|------|
| long_prompt | — | Prompt (passed to main node for inference, also used for "Estimated Splitting" preview) |
| **clip_mode** | `Clip_Tag` | How prompts map to video segments: `Clip_Tag` / `timeline` / `sequential` / `global`. `Clip_Tag` and `timeline` modes ignore `total_frames` and `chunk_frames`. |
| clip_tag | `Segment1` | Clip_Tag splitting tag template (must end with a numeric sequence number), only effective when `clip_mode=Clip_Tag` |
| prompt_format | `official` | Prompt output format: `official` / `legacy` / `raw`. `official` uses MiniMax H3 official [Shot] format, `legacy` is the old-style time tag, `raw` outputs as-is (used for Clip_Tag mode) |
| crop_mode | `stretch` | Reference image/first/last frames/reference video scaling/cropping: `center` / `stretch` / `none` |
| ref_sync_mode | `segmented` | Whether reference video/audio is sliced per segment: `global` (uses full material per segment) / `segmented` (slices by segment time ratio) |
| width × height | 960×544 | One-sample resolution (overridden by latent_input during two-sample) |
| total_frames | 362 | Total frames to generate (17n+5); only used as fallback in `Clip_Tag`/`timeline` modes (when no tags/time markers), ultimately overridden by the sum of segments |
| fps | 24 | Frame rate, used for audio synchronization and prompt second conversion |
| chunk_frames | 90 | Frames per segment generated (17n+5), only effective in `sequential` / `global` modes |
| context_frames | 22 | Inter-segment continuation frames (17n+5: 5/22/39/56…), recommended 22 or higher |
| lock_audio | `true` | Lock audio region during two-sample (noise_mask audio=0): redrawing video only, keeping one-sample audio unchanged |
| audio_drive | `false` | Audio drive switch, enables video generation following `drive_audio` |
| video_guide | `none` | Video extension parameter, supports per-segment. none: disabled (does not modify video reference logic); pre_guide: video continuation (sampler node ref_video_0 or + ref_video_audio_0 port); post_guide: video push-forward (sampler node ref_video_0 or + ref_video_audio_0 port); pre_post_guide: dual-video middle connection (sampler node ref_video_0 or + ref_video_audio_0 port, ref_video_1 or + ref_video_audio_1 port). Anchor frame count determined by `context_frames`. Note: Non-none values force the corresponding reference port of the sampling node to be cropped to the value set in `context_frames`. Reference logic is the same as normal references (only referenced if declared in the prompt) |

> The node displays "Estimated Splitting" preview in real-time (calculated by frontend JS, does not participate in inference).

### Minimax_H3_AutoContext_Sampler (Main Node)

| Parameter | Default Value | Description |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 model components |
| parameter | Required | Parameter group input (from parameter node) |
| sampler | Optional | External sampler object (SAMPLER), overrides built-in sampler_name/scheduler |
| sigmas | Optional | Custom sigma sequence (SIGMAS), highest priority |
| latent_input | Optional | Two-sample input latent (enables two-sample upon connection) |
| info | Optional | Parameter inheritance input (multi-sampling chaining, ensures segment consistency) |
| first_frame / last_frame | Optional | First/last frame anchoring (FL2VA) |
| video_context_denoise | 0.0 | Inter-segment continuation strength (only non-first segments): 0=exact continuation of previous segment ending, 1=regenerate, intermediate values=soft mix. Recommended to set 1 when connected to SplitSigmas to avoid screen artifacts |
| seed | 0 | Random seed (control_after_generate) |
| steps / cfg | 30 / 1.0 | Sampling steps / CFG |
| sampler_name / scheduler | euler / simple | Built-in sampler / scheduler |
| denoise | 1.0 | Redraw intensity (1=full resampling, smaller value preserves more original structure) |
| enable_cache | true | Store/read latent cache, automatically creates a folder named "node+nodeID" under “\ComfyUI\output\cache”, latent cache files are overwritten when upstream nodes or parameters change |
| clear_cache | false | Force rebuild latent cache files |
| ignore_latent_hash | false | Ignore hash validation of input port input_latent. Useful scenarios: Some latent processing nodes alter latent judgment information, causing minor latent changes to make cache unusable, wasting inference time, suggest setting to true in such cases |
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

> ⚠️ **Audio Constraint**: High/Low frequency **only affects video** (audio segments need complete sampling), audio should maintain complete sampling.

```
First Sample node: Complete sampling (not connected to high_sigmas, audio complete denoising)
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
| `fix_motion_preset` | `"off"` | **Seam Continuity (Optical Flow Alignment+Blending) Level**<br>`off`：No processing (recommended to first observe the effect with color level); <br>`low/medium/high/max`：Higher levels involve more frames and stronger blending, but may introduce slight blurring or breathing effects. |
| `fix_flash` | `false` | **Flash Processing** (instant brightness jumps at boundaries). Independent switch, uses temporal fusion logic. If the scene has reasonable rapid brightness changes like lightning, explosions, suppression will flatten these effects. Even when `fix_motion_preset=off`, it can take effect independently. |
| `flash_threshold` | `0.30` | Transient correction selection threshold (percentage of anomalous pixels), smaller value is more aggressive (corrects more frames), recommended `0.20` ~ `0.40`. |
| `cut_threshold` | `15.0` | PySceneDetect's sensitivity threshold (range `5.0` ~ `50.0`), smaller value is more sensitive, recommended `10` ~ `20`. |
| `blend_frames` | `2` | Seam level gradient window (frames, 0~8): After exposure alignment, smooth the brightness transition of the `blend_frames` before and after the boundary as a smooth ramp; larger value results in smoother transition, more natural, but large motion scenes may cause slight blurring/breathing; `0` means disabled. |
| `use_gpu` | `true` | Use CUDA GPU for statistics, color transformation, and optical flow calculation (automatically fallback to CPU if unavailable). |

⚠️ Removed the scene detection model, using PySceneDetect (pure CPU, no potential contamination).

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

> The entire prompt is used for all segments, suitable for homogeneous actions throughout the entire shot.

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

- `official` / `legacy`：Convert in-segment time markers to relative coordinates for rendering within the segment
- `raw`：Output the original prompt after removing tags, time markers remain unchanged (suitable for structured prompts generated by large models)
## <a id="limitations"></a> 📝 Prompt Precautions (Limitations of Nodes)

> The following precautions **do not apply** to simple, always-effective prompt scenarios (i.e., all segments share the same prompt, global mode),
> such as: voice-over digital humans (of course, lines need to be segmented), minimal changes in shots/construction in videos, or video character replacements, etc.

### 1️⃣ Core Principle: Temporal Exclusivity

> When using segmented reasoning (Chunks), please strictly adhere to the **temporal exclusivity** principle—each segment's prompt can only describe the **new changes** that are "occurring" in that segment relative to the end of the previous segment.

- **Segments as "Relays"**: When generating the Nth segment, its starting frame state (position, action posture, camera position) is entirely implicitly provided by the "anchoring frames (Context Frames)" at the end of the previous segment. You do not need to repeat describe this starting state in the prompt.
- **Prohibited "Retrospection" and "Overlap"**: The prompt for the Nth segment absolutely cannot repeat describe actions or camera movements already completed in the N-1th segment. If repeated, the model will receive conflicting instructions with the anchoring frame's visuals (instruction conflict), leading to jerky generation, illogical motion, or repeated actions.
- **Zeroing at Boundaries**: When switching segments, zero out the "ongoing actions" of the previous segment. The new segment's prompt should act as a "new instruction after taking a snapshot," targeting only the displacement, actions, or new elements that occur within the current new time period.

**❌ Incorrect Writing (Conflicting Overlap)**

```text
Segment 1: 3 seconds
"Object A moves to position B"
Segment 2: 3-6 seconds
"Object A moves to position B and then turns at position B"
```

> Problem Analysis: When the 1st segment ends, the anchoring frame shows Object A has arrived at position B and just stopped. However, the 2nd segment's prompt forcibly requires "Object A moves to position B," which conflicts with the anchoring frame's static result "already arrived," causing the model to attempt "restarting the movement," leading to creepy or frame-skipping effects.

**✅ Correct Writing (Seamless Progression)**

```text
Segment 1: 3 seconds
"Object A moves to position B and stops at position B" (emphasizing action closure)
Segment 2: 3-6 seconds
"Stand firm, then Object A slowly turns direction" (directly describe the new action after the end of the previous segment)
```

> Correct Logic: The 2nd segment completely discards the description of the "movement process," assuming "stopped at B point" is a given fact, and only describes the subsequent "turning" new action, allowing the model to perfectly continue using the anchoring frame.

> 🚀 **In a nutshell**: The end of the previous segment is the "result," and the start of the next segment is the "new action after the result." Don't put the "process that led to the result" into the next segment.

### 2️⃣ Core Principle: Per-Segment Referenced Declaration

> When using segmented reasoning with reference images/videos (image1, video1, etc.), please strictly adhere to the **per-segment referenced declaration** principle—each segment's prompt must independently and completely declare all the reference materials required for that segment. References are not "memorized" or "inherited" to the next segment (only the referenced references participate in reasoning for the current segment).

- **No Global Memory**: The node parses the reference labels explicitly written in the current segment's prompt to accurately determine which materials are needed for that segment. Writing image1 in the 1st segment only means it was used in the 1st segment; the next segment will re-scan.
- **No Write, No Transfer**: If the Nth segment does not write image1 again, that reference image will not be passed to this segment, causing inconsistency in characters/objects.

**❌ Incorrect Writing (Implicit Inheritance)**

```text
Segment 1[3s]: image1 is Object A, Object A is moving forward.
Segment 2[3-6s]: Object A stops, turns to look at the camera. (No image1 written)
```

**✅ Correct Writing (Explicit Per-Segment)**

```text
Segment 1[3s]: image1 is Object A, Object A is moving forward.
Segment 2[3-6s]: image1 is Object A, Object A stops, turns to look at the camera.
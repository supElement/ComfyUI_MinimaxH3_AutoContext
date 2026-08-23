<div align="center">

[![中文](https://img.shields.io/badge/Language-Simplified%20Chinese-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

> One-click MiniMax H3 long video automatic generation node: **Segmented Inference + Segment Continuity Anchor + Prompt Timeline Slicing + Secondary Sampling (SS) + Seam Correction**.

In the case of limited GPU memory, long videos are divided into multiple segments for independent inference, and seamless splicing between segments is achieved through superposition enhancement methods. At the same time, prompts are automatically sliced along the timeline, aligning the generated content with the rhythm of the prompts. Supports SS (low-quality one-pass + high-quality SS).

<img width="2230" height="976" alt="image" src="https://github.com/user-attachments/assets/5634914a-6f98-4d4f-b573-2c8b41e0c57e" />


<img width="2209" height="1030" alt="image" src="https://github.com/user-attachments/assets/9bbdda2a-d4ce-4836-b108-e359e72e31de" />

## 📖 Table of Contents

- [Node List](#nodes)
- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [Second Pass & SplitSigmas Low-High Frequencies](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Warnings (Node Limitations)](#limitations)

## <a id="nodes"></a> 🧩 Node List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter group node: centrally manage prompts/segments/resolution/audio, output `parameter`, and preview 'Expected Segments' in real-time |
| **Minimax_H3_AutoContext_Sampler** | Main node: segmented inference + continuity anchoring + sampling (common for one-pass/SS) |
| **Minimax_H3_Seam_Correction** | Seam correction node: pixel domain correction for segment junctions after decoding |

> Usage: `parameter node --parameter--> Main node`. Prompts are filled in the parameter node, and the main node receives `parameter` (required) through `parameter`.

## <a id="features"></a> ✨ Core Features

### 🧩 Segmented Inference

- Divided into multiple segments according to `total_frames` / `chunk_frames` (frame unit), frame numbers recommended to take 5, 22, 39, 56, 73, 90...
- The last segment is automatically lengthened to fill, avoiding overly short tail segments
- `fps` is only used for audio synchronization and prompt second conversion

### 🔗 Segment Continuity

- **Superposition Enhancement**: The non-first segment automatically "relays" the end of the previous segment, and the new content extends naturally from the end of the previous segment, eliminating stalling or position jumps at the junction
- The end of the previous segment is used as a motion reference for the current segment, helping to continue the direction and speed of motion
- The audio of the previous segment is also passed in as "previous content" to help the sound continue naturally
- Segment audio fades smoothly, aligned with the frame rate of the video

> Frame number rule: `total_frames` / `chunk_frames` / `context_frames` all take 5, 22, 39, 56, 73, 90... (multiples of 17 + 5), nodes will automatically align, generally no manual calculation is required.

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **auto** | Paragraphs containing time marks (such as `0-5s`) are automatically segmented, and paragraphs without marks (style/sound/banned items) are spliced into each window |
| **timeline** | Splitting according to time marks is enforced |
| **global** | The entire prompt is used for all segments (suitable for a single take of consistent actions) |
| **sequential** | Poured onto the timeline according to the reading order, paragraphs starting with `Global:`/`[Global]` are spliced into each window |

### 🏷️ Clip_Tag Tag Segmentation Mode

- Segmented by user-defined tags (such as `Segment 1`/`Segment 2`/`Segment 3`) to segment prompts, each segment = one chunk = all prompts in the segment
- Segment duration is determined by the prompt content (three priority levels):
  1. Duration following the tag line (such as `Segment 1:0-5 seconds` → 5 seconds; `Segment 1:3-8 seconds` → 5 seconds)
  2. Maximum end value of time marks within the segment (such as `【0-2 seconds】`+`【2-5 seconds】` → 5 seconds)
  3. Default value of `total_frames / fps` as a fallback (total frame number of a single segment matches `total_frames`)
- Segment time marks are **relative time** (each segment starts from 0), not global absolute time
- An additional segment is generated for non-first segments to overlap for splicing, which is automatically trimmed after generation
- Total duration is automatically aligned with the target total frame number, as close as possible to the expected duration
- The tag itself is removed during inference, and the rest of the prompt content is output according to `prompt_format`

### 🎯 Reference Smart Filtering (Image / Video / Audio)

- Automatically identify the reference image/video/audio used in each segment of the prompt, and only pass the materials referenced to that segment
- Video is bound with its matching audio track to avoid cross-interference between image and sound

### 🖌️ Secondary Sampling (SS)

- The main node `latent_input` is connected to the one-pass latent (or expanded through the latent expansion node) to enter the SS mode
- SS resolution is **based on the input latent** (ignore width/height), achieving low-quality one-pass → high-quality SS
- `denoise` controls the strength of redrawing; `sigmas` supports custom sigma sequences (like `SamplerCustomAdvanced`)
- `lock_audio`: SS only redraws video, reuse one-pass audio

### 🎵 Audio Drive

- `drive_audio` (AUDIO, optional) + `audio_drive` switch
- After enabling, the video follows the audio to generate, output audio = source audio itself (mouth shape/rhythm driven by it)

## <a id="install"></a> 📦 Installation

### Method 1: Manual Installation (Manual Installation)

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
```

### Method 2: Installation via Manager (Install using Manager)

Search for `ComfyUI_MinimaxH3_AutoContext` in the ComfyUI Manager and click Install.


## <a id="params"></a> ⚙️ Node Parameters

### Minimax_H3_AutoContext_parameter (Parameter Group Node)

| Parameter | Default Value | Description |
|------|--------|------|
| long_prompt | — | Prompt (given to the main node for inference and also used for 'Expected Segments' preview) |
| prompt_mode | auto | Prompt timeline mode (only Clip_Frame is effective) |
| clip_mode | Clip_Frame | Segmentation mode: Clip_Frame / Clip_Tag |
| clip_tag | Segment 1 | Clip_Tag segmentation tag template (must end with a number) |
| prompt_format | official | Prompt output format: official / legacy / raw |
| crop_mode | stretch | Reference image/first/last frame/reference video scaling and cropping: center / stretch / none |
| ref_sync_mode | segmented | Whether to slice reference video/audio by segment: global / segmented |
| width × height | 960×544 | One-pass resolution (covered by `latent_input` in SS) |
| total_frames | 362 | Total number of generated frames (17n+5); in Clip_Tag, segments without explicit duration are covered by `total_frames`, and the final total is covered by the sum of each segment |
| fps | 24 | Frame rate, used for audio synchronization and prompt second conversion |
| chunk_frames | 90 | Number of frames generated per segment (17n+5) |
| context_frames | 22 | Number of frames for segment continuity (17n+5: 5/22/39/56…) |
| lock_audio | enable | Lock audio area, only redraw video |
| audio_drive | disable | Audio drive switch |


> A real-time preview of 'Expected Segments' is displayed on the node (front-end JS calculation, not involved in inference).

### Minimax_H3_AutoContext_Sampler (Main Node)

| Parameter | Default Value | Description |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 model component |
| parameter | Required | Parameter group input (from parameter node) |
| sampler | Optional | External sampler object (SAMPLER), overrides built-in sampler_name/scheduler |
| sigmas | Optional | Custom sigma sequence (SIGMAS), highest priority |
| latent_input | Optional | SS input latent (entering enables SS) |
| info | Optional | Parameter inheritance input (multi-sampling linked, ensure consistent segmentation) |
| first_frame / last_frame | Optional | First/last frame anchoring (FL2VA) |
| video_context_denoise | 0.0 | Segment continuity strength (only for non-first segment): 0=precisely continue the end of the previous segment, 1=re-generate, intermediate values=soft blending. When connected to SplitSigmas, it is recommended to set 1 to avoid screen flickering. |
| seed | 0 | Random seed (control_after_generate) |
| steps / cfg | 30 / 1.0 | Sampling steps / CFG |
| sampler_name / scheduler | euler / simple | Built-in sampler / scheduler |
| denoise | 1.0 | Redrawing strength (1=full resampling, the smaller the more original structure is retained) |
| ref_image_N / ref_video_N / ref_video_audio_N / ref_audio_N | Optional | Reference materials (Autogrow dynamic port) |
| drive_audio | Optional | Audio drive source |

## <a id="output"></a> 📤 Output

| Output | Description |
|------|------|
| **latent** | Pasted audio-video latent, connected to VAE Decode, or expanded and connected to SS |
| **denoised_latent** | Clean latent output, used for SS接力 / preview |
| **info** | Segment parameters (Dict), passed to the next main node's info input, ensuring consistent segmentation for multi-sampling |



## <a id="second-pass"></a> 🔄 Second Pass & SplitSigmas Low-High Frequencies

### Basic SS (Low-Quality One-Pass → High-Quality SS)

```
parameter node ──parameter──> Main node (one-pass, 864×480)
    └─ latent / denoised_latent ──> [Separate AV] ──> video_latent ──> latent expansion ──> [Merge AV] ──> Main node (SS).latent_input
SS node: parameter shared (or inherited from info), denoise optional 0.4~0.6
```

- SS resolution is based on the `latent_input`, ignoring parameter width/height

### SplitSigmas Low-High Frequencies (Save Time, Improve Clarity)

> ⚠️ **Audio Constraint**: Low-high frequencies **only affect video** (audio segment continuity requires complete sampling), audio should be kept in complete sampling.

```
One-pass node: complete sampling (do not connect to high_sigmas, audio completely denoised)
          → denoised_latent → Separate expansion video (audio not moved) → Merge → SS.latent_input
SS node: sigmas ← low_sigmas (only run low sigma segments to improve detail)
          lock_audio = True (reuse one-pass complete audio)
          video_context_denoise = 1.0 (continuity area redrawn with the new area, avoiding screen flickering)
```

> 💡 **SS `video_context_denoise`**: When connected to SplitSigmas, if set to 0 (precisely continue), the continuity area and the newly redrawn new area may flicker at the boundary; set to 1.0 to synchronize the redraw of the continuity area to avoid this. If the seam is slightly discontinuous, it can be reduced to 0.3~0.5 for a compromise. Keep the default 0 for one-pass.

## <a id="seam"></a> 🧵 Seam Correction Node (Minimax_H3_Seam_Correction)

| Parameter | Default Value | Description |
|------|--------|------|
| `fix_color_preset` | `"medium"` | **Color/exposure processing preset**<br>`off`: do not process; <br>`low`: per-channel brightness gain, correction amount halved, most conservative, no color bias; <br>`medium`: per-channel brightness gain, only correct the level jump of the junction (recommended); <br>`high`: MKL linear color migration, longer statistical window, more stable when motion is large; <br>`max`: full frame brightness normalization, eliminating gradual shift within segments, but flattening the inherent brightness changes of the image itself (such as when it gets dark/enters a tunnel), near black frames are invalid (log report). |
| `fix_motion_preset` | `"off"` | **Junction continuity (optical flow alignment + fusion) preset**<br>`off`: do not process (recommended to observe the effect using the color preset first); <br>`low/medium/high/max`: The higher the level, the more frames and strength involved in the fusion, but it is more likely to bring slight blurring or breathing effect. |
| `fix_flash` | `false` | **Flash frame processing** (sudden brightness jump at the boundary). Independent switch, using time-domain
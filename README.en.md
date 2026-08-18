<div align="center">

[![Chinese](https://img.shields.io/badge/Language-Chinese-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

> One-click MiniMax H3 Long Video Auto-Generation Node: **Segmented Inference + Inter-segment Anchor + Prompt Timeline Slicing**.

Split long videos into multiple segments for independent inference under limited GPU memory, achieve seamless connection between segments through multi-frame keyframe anchoring, and automatically slice prompts along the timeline to align the generated content with the prompt rhythm.

## 📖 Table of Contents

- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Notes (Node Limitations)](#limitations)

## <a id="features"></a> ✨ Core Features

### 🧩 Segmented Inference

- Split into multiple segments by `total_frames` / `chunk_frames` (frame unit), parameter step size is 17 (optional 5, 22, 39, 56, 73, 90, ...)
- Each segment's frame count is adsorbed to the H3's 17n+5 VAE temporal grid
- The last segment is extended preferentially to absorb anchoring deficit (≤30% limit) to avoid generating small tail segments
- `fps` is only used for audio synchronization and prompt second conversion

### 🔗 Inter-segment Anchor

- Extract N latent frames from the end of the previous segment as independent keyframes and anchor them to the corresponding time coordinates of the current segment
- The model sees the real motion sequence rather than a single static frame, correctly continuing the motion direction and speed
- Context audio is passed through the ref channel (do not insert `<Audio j>` tag), and the model treats it as "previous content" rather than reference material
- Inter-segment audio cosine cross-fades and aligns with the frame count

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **auto** | Paragraphs with time marks (such as `0-5s`) are automatically split, and paragraphs without marks (styles/sounds/banned items) are spliced into each window |
| **timeline** | Forced split by time marks |
| **global** | The entire prompt is used for all windows (suitable for one-shot-throughout uniform action) |
| **sequential** | Poured onto the timeline according to the sentence reading order, paragraphs starting with `global:`/`[global]` are spliced into each window |

### 🏷️ Clip_Tag Tagging Segmentation Mode

- Split prompts by user-defined tags (such as `段1`/`段2`/`段3`), each segment = one chunk = all prompts of the segment
- Segment duration is determined by the prompt content (three levels of priority):
  1. Duration following the tag line (such as `段1:0-5 seconds` → 5 seconds; `段1:3-8 seconds` → 5 seconds)
  2. The maximum end value of time marks within the segment (such as `【0-2 seconds】`+`【2-5 seconds】` → 5 seconds)
  3. Default value of `chunk_frames / fps`
- Segment time marks are **relative time**, not global absolute time
- Continuation anchoring compensation: Non-first segments generate `context_frames` frames for head anchoring, automatically trimmed after generation to ensure continuity between segments
- Tags are removed during inference, and the rest of the prompt content is output according to `prompt_format`:
  - `official` / `legacy`: Time coordinates are automatically converted to relative within the segment
  - `raw`: Output after removing tags, without any conversion (used for Clip_Tag only)

### 🎯 Reference Intelligent Filtering (Image / Video / Audio)

- Parse references in each segment prompt (`image1`/`image 1`/`图像1`/`图片1`, `视频1`, `音频1`/`audio 1`, as well as native `<Picture N>`/`<Video N>`/`<Audio N>`, base 1), and only pass the referenced references to the current segment
- Numbers are automatically remapped to continuous sequence numbers, aligned with the model's native `<Picture N>`/`<Video N>`/`<Audio N>` tags
- Video is bound to its matching audio track; audio tracks and independent audio are counted as `<Audio N>` (audio track first) according to the official rules
- Avoid interference between image and sound when all references are passed at the same time

### 📐 Dynamic Ports (Autogrow)

- Only `_0` port is displayed by default, and `_1` is displayed after connection, and so on
- Supported: reference images (0-9), reference videos (0-3), matching audio tracks of reference videos (0-3), independent reference audio (0-3)
- `first_frame` / `last_frame` are always displayed (used for FL2VA head and tail frame anchoring mode)

### 🎵 Audio Drive (Audio Drive, Reference Audio Drive Video)

- New `drive_audio` (AUDIO, optional) input + `audio_drive` (disable/enable, default disable) switch
- When enabled: Lock the `drive_audio` encoded into the audio half of the latent, and set `noise_mask` (video=1 normal denoising, audio=0 not regenerating), so that video generation is "driven" by this audio (mouth shape/rhythm alignment), and at the same time, **output audio = `drive_audio` itself** (lossless, bypassing VAE round-trip)
  - `ref_audio_0` is responsible for semantic driving, and `drive_audio` is responsible for locking the output
- Slice the source audio according to the output time axis for each segment (including replay heads of non-first segments), spliced together and aligned with the video frame by frame
- Automatically fill in zeros when the source audio is shorter than the target duration, and truncate automatically when it is longer

### 🔧 Other

- Node progress bar and preview image are displayed in real time
- PackedLayout time coordinate correction: `cond_t` of keyframes is based on the actual starting coordinate of the video segment, not text_len
- extra_conds splicing repair: keyframe cond rows are spliced with ref rows rather than overwritten

## <a id="install"></a> 📦 Installation

### Method One: Manual Installation (Manual Installation)

Enter the `./ComfyUI/custom_nodes` directory and run:

```bash
git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
```

### Method Two: Installation via Manager (Install using Manager)

Search for `ComfyUI_MinimaxH3_AutoContext` in the ComfyUI Manager, then click Install.

> **Dependencies**: `torchaudio` (used for audio resampling, optional).

## <a id="params"></a> ⚙️ Node Parameters

### Model Components

| Parameter | Default Value | Description |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 model component |

### Keyframes and Prompts

| Parameter | Default Value | Description |
|------|--------|------|
| first_frame | Optional | First frame anchoring (FL2VA mode) |
| last_frame | Optional | Last frame anchoring (FL2VA mode) |
| long_prompt | — | Long prompt, supports time-marked segmentation |
| clip_mode | Clip_Frame | Segmentation mode: Clip_Frame (uniformly cut by frame) / Clip_Tag (cut by custom tags) |
| clip_tag | 段1 | Clip_Tag segmentation tag template, must end with a number (such as `段1`/`A01`/`[片段001]`) |
| prompt_mode | auto | Prompt timeline mode (only effective for Clip_Frame) |
| prompt_format | official | Prompt output format: official / legacy / raw (raw is used for Clip_Tag, output as is) |

### Image Processing

| Parameter | Default Value | Description |
|------|--------|------|
| crop_mode | stretch | Scaling and cropping of reference images/first and last frames/reference videos: center (proportionally scaled to the shortest side and centered cropped) / stretch (stretched directly) / none (maintain original resolution, only 32 aligned, advanced users) |
| ref_sync_mode | global | Whether reference video/audio is segmented by segment: global (use the complete reference for each segment) / segmented (take the corresponding time segment for each segment, used for mouth shape synchronization) |
| decode_output | disable | Whether to internally decode: disable (default, only output latent) / enable (output image+audio+latent) |

### Generation Parameters

| Parameter | Default Value | Description |
|------|--------|------|
| width × height | 960×544 | Output resolution |
| total_frames | 362 | Total number of generated frames (must meet 17n+5, about 15s@24fps), the sum of each segment is covered by the segments in Clip_Tag mode |
| fps | 24 | Frame rate, only used for audio synchronization and prompt second conversion |
| chunk_frames | 90 | Number of frames generated per segment (must meet 17n+5, about 3.75s@24fps), the default value as duration backup in Clip_Tag mode |
| context_frames | 22 | Inter-segment anchoring frame count, the larger the value, the stronger the continuity |
| steps / cfg / sampler / scheduler | 30 / 1.0 / euler / simple | Sampling parameters |
| seed | 0 | Random seed |

### Reference Materials

| Parameter | Default Value | Description |
|------|--------|------|
| ref_image_N | Optional | Reference image (use `image1`/`image 1`/`图像1`/`图片1` or `<Picture N>` in the prompt, base 1) |
| ref_video_N | Optional | Reference video frames |
| ref_video_audio_N | Optional | Matching audio track of the same number reference video |
| ref_audio_N | Optional | Independent reference audio |

### Audio Drive

| Parameter | Default Value | Description |
|------|--------|------|
| drive_audio | Optional | Audio drive source (to be locked), use with `audio_drive=enable` |
| audio_drive | disable | Audio drive switch: enable locks `drive_audio` (noise_mask=0), and the output audio = source audio itself |

## <a id="output"></a> 📤 Output

| Output | Description |
|------|------|
| **video_frames** | Packed complete video frame sequence (internally decoded, segment-wise cropping aligned) |
| **audio** | Packed audio (cosine cross-fades between segments) |
| **latent** | Common latent for audio and video (NestedTensor), spliced after removing the segment-wise replay area, can be decoded with ComfyUI's native VAE Decode |

> 💡 **Recommended Usage**: `latent → VAE Decode` to get the video, `latent → VAE audio Decode`, if there is a problem with the audio, you can try the `audio` port output audio.

## <a id="prompt-examples"></a> ✍️ Prompt Writing Examples

### Timeline Mode (auto / timeline)

```text
0-5s: ...
5-10s: ...

integrated_multimodal_description
....

overall_soundscape
```

> Paragraphs containing `0-5s` marks are split by time, and paragraphs without marks (styles/sounds/banned items) are automatically spliced into each window.

### Global Mode (global)

> The entire prompt is used for all segments, suitable for one-shot-throughout uniform action.

### Clip_Tag Mode (Segment by Tag)

> Set `clip_mode` to `Clip_Tag`, and fill in the tag template (`clip_tag`) (must end with a number).

**Tag Template Example**

| Template | Match |
|------|------|
| `段1` | `段1` / `段2` / `段3` (prefix "段"+number) |
| `A01` | `A01` / `A02` / `A03` (prefix "A"+number) |
| `[片段001]` | `[片段001]` / `[片段002]` (prefix "[片段"+number+suffix"]") |

**Tag Writing**: Tags occupy a line alone as a separation point, and it is recommended to change lines after the tag. It can also be processed without line breaks (skip separators to take segment content):

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

1. Duration following the tag line: `段1:0-5 seconds` → 5 seconds; `段1:3-8 seconds` → 5 seconds (time mark will be removed from the prompt)
2. Time marks within the segment 0-based: `【0-2 seconds】`+`【2-5 seconds】` → 5 seconds
3. None → `chunk_frames / fps` as a backup

**prompt_format Selection**

- `official` / `legacy`: Time marks within the segment are automatically converted to relative coordinates rendering
- `raw`: Output after removing tags, time marks remain unchanged (suitable for structured prompts generated by large models)

**Continuation Compensation**: Non-first segments generate `context_frames` frames for head anchoring (replay the tail of the previous segment), automatically trimmed after generation to ensure continuity between segments. Actual output duration = sum of effective new additions of each segment, the actual value will be printed in the run log.

## <a id="limitations"></a> 📝 Prompt Notes (Node Limitations)

> The following notes**do not apply** to simple, always effective prompt scenarios (i.e., all segments share the same prompt, global mode),
> such as: mouth-to-mouth digital humans (of course, the dialogue needs to be segmented), or scenes where the camera composition of the video does not change much, or video replacement roles and other general prompt scenarios.

### 1️⃣ Core Principle: Temporal Exclusivity

> When using segmented inference (Chunk), please strictly comply with the principle of **temporal exclusivity**—each segment's prompt can only describe the **new changes** that **are happening** in the segment relative to the end of the previous segment.

- **Segmenting is "Relay"**: When the Nth segment is generated, its starting image state (position, action posture, camera position) is completely provided by the "anchoring frame (Context Frames)" at the end of the previous segment. You do not need to repeat the starting state in the prompt.
- **Prohibition of "Retrospection" and "Overlap"**: The prompt of the Nth segment absolutely cannot repeat the actions or camera movements that have been completed in the N-1th segment. If repeated (such as "crossing" in the previous segment, and "crossing" again in this segment), the instructions received by the model will conflict with the image of the anchoring frame (i.e., instruction conflict), causing the generated image to be stuck, the motion logic to be confused, or the action to be repeated.
- **Boundary Zeroing**: When switching segments, please zero out the "action in progress" of the previous segment. The prompt of the new segment should be like "new instructions after pressing the shutter", only describing the displacement, action, or new element appearance that occurs in the new time period.

**❌ Incorrect Writing (Conflict Overlap**)

```text
Segment 1 description: "Object A moves to position B" (covering time period 0-5s)
Segment 2 description: "Object A moves to position B after stopping and turning to face the camera" (covering time period 5-10s)
```

> Problem Analysis: At the end of the first segment, the anchoring frame shows that object A has reached position B and just stopped. But the prompt of the second segment forcibly requires "Object A to move to position B", which conflicts with the static result of "already reached" in the anchoring frame, causing the model to try to "relocate", resulting in stroboscopic or jumping frames.

**✅ Correct Writing (Seamless Progression**)

```text
Segment 1 description: "Object A moves to position B and finally stops at position B" (emphasizing action closure)
Segment 2 description: "After standing still, object A slowly turns its direction" (directly describing the new action after the end of the previous segment)
```

> Correct Logic: The second segment completely abandons the description of the "moving process", defaults to "stopped at B point" as a fact, and only describes the "turning" new action that follows, so that the model can perfectly continue using the anchoring frame.

> 🚀 **Summary in One Sentence**: The end of the previous segment is "the result", and the beginning of the next segment is "the new action after the result", don't write the "process leading to the result" into the next segment.

### 2️⃣ Core Principle: Sequential Reference Declaration

> When using segmented inference and pairing reference images/videos (image1, video1, etc.), please strictly comply with the principle of **
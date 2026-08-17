<div align="center">

[![Chinese](https://img.shields.io/badge/Language-Chinese-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

One-click MiniMax H3 Long Video Automatic Generation Node: **Segmented Inference + Inter-segment Continuation Anchoring + Prompt Timeline Slicing**.

In situations where GPU memory is limited, long videos are divided into multiple segments for independent inference. Seamless connection between segments is achieved through multi-frame keyframe anchoring, and prompts are automatically sliced along the timeline to align the generated content with the prompt rhythm.

## Core Features

### Segmented Inference
- Divided into multiple segments based on `total_frames` / `chunk_frames` (frame unit), parameter step size is 17 (optional: 5, 22, 39, 56, 73, 90, ...).
- Each segment's frame count is吸附 to the H3's 17n+5 VAE temporal grid.
- The last segment is extended by a priority to absorb the anchoring deficit (≤30% limit) to avoid producing tiny tail segments.
- `fps` is only used for audio synchronization and prompt second conversion.

### Inter-segment Continuation Anchoring
- Extract N latent frames from the end of the previous segment as independent keyframes and anchor them to the corresponding time coordinate of the current segment.
- The model sees a real motion sequence rather than a single static frame, correctly continuing the direction and speed of motion.
- Context audio is passed through the ref channel (no `<Audio j>` tag inserted), and the model treats it as "previous content" rather than reference material.
- Inter-segment audio is cosine cross-faded and strictly aligned with the number of video frames.

### Prompt Timeline
- **auto**: Segments with time markers (such as `0-5s`) are automatically split. Segments without markers (style/sound/ban items) are merged into each window.
- **timeline**: Splitting is forced by time markers.
- **global**: The entire segment prompt is used for all windows (suitable for one-take shots with consistent actions throughout).
- **sequential**: Poured onto the timeline according to the sentence reading order, with segments starting with `全局:`/`[全局]` merged into each window.

### Clip_Tag Tag-based Segmentation Mode
- Split prompts based on user-defined tags (such as `段1`/`段2`/`段3`). Each segment = one chunk = all prompts of that segment.
- Segment duration is determined by the prompt content (three priority levels):
  1. Duration immediately following the tag line (such as `段1:0-5秒` → 5 seconds; `段1:3-8秒` → 5 seconds).
  2. Maximum end value of time markers within the segment (such as `[0-2秒】`+`【2-5秒】` → 5 seconds).
  3. Default value of `chunk_frames / fps`.
- Segment time markers are**relative time** (starting from 0 in each segment), not global absolute time.
- Continuation anchoring compensation: Non-first segments generate `context_frames` frames for head anchoring, which are automatically trimmed after generation to ensure continuity between segments.
- Tags are removed during inference, and the rest of the prompt content is output according to `prompt_format`:
  - `official`/`legacy`: Time coordinates are automatically converted to relative coordinates within the segment.
  - `raw`: Tags are removed and the content is output as is, without any conversion (used with Clip_Tag only).

### Reference Smart Filtering (Image / Video / Audio)
- Parse references in each segment prompt (`image1`/`image 1`/`图像1`/`图片1`, `视频1`, `音频1`/`audio 1`, as well as native `<Picture N>`/`<Video N>`/`<Audio N>`, 1-based), and only pass the referenced references to the current segment.
- The numbering is automatically remapped to a continuous sequence, aligning with the native `<Picture N>`/`<Video N>`/`<Audio N>` tags.
- Video is bound to its paired audio track; tracks and independent audio are counted as `<Audio N>` (track first) according to official rules.
- Avoid interference between image and sound when all references are passed at the same time.

### Dynamic Ports (Autogrow)
- Only the `_0` port is displayed by default, and `_1` is displayed after connection, and so on.
- Supported: Reference images (0-9), reference videos (0-3), paired audio tracks of reference videos (0-3), and independent reference audio (0-3).
- `first_frame`/`last_frame` are always displayed (used for FL2VA head and tail frame anchoring mode).

### Audio Drive (Audio Drive, reference audio drive video)
- New `drive_audio` (AUDIO, optional) input + `audio_drive` (disable/enable, default disable) switch.
- When enabled: The `drive_audio` is encoded and locked into the audio half of the latent space, and `noise_mask` (video=1 normal denoising, audio=0 not regenerated) is set, so that video generation is "driven" by this audio (mouth shape/rhythm alignment), and at the same time**output audio = `drive_audio` itself** (lossless, bypassing the VAE round trip).
  —— `ref_audio_0` is responsible for semantic driving, and `drive_audio` is responsible for locking the output.
- Source audio is sliced according to the output timeline for each segment (including replay heads of non-first segments), and then spliced and aligned with the video frame by frame.
- Automatically zero-padded when source audio is shorter than the target duration, and automatically truncated when longer.

### Other
- Node progress bar and preview image are displayed in real time.
- PackedLayout time coordinate correction: `cond_t` of keyframes is based on the actual starting coordinate of the video segment, not text_len.
- extra_conds splicing repair: keyframe cond rows are spliced with ref rows instead of overwritten.

## Installation

- Manual Installation (Manual Installation)<br>
Enter the ./ComfyUI/custom_nodes directory and run the following code:<br>

      git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git

- Install using Manager (Install using Manager)<br>

  Search for ComfyUI_MinimaxH3_AutoContext in the ComfyUI Manager, then Install.

Dependencies: `torchaudio` (used for audio resampling, optional).

## Node Parameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| model / vae / audio_vae / clip | — | MiniMax H3 model component |
| first_frame | Optional | First frame anchoring (FL2VA mode) |
| last_frame | Optional | Tail frame anchoring (FL2VA mode) |
| long_prompt | — | Long prompt, supports time-marked segmentation |
| clip_mode | Clip_Frame | Segmentation mode: Clip_Frame (uniformly cut by frame) / Clip_Tag (split by custom tags) |
| clip_tag | 段1 | Clip_Tag segmentation tag template, must end with a digit (such as `段1`/`段2`/`段3`) |
| prompt_mode | auto | Prompt timeline mode (only effective for Clip_Frame) |
| prompt_format | official | Prompt output format: official / legacy / raw (raw is used with Clip_Tag, output as is) |
| crop_mode | stretch | Scaling and cropping of reference images/first and last frames/reference videos: center (proportional scaling to the shortest side and center cropping) / stretch (stretch directly) / none (keep original resolution, only 32 aligned, advanced users) |
| ref_sync_mode | global | Whether to slice reference videos/audio by segment: global (use complete reference for each segment) / segmented (take corresponding time segment for each segment, used for mouth shape synchronization) |
| decode_output | disable | Internal decoding: disable (default, only output latent) / enable (output image+audio+latent) |
| width × height | 960×544 | Output resolution |
| total_frames | 362 | Total number of generated frames (must meet 17n+5, about 15s@24fps), covered by the sum of each segment in Clip_Tag mode |
| fps | 24 | Frame rate, only used for audio synchronization and prompt second conversion |
| chunk_frames | 90 | Number of frames generated per segment (must meet 17n+5, about 3.75s@24fps), used as the default value for duration in Clip_Tag mode |
| context_frames | 22 | Inter-segment continuation frames, the larger the value, the stronger the continuity |
| steps / cfg / sampler / scheduler | 30 / 1.0 / euler / simple | Sampling parameters |
| seed | 0 | Random seed |
| ref_image_N | Optional | Reference image (used in prompt with `image1`/`image 1`/`图像1`/`图片1` or `<Picture N>` reference, base 1) |
| ref_video_N | Optional | Reference video frame |
| ref_video_audio_N | Optional | Paired audio track of the same number as the reference video |
| ref_audio_N | Optional | Independent reference audio |
| drive_audio | Optional | Audio drive source (source audio to be locked), used with `audio_drive=enable` |
| audio_drive | disable | Audio drive switch: enable locks `drive_audio` (noise_mask=0), output audio = source audio itself |

## Prompt Writing Examples

### Timeline Mode (auto / timeline)

```
0-5s: ...
5-10s: ...

integrated_multimodal_description
....

overall_soundscape

```

Segments with `0-5s` markers are split by time, and segments without markers (style/sound/ban items) are automatically merged into each window.

### Global Mode (global)

The entire segment prompt is used for all segments, suitable for one-take shots with consistent actions throughout.

### Clip_Tag Mode (split by tags)

Set `clip_mode` to `Clip_Tag`, and fill in the tag template (`clip_tag`) (must end with a digit).

**Tag template example**:
- `段1` → Matches `段1`/`段2`/`段3` (prefix "段"+digit)
- `A01` → Matches `A01`/`A02`/`A03` (prefix "A"+digit)
- `[片段001]` → Matches `[片段001]`/`[片段002]` (prefix "[片段"+digit+suffix"]")

**Tag writing**: The tag occupies a line on its own as a separator, and it is recommended to have a line break after the tag. It can also be processed (skip the separator to take the segment content) without a line break:
```
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

**Segment duration rules** (three priority levels):
1. Duration immediately following the tag line: `段1:0-5秒` → 5 seconds; `段1:3-8秒` → 5 seconds (duration markers are removed from the prompt).
2. 0-based time markers within the segment: `[0-2秒】`+`【2-5秒】` → 5 seconds.
3. None of the above → `chunk_frames / fps` as a fallback value.

**prompt_format selection**:
- `official`/`legacy`: Segment time markers are automatically converted to relative coordinates within the segment.
- `raw`: Tags are removed and the content is output as is, without any conversion (suitable for structured prompts generated by large models).

**Continuation compensation**: Non-first segments generate `context_frames` frames for head anchoring (replaying the end of the previous segment), which are automatically trimmed after generation to ensure continuity between segments. The actual output duration = sum of effective new content of each segment, and the actual value will be printed in the run log.

## Output

- **video_frames**: The complete video frame sequence after splicing (internally decoded, spliced and aligned between segments).
- **audio**: The audio after splicing (cosine cross-fading between segments).
- **latent**: The audio and video share the same latent (NestedTensor), spliced after removing the redundant area between segments, and can be decoded using ComfyUI's native VAE Decode. Recommended usage: latent → VAE Decode to get video, and output audio (waveform level cross-fading, better continuity) along with the audio port.
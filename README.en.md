<div align="center">

[![Chinese](https://img.shields.io/badge/Language-Simplified%20Chinese-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

One-click MiniMax H3 long-video auto-generation node: **chunked inference + inter-chunk continuation anchoring + prompt timeline slicing**.

With limited VRAM, split a long video into multiple independently inferred chunks, anchor them seamlessly via multi-frame keyframes, and automatically slice prompts along the timeline so each chunk's content aligns with the prompt rhythm.

## Core Features

### Chunked Inference
- Split into chunks by `total_frames` / `chunk_frames` (in frames); parameter stride is 17 (optional: 5, 22, 39, 56, 73, 90, ...)
- Each chunk's frame count snaps to H3's 17n+5 VAE temporal grid
- The last chunk is preferentially lengthened to absorb the anchoring deficit (≤30% cap), avoiding tiny trailing chunks
- `fps` is only used for audio sync and seconds conversion inside prompts

### Inter-chunk Continuation Anchoring
- Extract N latent frames from the tail of the previous chunk as standalone keyframes, anchored to the corresponding time coordinates of the current chunk
- The model sees a real motion sequence rather than a single static frame, correctly continuing motion direction and speed
- Context audio is passed through the ref channel (no `<Audio j>` tags inserted); the model treats it as "previous content" rather than reference material
- Inter-chunk audio cosine cross-fade, strictly aligned with the video frames

### Prompt Timeline
- **auto**: paragraphs containing time markers (e.g. `0-5s`) are sliced automatically; unmarked paragraphs (style / sound effects / prohibitions) are merged into every window
- **timeline**: force-slice by time markers
- **global**: the whole prompt is used for all windows (suitable for a one-take shot with homogeneous motion throughout)
- **sequential**: spread paragraphs in reading order across the timeline; paragraphs starting with `全局:`/`[全局]` are merged into every window

### Clip_Tag Paragraph Segmentation Mode
- Slice prompts by user-defined tags (e.g. `段1`/`段2`/`段3`); each segment = one chunk = all of that segment's prompt
- Segment duration is determined by the prompt content (three-level priority):
  1. Duration right after the tag line (e.g. `段1:0-5秒` → 5s; `段1:3-8秒` → 5s)
  2. The maximum end value of in-segment time markers (e.g. `【0-2秒】`+`【2-5秒】` → 5s)
  3. `chunk_frames / fps` as the fallback default
- In-segment time markers are **relative** (each segment starts from 0), not global absolute time
- Continuation anchoring compensation: non-first segments generate `context_frames` extra frames for head anchoring, automatically cropped after generation to keep segments continuous
- At inference the tags themselves are removed; the remaining prompt content is output per `prompt_format`:
  - `official`/`legacy`: time coordinates are auto-converted to segment-relative
  - `raw`: output as-is after tag removal, no conversion

### Smart Reference Filtering (Image / Video / Audio)
- Parse references in each segment's prompt (`image1`/`image 1`/`图像1`/`图片1`, `视频1`, `音频1`/`audio 1`, and native `<Picture N>`/`<Video N>`/`<Audio N>`, 1-based); only pass the referenced references to the current segment
- Numbers are auto-remapped to consecutive indices, aligned with the model's native `<Picture N>`/`<Video N>`/`<Audio N>` tags
- Videos are bound to their paired audio tracks; tracks and standalone audio are counted as `<Audio N>` per official rules (tracks first)
- Avoids visual/audio crosstalk caused by passing all references at once

### Dynamic Ports (Autogrow)
- Only the `_0` port is shown by default; `_1` appears after connecting, and so on
- Supported: reference images (0-9), reference videos (0-3), paired audio tracks for reference videos (0-3), standalone reference audio (0-3)
- `first_frame` / `last_frame` are always shown (for FL2VA first/last frame anchoring mode)

### Audio Drive (drive the video with reference audio)
- New `drive_audio` (AUDIO, optional) input + `audio_drive` (disable/enable, default disable) switch
- When enabled: encode `drive_audio`, lock it into the latent audio half, and set `noise_mask` (video=1 normal denoising, audio=0 no regeneration),
  so the video generation is "driven" by this audio (lip sync / rhythm alignment), while **output audio = `drive_audio` itself** (lossless, bypassing the VAE round-trip)
  —— `ref_audio_0` handles semantic driving, `drive_audio` locks the output
- Each chunk slices the source audio along the output timeline (including the replay head of non-first chunks); after concatenation it aligns frame-by-frame with the video
- Source audio shorter than the target duration is zero-padded; longer is truncated

### Other
- Node progress bar and preview image shown in real time
- PackedLayout time-coordinate fix: keyframe `cond_t` is based on the video segment's actual start coordinate, not text_len
- extra_conds concatenation fix: keyframe cond rows and ref rows are concatenated rather than overwritten

## Installation

Put the `ComfyUI_MinimaxH3_AutoContext` folder into ComfyUI's `custom_nodes` directory and restart ComfyUI.

Dependencies: `torchaudio` (for audio resampling, optional).

## Node Parameters

| Parameter | Default | Description |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 model components |
| first_frame | optional | First-frame anchoring (FL2VA mode) |
| last_frame | optional | Last-frame anchoring (FL2VA mode) |
| long_prompt | — | Long prompt; supports time-marker segmentation |
| clip_mode | Clip_Frame | Segmentation mode: Clip_Frame (uniform frame slicing) / Clip_Tag (custom tag slicing) |
| clip_tag | 段1 | Split-tag template for Clip_Tag mode; must end with a numeric index (e.g. `段1`/`A01`/`[片段001]`) |
| prompt_mode | auto | Prompt timeline mode (only effective in Clip_Frame) |
| prompt_format | official | Prompt output format: official / legacy / raw (raw is Clip_Tag-only, outputs as-is) |
| crop_mode | stretch | Scale/crop for reference images / first-last frames / reference videos: center (scale to shortest edge with center crop) / stretch (direct stretch) / none (keep original resolution, 32-aligned only, advanced users) |
| ref_sync_mode | global | Whether reference video/audio is sliced per chunk: global (each chunk uses the full reference) / segmented (each chunk takes the corresponding time slice, for lip sync) |
| decode_output | disable | Whether to decode internally: disable (default, latent only) / enable (output image+audio+latent) |
| width × height | 960×544 | Generation resolution |
| total_frames | 362 | Total frames to generate (must satisfy 17n+5, ~15s@24fps); overridden by the sum of segments in Clip_Tag mode |
| fps | 24 | Frame rate; only used for audio sync and seconds conversion inside prompts |
| chunk_frames | 90 | Frames per chunk (must satisfy 17n+5, ~3.75s@24fps); acts as the fallback duration in Clip_Tag mode |
| context_frames | 22 | Inter-chunk continuation frames; larger = stronger continuity |
| steps / cfg / sampler / scheduler | 30 / 1.0 / euler / simple | Sampling parameters |
| seed | 0 | Random seed |
| ref_image_N | optional | Reference images (referenced in prompts via `image1`/`image 1`/`图像1`/`图片1` or `<Picture N>`, 1-based) |
| ref_video_N | optional | Reference video frames |
| ref_video_audio_N | optional | Paired audio track of the reference video with the same number |
| ref_audio_N | optional | Standalone reference audio |
| drive_audio | optional | Audio-drive source (source audio to lock in), used with `audio_drive=enable` |
| audio_drive | disable | Audio-drive switch: when enable, locks `drive_audio` (noise_mask=0), output audio = the source audio itself |

## Prompt Writing Examples

### Timeline Mode (auto / timeline)

```
0-5s: ...
5-10s: ...

integrated_multimodal_description
....

overall_soundscape

```

Paragraphs containing the `0-5s` marker are sliced by time; unmarked paragraphs (style / sound effects / prohibitions) are automatically merged into every window.

### Global Mode (global)

The whole prompt is used for all chunks — suitable for a one-take shot with homogeneous motion throughout.

### Clip_Tag Mode (tag-based segmentation)

Set `clip_mode` to `Clip_Tag` and fill `clip_tag` with the tag template (must end with a numeric index).

**Tag template examples**:
- `段1` → matches `段1`/`段2`/`段3` in the prompt (prefix "段" + number)
- `A01` → matches `A01`/`A02`/`A03` (prefix "A" + number)
- `[片段001]` → matches `[片段001]`/`[片段002]` (prefix "[片段" + number + suffix "]")

**Tag syntax**: the tag occupies its own line as a split point; a newline after the tag is recommended. Lines without a newline are also handled (the separator is skipped and the segment content is taken):
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

**Segment duration rules** (three-level priority):
1. Duration right after the tag line: `段1:0-5秒` → 5s; `段1:3-8秒` → 5s (the duration marker is removed from the prompt)
2. 0-based in-segment time markers: `【0-2秒】`+`【2-5秒】` → 5s
3. Neither → fallback to `chunk_frames / fps`

**prompt_format selection**:
- `official`/`legacy`: in-segment time markers are auto-converted to segment-relative coordinates when rendered
- `raw`: output as-is after tag removal; time markers are kept unchanged (suitable for structured prompts generated by LLMs)

**Continuation compensation**: non-first segments generate `context_frames` extra frames for head anchoring (replaying the previous segment's tail), automatically cropped after generation to keep segments continuous and stutter-free. Actual output duration = the sum of each segment's effective additions; the runtime log prints the actual value.

## Output

- **video_frames**: the concatenated full video frame sequence (internal decode, cropped and aligned between segments)
- **audio**: the concatenated audio (cosine cross-fade between segments)
- **latent**: the shared audio-video latent (NestedTensor), concatenated after cropping the inter-segment replay regions; can be decoded with ComfyUI's native VAE Decode. Recommended usage: latent → VAE Decode for the video, paired with the audio port's output (waveform-level cross-fade, better continuity)

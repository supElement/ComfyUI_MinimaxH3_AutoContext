<div align="center">

[![中文](https://img.shields.io/badge/语言-简体中文-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

> One-click MiniMax H3 long-video auto-generation node: **segmented inference + inter-segment continuation anchoring + prompt timeline slicing + second pass (hi-res resample) + seam correction**.

With limited VRAM, long videos are split into multiple independently inferred segments; seamless connections between segments are achieved through multi-frame keyframe anchoring, while the prompt is automatically sliced along the timeline so that each segment's generated content stays aligned with the prompt's rhythm. Supports a second pass (low-res first pass + hi-res second pass) to improve face/detail clarity.

## 📖 Table of Contents

- [Node List](#nodes)
- [Key Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Outputs](#output)
- [Second Pass and SplitSigmas Hi/Lo Frequency](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Notes (Node Limitations)](#limitations)

## <a id="nodes"></a> 🧩 Node List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter group node: centrally manages prompt/segmentation/resolution/audio parameters, outputs `parameter`, and previews the "expected segments" in real time |
| **Minimax_H3_AutoContext_Sampler** | Main node: segmented inference + continuation anchoring + sampling (shared by first and second pass) |
| **Minimax_H3_Seam_Correction** | Seam correction node: pixel-domain correction of inter-segment seams on the decoded video |

> Usage: `parameter node --parameter--> main node`. The prompt is written in the parameter node; the main node receives it via `parameter` (required).

## <a id="features"></a> ✨ Key Features

### 🧩 Segmented Inference

- Splits into multiple segments by `total_frames` / `chunk_frames` (frame units), with a parameter stride of 17 (options: 5, 22, 39, 56, 73, 90, ...)
- Each segment's frame count snaps to H3's 17n+5 VAE temporal grid
- The last segment is preferentially extended to absorb anchoring deficits (≤30% cap), avoiding tiny tail segments
- `fps` is only used for audio sync and converting seconds in the prompt

### 🔗 Inter-Segment Continuation Anchoring

- Extracts N latent frames from the tail of the previous segment as independent keyframes, anchoring them to the corresponding time coordinates of the current segment
- The model sees a real motion sequence rather than a single still frame, correctly continuing motion direction and speed
- Context audio is passed through the ref channel (without inserting `<Audio j>` tags); the model treats it as "previous content" rather than reference material
- Cosine crossfade between segment audio, strictly aligned with the video frame count

> 📐 **Frame accounting** (H3 17n+5 grid):
> - First segment = `17n+5` (includes the 5 "intro" frames)
> - Non-first segments add **a multiple of 17 frames** (e.g., 68/85 — "pure bead strings" in the middle-to-later part, no intro)
> - `context_frames` (continuation anchoring) must be of the form `17n+5` (5/22/39/56…)
> - The whole latent timeline = intro(5) + N strings(17n), satisfying 17n+5 overall

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **auto** | Paragraphs containing time markers (e.g., `0-5s`) are sliced automatically; paragraphs without markers (style/sound effects/prohibitions) are merged into every window |
| **timeline** | Forces slicing by time markers |
| **global** | The whole prompt is used for all windows (suitable for a one-shot take with uniform motion throughout) |
| **sequential** | Content is spread across the timeline in sentence order; paragraphs starting with `global:`/`[global]` are merged into every window |

### 🏷️ Clip_Tag Label Segmentation Mode

- Slices the prompt by user-defined labels (e.g., `段1`/`段2`/`段3`); each segment = one chunk = all prompt content of that segment
- Segment duration is determined by the prompt content (three-level priority):
  1. The duration immediately following the label line (e.g., `段1:0-5s` → 5 s; `段1:3-8s` → 5 s)
  2. The maximum end value of the time markers within the segment (e.g., `【0-2s】`+`【2-5s】` → 5 s)
  3. Falls back to the `chunk_frames / fps` default
- Time markers inside a segment are **relative times** (each segment starts at 0), not global absolute times
- Continuation anchoring compensation: non-first segments generate `context_frames` extra frames for head anchoring, automatically cropped after generation
- Total duration anchoring: the difference between the sum of each segment's added frames and the target total frame count is compensated from the last segment at a granularity of 17 frames, staying as close to the target total duration as possible
- During inference the labels themselves are removed; the remaining prompt content is output according to `prompt_format`

### 🎯 Reference Smart Filtering (image / video / audio)

- Parses the references in each segment's prompt and passes only the referenced materials to the current segment
- Numbers are automatically remapped to consecutive indices, aligned with the model's native labels
- A video is bound together with its paired audio track; audio tracks and standalone audio are uniformly counted as `<Audio N>`
- Avoids visual/audio crosstalk caused by passing all references at once

### 🖌️ Second Pass (hi-res resample)

- Connecting a first-pass latent (or via a latent upscale node) to the main node's `latent_input` enters second-pass mode
- Second-pass spatial resolution **follows the input latent** (ignores the parameter node's width/height), enabling low-res first pass → hi-res second pass
- `denoise` controls the redraw strength; `sigmas` supports custom sigma sequences (same style as `SamplerCustomAdvanced`)
- `lock_audio` locks the audio region (noise_mask audio=0); the second pass only redraws video and reuses the first-pass audio
- Outputs `denoised_latent` (clean x0 prediction), for multi-pass chaining / hi-lo frequency relay
- Inter-segment anchoring uniformly uses the previous segment's clean x0, keeping seams continuous

### 🎵 Audio Drive

- `drive_audio` (AUDIO, optional) + `audio_drive` switch
- When enabled, `drive_audio` is locked into the audio half of the latent (noise_mask audio=0); the video is driven by the audio, and the output audio = the source audio itself

## <a id="install"></a> 📦 Installation

### Method 1: Manual Installation

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
```

### Method 2: Install using Manager

Search for `ComfyUI_MinimaxH3_AutoContext` in ComfyUI Manager and click Install.

> **Dependencies**: `torchaudio` (for audio resampling, optional).

## <a id="params"></a> ⚙️ Node Parameters

### Minimax_H3_AutoContext_parameter (Parameter Group Node)

| Parameter | Default | Description |
|------|--------|------|
| long_prompt | — | The prompt (passed to the main node for inference; also used for the "expected segments" preview) |
| prompt_mode | auto | Prompt timeline mode (only effective in Clip_Frame) |
| clip_mode | Clip_Frame | Segmentation mode: Clip_Frame / Clip_Tag |
| clip_tag | 段1 | Clip_Tag split label template (must end with a numeric index) |
| prompt_format | official | Prompt output format: official / legacy / raw |
| crop_mode | stretch | Reference image / first-last frame / reference video scaling & cropping: center / stretch / none |
| ref_sync_mode | segmented | Whether reference video/audio is sliced per segment: global / segmented |
| width × height | 960×544 | First-pass resolution (overridden by `latent_input` during second pass) |
| total_frames | 362 | Total frames to generate (17n+5); overridden by the sum of segments in Clip_Tag mode |
| fps | 24 | Frame rate, used for audio sync and converting seconds in the prompt |
| chunk_frames | 90 | Frames per segment (17n+5) |
| context_frames | 22 | Inter-segment continuation frames (17n+5: 5/22/39/56…) |
| lock_audio | enable | Locks the audio region during second pass; only redraws video |
| audio_drive | disable | Audio drive switch |
| decode_output | disable | Whether to decode internally |

> The node displays a live "expected segments" preview on the widget (computed by front-end JS, not involved in inference).

### Minimax_H3_AutoContext_Sampler (Main Node)

| Parameter | Default | Description |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 model components |
| parameter | required | Parameter group input (from the parameter node) |
| sampler | optional | External sampler object (SAMPLER), overrides the built-in sampler_name/scheduler |
| sigmas | optional | Custom sigma sequence (SIGMAS), highest priority |
| latent_input | optional | Second-pass input latent (connecting enables second pass) |
| info | optional | Parameter inheritance input (multi-pass chaining, keeps segmentation consistent) |
| first_frame / last_frame | optional | First/last frame anchoring (FL2VA) |
| seed | 0 | Random seed (control_after_generate) |
| steps / cfg | 30 / 1.0 | Sampling steps / CFG |
| sampler_name / scheduler | euler / simple | Built-in sampler / scheduler |
| denoise | 1.0 | Redraw strength (when connected to sigmas and ≠1, final_sigmas = sigmas×denoise) |
| ref_image_N / ref_video_N / ref_video_audio_N / ref_audio_N | optional | Reference materials (Autogrow dynamic ports) |
| drive_audio | optional | Audio drive source |

## <a id="output"></a> 📤 Outputs

| Output | Description |
|------|------|
| **video_frames** | The full stitched video frame sequence (when `decode_output=enable`) |
| **audio** | The stitched audio (cosine crossfade between segments) |
| **latent** | Shared audio-video latent (NestedTensor), stitched after cropping the inter-segment replay region |
| **denoised_latent** | Clean x0 prediction (after merge), for second-pass relay / preview |
| **info** | Segmentation parameters (Dict), passed to the next main node's info input to keep multi-pass segmentation consistent |

> 💡 **Recommended usage**: `latent → VAE Decode` to get the video; for second pass, `latent / denoised_latent → upscale → second-pass node`.

## <a id="second-pass"></a> 🔄 Second Pass and SplitSigmas Hi/Lo Frequency

### Basic Second Pass (low-res first pass → hi-res second pass)

```
parameter node ──parameter──> main node (first pass, 864×480)
    └─ latent / denoised_latent ──> [Split AV] ──> video_latent ──> latent upscale ──> [Merge AV] ──> main node (second pass).latent_input
Second-pass node: parameter shared (or info inherited), denoise 0.4~0.6
```

- Second-pass resolution follows `latent_input`, ignoring the parameter node's width/height
- Second-pass inter-segment anchoring uses the clean x0, keeping seams continuous

### SplitSigmas Hi/Lo Frequency (save time, improve clarity)

> ⚠️ **audio constraint**: for the segmented node, the inter-segment audio ref anchoring fails when "stopping at mid sigmas" (semi-finished product), so hi/lo frequency **only applies to video**. Audio must be fully sampled + second-pass locked.

```
First-pass node: full sampling (no high_sigmas; audio fully denoised)
          → denoised_latent → split & upscale video (audio untouched) → merge → second-pass.latent_input
Second-pass node: sigmas ← low_sigmas (run only the low sigma segment to refine details)
          lock_audio = True (reuse the complete first-pass audio)
```

## <a id="seam"></a> 🧵 Seam Correction Node (Minimax_H3_Seam_Correction)

- Input `images` (decoded full video frames), outputs corrected `images`
- Automatically detects inter-segment seams (inter-frame brightness jumps + local peaks)
- Classification: **scene cut / transient flash / exposure gradient / motion stutter**
- Toggle parameters: `fix_flash` / `fix_exposure` / `fix_motion` / `fix_scene_cut` + `threshold` detection threshold
- Only applies `lerp` transitions to exposure/flash-type jumps; scene cuts are not smoothed by default

> Usage: `VAE Decode → H3_Seam_Correction → Save/Video`.

## <a id="prompt-examples"></a> ✍️ Prompt Writing Examples

### Timeline Mode (auto / timeline)

```text
0-5s: ...
5-10s: ...

integrated_multimodal_description
....

overall_soundscape
```

> Paragraphs with `0-5s` markers are sliced by time; paragraphs without markers (style/sound effects/prohibitions) are automatically merged into every window.

### Global Mode (global)

> The whole prompt is used for all segments, suitable for a one-shot take with uniform motion throughout.

### Clip_Tag Mode (label-based segmentation)

> Set `clip_mode` to `Clip_Tag` and fill `clip_tag` with the label template (must end with a numeric index).

**Label template examples**

| Template | Matches |
|------|------|
| `段1` | `段1` / `段2` / `段3` (prefix "段" + number) |
| `A01` | `A01` / `A02` / `A03` (prefix "A" + number) |
| `[片段001]` | `[片段001]` / `[片段002]` (prefix "[片段" + number + suffix "]") |

**Label syntax**: each label occupies its own line as a split point; a newline after the label is recommended. It also works without a newline (the separator is skipped to take the segment content):

```text
段1:3s
Video:
...
Audio design:
...


段2:3-8s
Video:
0-2s:
...
2-5s:
...
Audio design:
0-5s: ...
```

**Segment duration rules** (three-level priority):

1. Duration right after the label line: `段1:0-5s` → 5 s; `段1:3-8s` → 5 s (the duration marker is removed from the prompt)
2. Zero-based time markers within the segment: `【0-2s】`+`【2-5s】` → 5 s
3. Neither → falls back to `chunk_frames / fps`

**prompt_format selection**

- `official` / `legacy`: in-segment time markers are automatically rendered as relative in-segment coordinates
- `raw`: outputs as-is after removing labels; time markers stay unchanged (suitable for structured prompts generated by large models)

## <a id="limitations"></a> 📝 Prompt Notes (Node Limitations)

> The notes below **do not apply** to simple, always-valid prompt scenarios (i.e., all segments share the same prompt, global mode),
> such as: talking-head digital humans (of course, the lines still need segmentation), scenes where the camera/composition barely changes, or scenarios like character replacement in videos where the prompt stays generic.

### 1️⃣ Core Principle: Temporal Exclusivity

> When using segmented inference (Chunk), please strictly follow the **temporal exclusivity** principle — each segment's prompt may only describe what is "happening" in that segment, i.e., the **new changes** relative to the end of the previous segment.

- **Segmentation is a "relay race"**: when segment N is generated, its starting visual state (position, pose, camera position) is fully provided implicitly by the "anchoring frames (Context Frames)" at the end of the previous segment. You do not need to re-describe this starting state in the prompt.
- **No "flashbacks" or "overlaps"**: segment N's prompt must never repeat actions or camera movements already completed in segment N-1. If you do, the model receives instructions that logically conflict with the anchoring frames (instruction conflict), causing stuttering, broken motion logic, or repeated actions.
- **Reset at boundaries**: when switching segments, reset the previous segment's "ongoing actions". The new segment's prompt should act like a "new instruction after pressing the shutter", describing only the displacement, actions, or new elements occurring within the new time window.

**❌ Incorrect example (conflicting overlap)**

```text
段1: 3s
"Object A moves toward position B" (covers 0-5s)
段2: 3-6s
"After object A reaches position B, it turns around at position B" (covers 5-10s)
```

> Problem analysis: when segment 1 ends, the anchoring frames show object A has already arrived at position B and just stopped. But segment 2's prompt forcibly requires "object A moves to position B", which conflicts with the "already arrived" static result in the anchoring frames. The model will try to "move it again", causing glitchy or jumpy frames.

**✅ Correct example (seamless progression)**

```text
段1: 3s
"Object A moves toward position B and finally stops at position B" (emphasizes the action loop closure)
段2: 3-6s
"After standing still, object A slowly turns around" (directly describes the new action after the previous segment ends)
```

> Correct logic: segment 2 completely drops the description of the "movement process", treats "stopped at B" as an established fact, and only describes the new "turning" action, so the model can perfectly continue using the anchoring frames.

> 🚀 **In one sentence**: the end of the previous segment is the "result"; the start of the next segment is "the new action after the result". Don't write the "process that led to the result" into the next segment.

### 2️⃣ Core Principle: Per-Segment Reference Declaration

> When using segmented inference together with reference images/videos (image1, video1, etc.), please strictly follow the **per-segment reference declaration** principle — each segment's prompt must independently and completely declare all reference materials that segment needs; references are not "remembered" or "inherited" into the next segment.

- **No global memory**: the node parses the reference labels written in the current segment's prompt to precisely determine which materials that segment needs. Writing image1 in the previous segment only means that segment used it; the next segment is scanned anew.
- **Not written = not passed**: if segment N does not mention image1 again, that reference image will not be passed to that segment, causing character/object inconsistency.

**❌ Incorrect example (implicit inheritance)**

```text
段1[3s]: image1 is object A; object A is moving forward.
段2[3-6s]: Object A stops and turns to look at the camera. (image1 not written)
```

**✅ Correct example (explicit per segment)**

```text
段1[3s]: image1 is object A; object A is moving forward.
段2[3-6s]: image1 is object A; object A stops and turns to look at the camera.
```

<div align="center">

[![Chinese](https://img.shields.io/badge/Language-Chinese-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

> One-click MiniMax H3 long-video auto-generation node: **Segmented Inference + Inter-segment Continuation Anchoring + Prompt Timeline Slicing + Second Pass + Seam Correction**.

Under limited VRAM, long videos are split into multiple segments for independent inference; seamless inter-segment continuity is achieved through multi-frame keyframe anchoring, and prompts are auto-sliced along the timeline so each segment's content aligns with the prompt rhythm. Supports a second pass (low-res pass 1 + high-res pass 2).

## 📖 Table of Contents

- [Node List](#nodes)
- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [Second Pass & SplitSigmas High/Low Frequency](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Notes (Node Limitations)](#limitations)

## <a id="nodes"></a> 🧩 Node List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter-group node: centrally manages prompt / segmentation / resolution / audio parameters, outputs `parameter`, and previews "expected segmentation" in real time |
| **Minimax_H3_AutoContext_Sampler** | Main node: segmented inference + continuation anchoring + sampling (shared by pass 1 / pass 2) |
| **Minimax_H3_Seam_Correction** | Seam correction node: pixel-domain correction of inter-segment seams in the decoded video |

> Usage: `parameter node --parameter--> main node`. Fill in the prompt in the parameter node; the main node receives it via `parameter` (required).

## <a id="features"></a> ✨ Core Features

### 🧩 Segmented Inference

- Split into multiple segments by `total_frames` / `chunk_frames` (frame unit); the parameter step size is 17 (options: 5, 22, 39, 56, 73, 90, ...)
- Each segment's frame count snaps to the H3 17n+5 VAE temporal grid
- The last segment is preferentially extended to absorb the anchoring deficit (≤30% cap) to avoid tiny tail segments
- `fps` is only used for audio synchronization and seconds conversion in the prompt

### 🔗 Inter-segment Continuation Anchoring

- Extract N latent frames from the tail of the previous segment as independent keyframes, anchored to the corresponding time coordinates of the current segment
- The model sees a real motion sequence rather than a single static frame, correctly continuing motion direction and speed
- Context audio is passed through the ref channel (no `<Audio j>` tag inserted); the model treats it as "previous content" rather than reference material
- Inter-segment audio cosine cross-fade, strictly aligned with the video frame count

> 📐 **Frame accounting** (H3 17n+5 grid):
> - First segment = `17n+5` (includes the 5-frame "intro")
> - Non-first segments add **multiples of 17 frames** (e.g. 68/85 — the "pure bead runs" in the middle-to-late part, no intro)
> - `context_frames` (continuation anchoring) must be of the form `17n+5` (5/22/39/56…)
> - The whole latent timeline = intro (5) + N runs (17n), satisfying 17n+5 overall

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **auto** | Paragraphs with time marks (e.g. `0-5s`) are auto-split; unmarked paragraphs (style / sfx / banned items) are merged into every window |
| **timeline** | Force split by time marks |
| **global** | The whole prompt is used for all windows (suited to a one-shot scene with uniform action) |
| **sequential** | Laid onto the timeline in sentence reading order; paragraphs starting with `global:` / `[global]` are merged into every window |

### 🏷️ Clip_Tag Tag Segmentation Mode

- Split the prompt by user-defined tags (e.g. `段1` / `段2` / `段3`); each segment = one chunk = that segment's full prompt
- Segment duration is determined by the prompt content (three-level priority):
  1. Duration right after the tag line (e.g. `段1:0-5秒` → 5 s; `段1:3-8秒` → 5 s)
  2. The maximum end value of in-segment time marks (e.g. `【0-2秒】` + `【2-5秒】` → 5 s)
  3. `chunk_frames / fps` default fallback
- In-segment time marks are **relative time** (each segment starts at 0), not global absolute time
- Continuation anchoring compensation: non-first segments generate an extra `context_frames` for head anchoring, automatically trimmed after generation
- Total-duration anchoring: the difference between the sum of each segment's newly added frames and the target total frames is compensated from the last segment at 17-frame granularity, getting as close as possible to the target total duration
- Tags themselves are stripped during inference; the remaining prompt content is output according to `prompt_format`

### 🎯 Intelligent Reference Filtering (Image / Video / Audio)

- Parse the references in each segment's prompt and pass only the referenced materials to the current segment
- Numbers are automatically remapped to consecutive indices, aligned with the model's native tags
- A video is bound to its paired audio track; audio tracks and standalone audio are counted together as `<Audio N>`
- Avoids image/sound crosstalk caused by passing all references at once

### 🖌️ Second Pass

- Connecting pass-1 latent to the main node's `latent_input` (directly or via a latent upscaler) enters second-pass mode
- Second-pass spatial resolution **follows the input latent** (ignores the parameter node's width/height), enabling low-res pass 1 → high-res pass 2
- `denoise` controls redraw strength; `sigmas` supports custom sigma schedules (same as `SamplerCustomAdvanced`)
- `lock_audio` locks the audio region (noise_mask audio=0); pass 2 only redraws video and reuses pass-1 audio

### 🎵 Audio Drive

- `drive_audio` (AUDIO, optional) + `audio_drive` switch
- When enabled, `drive_audio` is locked into the audio half of the latent (noise_mask audio=0); the video is driven by the audio, and the output audio = the source audio itself

## <a id="install"></a> 📦 Installation

### Method One: Manual Installation

```bash
cd ./ComfyUI/custom_nodes
git clone https://github.com/supElement/ComfyUI_MinimaxH3_AutoContext.git
```

### Method Two: Install via Manager

Search `ComfyUI_MinimaxH3_AutoContext` in ComfyUI Manager and click Install.

## <a id="params"></a> ⚙️ Node Parameters

### Minimax_H3_AutoContext_parameter (parameter-group node)

| Parameter | Default | Description |
|------|--------|------|
| long_prompt | — | Prompt (passed to the main node for inference; also used for the "expected segmentation" preview) |
| prompt_mode | auto | Prompt timeline mode (only effective for Clip_Frame) |
| clip_mode | Clip_Frame | Segmentation mode: Clip_Frame / Clip_Tag |
| clip_tag | 段1 | Clip_Tag split tag template (must end with a number) |
| prompt_format | official | Prompt output format: official / legacy / raw |
| crop_mode | stretch | Reference image / first-last frame / reference video scaling & cropping: center / stretch / none |
| ref_sync_mode | segmented | Whether reference video/audio is sliced per segment: global / segmented |
| width × height | 960×544 | Pass-1 resolution (overridden by latent_input in pass 2) |
| total_frames | 362 | Total generated frames (17n+5); in Clip_Tag mode it is overridden by the sum of the segments |
| fps | 24 | Frame rate; used for audio sync and prompt seconds conversion |
| chunk_frames | 90 | Frames generated per segment (17n+5) |
| context_frames | 22 | Inter-segment continuation frames (17n+5: 5/22/39/56…) |
| lock_audio | enable | Lock the audio region; only redraw video |
| audio_drive | disable | Audio drive switch |
| decode_output | disable | Whether to decode internally |

> The node displays the "expected segmentation" preview in real time (frontend JS calculation; not involved in inference).

### Minimax_H3_AutoContext_Sampler (main node)

| Parameter | Default | Description |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 model components |
| parameter | required | Parameter-group input (from the parameter node) |
| sampler | optional | External sampler object (SAMPLER); overrides the built-in sampler_name/scheduler |
| sigmas | optional | Custom sigma schedule (SIGMAS); highest priority |
| latent_input | optional | Pass-2 input latent (connecting it enables pass 2) |
| info | optional | Parameter inheritance input (chained multi-pass; keeps segmentation consistent) |
| first_frame / last_frame | optional | First / last frame anchoring (FL2VA) |
| seed | 0 | Random seed (control_after_generate) |
| steps / cfg | 30 / 1.0 | Sampling steps / CFG |
| sampler_name / scheduler | euler / simple | Built-in sampler / scheduler |
| denoise | 1.0 | Redraw strength (with sigmas and ≠ 1, final_sigmas = sigmas × denoise) |
| ref_image_N / ref_video_N / ref_video_audio_N / ref_audio_N | optional | Reference materials (Autogrow dynamic ports) |
| drive_audio | optional | Audio drive source |

## <a id="output"></a> 📤 Output

| Output | Description |
|------|------|
| **video_frames** | Spliced complete video frame sequence (when `decode_output=enable`) |
| **audio** | Spliced audio (inter-segment cosine cross-fade) |
| **latent** | Shared audio/video latent (NestedTensor), spliced after trimming the inter-segment replay regions |
| **denoised_latent** | Clean x0 prediction (after merge), used for pass-2 relay / preview |
| **info** | Segmentation parameters (Dict), passed to the next main node's info input to keep chained passes' segmentation consistent |

> 💡 **Recommended usage**: `latent → VAE Decode`; for pass 2: `latent / denoised_latent → upscale → pass-2 node`.

## <a id="second-pass"></a> 🔄 Second Pass & SplitSigmas High/Low Frequency

### Basic second pass (low-res pass 1 → high-res pass 2)

```
parameter node ──parameter──> main node (pass 1, 864×480)
    └─ latent / denoised_latent ──> [split AV] ──> video_latent ──> latent upscale ──> [merge AV] ──> main node (pass 2).latent_input
pass-2 node: shared parameter (or info inheritance); optional denoise 0.4~0.6
```

- Pass-2 resolution follows `latent_input`, ignoring the parameter node's width/height

### SplitSigmas high/low frequency (save time, improve clarity)

> ⚠️ **audio constraint**: the segmented node's audio inter-segment ref anchoring fails when "stopping at a middle sigma" (half-finished), so high/low frequency **only applies to video**. Audio must be fully sampled.

```
Pass-1 node: full sampling (do not connect high_sigmas; audio fully denoised)
          → denoised_latent → split & upscale video (audio untouched) → merge → pass-2.latent_input
Pass-2 node: sigmas ← low_sigmas (only run the low-sigma range to add detail)
          lock_audio = True (reuse the complete pass-1 audio)
```

## <a id="seam"></a> 🧵 Seam Correction Node (Minimax_H3_Seam_Correction)

- Input `images` (decoded full video frames), output corrected `images`
- Automatically detects inter-segment seams: local peaks of the frame-to-frame luma difference (downsampled); a value above `threshold` marks a candidate seam
- Each seam is classified as: **Scene Cut (scene_cut) / Transient Flash (flash) / Exposure Drift (exposure) / Motion Seam (motion)**
- For hit seams, applies **GPU optical-flow alignment (warp) → local color matching → multi-frame blending over a temporal window**, rather than a simple lerp
- Each category is controlled independently by its own toggle; `fix_scene_cut` is off by default (scene cuts are usually left untouched)

| Parameter | Default | Description |
|------|------|------|
| fix_flash | true | Fix transient flash / brightness jumps |
| fix_exposure | true | Fix exposure drift |
| fix_motion | true | Fix motion seams (optical-flow alignment) |
| fix_scene_cut | false | Fix scene cuts; usually leave off |
| threshold | 0.05 | Seam detection threshold (frame-to-frame luma difference) |
| seam_window | 3 | Number of frames blended on both sides of a seam |
| flow_strength | 0.75 | GPU optical-flow displacement compensation strength |
| blend_strength | 0.65 | Local color/brightness continuity correction strength |
| flow_iterations | 12 | Optical-flow iterations; higher is slower |
| flow_pyramid | 3 | Optical-flow pyramid levels |
| use_gpu | true | Use CUDA GPU for optical flow and seam processing |

> Usage: `VAE Decode → H3_Seam_Correction → Save/Video`.

> ⚠️ Note: This node only performs pixel-domain seam correction and cannot fix artifacts produced upstream in the second pass. If the second-pass input latent was zero-padded due to a resolution mismatch (visible as a flickering strip/glow at the bottom of the frame), this node may misclassify that strip as a brightness jump and amplify the flicker — make sure the second-pass resolution is 32-aligned and the latent spatial size is even before enabling this node.

## <a id="prompt-examples"></a> ✍️ Prompt Writing Examples

### Timeline Mode (auto / timeline)

```text
0-5s: ...
5-10s: ...

integrated_multimodal_description
....

overall_soundscape
```

> Paragraphs containing `0-5s` marks are split by time; unmarked paragraphs (style / sfx / banned items) are automatically merged into every window.

### Global Mode (global)

> The entire prompt is used for all segments; suited to a one-shot scene with uniform action throughout.

### Clip_Tag Mode (Segment by Tag)

> Set `clip_mode` to `Clip_Tag` and fill in the tag template (`clip_tag`) (must end with a number).

**Tag Template Examples**

| Template | Match |
|------|------|
| `段1` | `段1` / `段2` / `段3` (prefix "段"+number) |
| `A01` | `A01` / `A02` / `A03` (prefix "A"+number) |
| `[片段001]` | `[片段001]` / `[片段002]` (prefix "[片段"+number+suffix"]") |

**Tag Writing**: a tag occupies a line alone as the split point; a line break after the tag is recommended. It also works without a line break (the separator is skipped and the segment content is taken):

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

**Segment Duration Rules** (three-level priority):

1. Duration right after the tag line: `段1:0-5秒` → 5 s; `段1:3-8秒` → 5 s (the duration mark is removed from the prompt)
2. In-segment time marks, 0-based: `【0-2秒】` + `【2-5秒】` → 5 s
3. Neither present → `chunk_frames / fps` fallback

**prompt_format Selection**

- `official` / `legacy`: in-segment time marks are automatically rendered as in-segment relative coordinates
- `raw`: tags are removed and output as-is; time marks stay unchanged (suited to structured prompts generated by large models)

## <a id="limitations"></a> 📝 Prompt Notes (Node Limitations)

> The following notes **do not apply** to simple, always-valid prompt scenarios (i.e., all segments share the same prompt, global mode),
> e.g. talking-head digital humans (of course, dialogue needs segmentation), videos with little change in shot/composition, or character-replacement scenarios where the prompt stays generic.

### 1️⃣ Core Principle: Temporal Exclusivity

> When using segmented inference (chunking), you must follow the **temporal exclusivity** principle — each segment's prompt can only describe the **new changes** "happening" in that segment relative to the end of the previous segment.

- **Segmentation is a "relay"**: when the Nth segment is generated, its starting visual state (position, pose, camera position) is fully provided implicitly by the "anchoring frames (Context Frames)" at the end of the previous segment. You don't need to re-describe that starting state in the prompt.
- **No "retrospection" or "overlap"**: the Nth segment's prompt must never re-describe actions or camera moves already completed in the N-1th segment. Repeating them makes the model receive instructions that logically conflict with the anchoring frames (instruction conflict), causing stuttering, confused motion logic, or repeated actions.
- **Zero the boundary**: when switching segments, clear the previous segment's "action in progress". The new segment's prompt should be like "new instructions after pressing the shutter" — only describe the displacement, action, or new elements occurring in the new time window.

**❌ Incorrect Writing (Conflict/Overlap)**

```text
Segment 1: 3 seconds
"Object A moves toward position B" (covering time period 0-5s)
Segment 2: 3-6 seconds
"After object A moves to position B, it turns around at position B" (covering time period 5-10s)
```

> Problem analysis: at the end of segment 1, the anchoring frame shows object A has already arrived at position B and just stopped. But segment 2's prompt forcibly requires "object A to move to position B", which conflicts with the "already arrived" static result of the anchoring frame; the model tries to "move again", causing glitching or skipped frames.

**✅ Correct Writing (Seamless Progression)**

```text
Segment 1: 3 seconds
"Object A moves toward position B and finally stops at position B" (emphasizing action closure)
Segment 2: 3-6 seconds
"After standing still, object A slowly turns its direction" (directly describing the new action after the previous segment ends)
```

> Correct logic: segment 2 completely abandons describing the "moving process", treats "stopped at B" as a given fact, and only describes the following "turning" new action — the model can then continue seamlessly using the anchoring frames.

> 🚀 **One-sentence summary**: the end of the previous segment is the "result", and the start of the next segment is the "new action after the result" — don't write the "process that led to the result" into the next segment.

### 2️⃣ Core Principle: Per-Segment Reference Declaration

> When using segmented inference together with reference images/videos (image1, video1, etc.), you must follow the **per-segment reference declaration** principle — every segment's prompt must independently and completely declare all the reference materials it needs; references are not "remembered" or "inherited" into the next segment.

- **No global memory**: the node parses the reference tags written in the current segment's prompt to precisely decide which materials that segment needs. Writing image1 in the previous segment only means the previous segment used it; the next segment is scanned anew.
- **Not written, not passed**: if the Nth segment doesn't mention image1 again, that segment won't receive that reference image, causing character/object inconsistency.

**❌ Incorrect Writing (Implicit Inheritance)**

```text
Segment 1 [3s]: image1 is object A; object A is moving forward.
Segment 2 [3-6s]: Object A stops and turns to face the camera. (image1 not written)
```

**✅ Correct Writing (Explicit Per Segment)**

```text
Segment 1 [3s]: image1 is object A; object A is moving forward.
Segment 2 [3-6s]: image1 is object A; object A stops and turns to face the camera.
```

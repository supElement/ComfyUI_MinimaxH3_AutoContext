<div align="center">

[![Chinese](https://img.shields.io/badge/Language-Chinese-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

One-click MiniMax H3 Long Video Auto-Generated Node: **Segmented Inference + Segment Transition Anchoring + Prompt Timeline Slicing + Secondary Sampling (SS) + Seam Correction**.
Split long videos into multiple segments for independent inference under limited GPU memory, achieve seamless connection between segments through superposition enhancement, and automatically slice prompts along the time axis to align the generated content with the prompt rhythm. Also, slice and align audio-video references. Supports SS (Low-resolution One-pass + High-resolution SS).
Supports latent caching and retrieval, making it convenient to quickly skip the already inferred segments and cache files when the inference is interrupted for some reason, storing cache files by segment. Reading the existing latent cache files under the same upstream parameters in the sampling node.

Note: When changing the model, including lora, sageattention, etc., latent detection will not find changes, so latent cache must be deleted. Two methods to delete latent cache:
- Enable the clear_cache parameter on the Minimax_H3_AutoContext_Sampler node, which will force a complete re-establishment of the cache file of this node when the sampling starts.
- Manually delete the corresponding folder in the cache directory (\ComfyUI\output\cache), named "node_" + "node ID".

<img width="2230" height="976" alt="image" src="https://github.com/user-attachments/assets/5634914a-6f98-4d4f-b573-2c8b41e0c57e" />


<img width="2209" height="1030" alt="image" src="https://github.com/user-attachments/assets/9bbdda2a-d4ce-4836-b108-e359e72e31de" />

## Bug Fixes and Optimizations

V0.6.5

- Optimized latent cache processing logic, removed manual cache directory specification, and changed it to automatically specify a unique cache directory for each node ("node + node ID"), preventing the latent cache logic of sampling nodes from being overlapped due to incorrect operations.
- Established cache and verification logic in segmented mode. If the upstream node only adds prompts or increases segments without changing the other prompts submitted to the sampling, and the other parameters associated with the sampling node do not change, the existing corresponding cache is still considered valid and called. New segments will also automatically establish latent cache. Downstream sampling nodes (SS) will also retain the existing latent cache and call them, only creating caches for the added segments.
- The position of the prompt change determines which latent caches can be reused. Segments after the changed prompts will be forcibly rebuilt, and the same processing logic is applied to downstream nodes.

V0.5.8
- Improved hash value detection parameters, solved the error of tensor mismatch caused by the change of parameter parameters in the upstream node of the sampler.
- Removed the lens detection model from the Minimax_H3_Seam_Correction node, as the detection model would cause the preview of the sampling node to go "white screen". Replaced it with PySceneDetect method (pure CPU, no potential pollution).

## 📖 Directory

- [Node List](#nodes)
- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [Second Pass and SplitSigmas High-Low Frequency](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Notes (Node Limitations)](#limitations)

## <a id="nodes"></a> 🧩 Node List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter Group Node: Centralized management of prompts/segments/resolution/audio, output `parameter`, and preview "Estimated Segments" in real-time |
| **Minimax_H3_AutoContext_Sampler** | Main Node: Segmented Inference + Transition Anchoring + Sampling (common for one-pass and SS) |
| **Minimax_H3_Seam_Correction** | Seam Correction Node: Pixel Domain Correction for Segment Joints in Decoded Video |

> Usage: `parameter node --parameter--> Main Node`. Prompts are filled in the parameter node, and the main node receives `parameter` (required) through `parameter`.

## <a id="features"></a> ✨ Core Features

### 🧩 Segmented Inference

- Split into multiple segments according to `total_frames` / `chunk_frames` (frame unit), frame numbers are recommended to be 5, 22, 39, 56, 73, 90...
- The last segment is automatically extended to make up for it, avoiding too short tail segments
- `fps` is only used for audio synchronization and prompt second conversion

### 🔗 Segment Transition

- **Superposition Enhancement**: Automatically "hand off" the end of the previous segment to the current segment for non-first segments, so that the new content extends naturally from the end of the previous segment, eliminating stalling or position jumps at the junction.
- The end of the previous segment is passed as a motion reference to the current segment to help maintain the direction and speed of motion.
- The audio of the previous segment is also passed in as "previous content" to help the sound naturally continue.
- Segment audio is smoothly faded out and aligned with video frame numbers

> Frame number rules: `total_frames` / `chunk_frames` / `context_frames` are all 5, 22, 39, 56, 73, 90... (multiples of 17 plus 5), the node will automatically align, generally no need to calculate manually.

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **auto** | Paragraphs with time marks (such as `0-5s`) are automatically segmented, and paragraphs without marks (style/sound/forbidden items) are spliced into each window |
| **timeline** | Force segmentation according to time marks |
| **global** | The entire prompt is used for all segments (suitable for one-shot action throughout the entire shot) |
| **sequential** | Laid out on the time axis according to the order of reading, paragraphs starting with `Global:`/`[Global]` are spliced into each window |

### 🏷️ Clip_Tag Segment Mode

- Segment prompts according to user-defined tags (such as `Segment1`/`Segment2`/`Segment3`), each segment = one chunk = all prompts of the segment
- Segment duration is determined by the prompt content (three levels of priority):
  1. Time duration following the tag line (such as `Segment1:0-5 seconds` → 5 seconds; `Segment1:3-8 seconds` → 5 seconds)
  2. Maximum end value of time marks within the segment (such as `【0-2 seconds】`+`【2-5 seconds】` → 5 seconds)
  3. Default value of `total_frames / fps` (single segment aligns with `total_frames`)
- Segment time marks are **relative time**, not global absolute time
- An additional segment is generated for non-first segments for overlap, which is automatically trimmed after generation
- Total duration is automatically aligned with the target total frame number, as close as possible to the expected duration
- Tags are removed during inference, and the rest of the prompt content is output according to `prompt_format`

### 🎯 Reference Intelligent Filtering (Image / Video / Audio)

- Automatically identify the referenced images/videos/audio used in each segment of the prompt, and only pass the materials referenced in the segment to it
- Video is bound with its corresponding audio track, avoiding interference between image and sound

### 🖌️ Secondary Sampling (SS)

- The main node `latent_input` connects to the one-pass latent (or after latent amplification) to enter the SS mode
- SS resolution is **based on the input latent**, ignoring width/height, to achieve low-resolution one-pass → high-resolution SS
- `denoise` controls the strength of redrawing; `sigmas` supports custom sigma sequences (same as `SamplerCustomAdvanced`)
- `lock_audio`: SS only redraws video, reuse audio from the one-pass

### 🎵 Audio Drive

- `drive_audio` (AUDIO, optional) + `audio_drive` switch
- When enabled, the video follows the audio to generate, the output audio = source audio itself (mouth shape/rhythm driven by it)

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
| long_prompt | — | Prompt (given to the main node for inference and also used for "Estimated Segments" preview) |
| prompt_mode | auto | Prompt Timeline Mode (only Clip_Frame takes effect) |
| clip_mode | Clip_Frame | Segmentation Mode: Clip_Frame / Clip_Tag |
| clip_tag | Segment1 | Clip_Tag Split Tag Template (must end with a digit) |
| prompt_format | official | Prompt Output Format: official / legacy / raw |
| crop_mode | stretch | Reference Image/First and Last Frame/Reference Video Scaling and Cropping: center / stretch / none |
| ref_sync_mode | segmented | Whether Reference Video/Audio is Sliced by Segment: global / segmented |
| width × height | 960×544 | One-pass Resolution (ignored by latent_input during SS) |
| total_frames | 362 | Total Number of Generated Frames (17n+5); In Clip_Tag, segments without explicit time duration are covered by `total_frames` |
| fps | 24 | Frame Rate, Used for Audio Synchronization and Prompt Second Conversion |
| chunk_frames | 90 | Number of Generated Frames per Segment (17n+5) |
| context_frames | 22 | Number of Frames for Segment Transition (17n+5: 5/22/39/56…) |
| lock_audio | enable | Lock Audio Area, Only Redraw Video |
| audio_drive | disable | Audio Drive Switch |


> A real-time preview of "Estimated Segments" is displayed on the node (calculated by front-end JS, not involved in inference).

### Minimax_H3_AutoContext_Sampler (Main Node)

| Parameter | Default Value | Description |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 Model Component |
| parameter | Required | Parameter Group Input (from parameter node) |
| sampler | Optional | External Sampler Object (SAMPLER), overrides built-in sampler_name/scheduler |
| sigmas | Optional | Custom Sigma Sequence (SIGMAS), highest priority |
| latent_input | Optional | SS Input Latent (connects to enable SS) |
| info | Optional | Parameter Inheritance Input (multi-sampling, ensures consistent segmentation) |
| first_frame / last_frame | Optional | First/Last Frame Anchoring (FL2VA) |
| video_context_denoise | 0.0 | Strength of Segment Transition (only for non-first segments): 0=exact continuation of the end of the previous segment, 1=re-generated, intermediate values=soft mixing. When connected to SplitSigmas, it is recommended to set 1 to avoid screen artifacts. |
| seed | 0 | Random Seed (control_after_generate) |
| steps / cfg | 30 / 1.0 | Sampling Steps / CFG |
| sampler_name / scheduler | euler / simple | Built-in Sampler / Scheduler |
| denoise | 1.0 | Redrawing Strength (1=full resampling, the smaller the more original structure is retained) |
| enable
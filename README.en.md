<div align="center">

[![Chinese](https://img.shields.io/badge/Language-Simplified%20Chinese-red?style=for-the-badge)](./README.md)
[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](./README.en.md)

</div>

# ComfyUI_MinimaxH3_AutoContext

One-click MiniMax H3 long video automatic generation node: **segmented inference + inter-segment anchoring + prompt timeline slicing + secondary sampling (secondary sample) + seam correction**.
Under limited GPU memory, long videos are divided into multiple segments for independent inference, and seamless connection between segments is achieved through superposition enhancement. At the same time, prompts are automatically sliced along the timeline to align the generated content with the prompt rhythm. The audio and video references are also sliced and aligned. Only the references mentioned in the current segment participate in the inference. Supports secondary sampling (low-quality one sample + high-quality secondary sample).
Supports latent caching and retrieval, which is convenient for quickly skipping the inferred segments and reading the cached files in case of inference interruption due to some reasons, caching files are stored by segments, and latent cache files are read under the condition that the upstream parameters of the sampling node remain unchanged.

Note: When changing the model, including lora, sageattention, etc., latent detection will not find the change, so it is necessary to delete the latent cache. There are two ways to delete the latent cache:
- Enable the clear_cache parameter on the Minimax_H3_AutoContext_Sampler node, which will force a new establishment of the cache file for this node at the beginning of sampling.
- Manually delete the corresponding folder in the cache directory (\ComfyUI\output\cache), the folder name is "node_" + "node ID".

<img width="2230" height="976" alt="image" src="https://github.com/user-attachments/assets/5634914a-6f98-4d4f-b573-2c8b41e0c57e" />


<img width="2209" height="1030" alt="image" src="https://github.com/user-attachments/assets/9bbdda2a-d4ce-4836-b108-e359e72e31de" />

## Bug Fixes and Optimizations

V0.6.5

- Optimized latent cache processing logic, removed manual cache directory specification, and changed it to automatically specify a unique cache directory for each node ("node + node ID"), to prevent the latent cache logic of sampling nodes from being overlapped due to misoperation.
- Caching and verification logic are established in segmented mode. If the upstream node only adds prompts or increases segments without changing the other prompts submitted to sampling, and the other parameters associated with the sampling node do not change, the existing corresponding cache is still considered valid and called, and the new added segment will also automatically establish latent cache. The downstream sampling node (secondary sampling) will also retain the existing latent cache and call it, only a new cache for the increased segment will be created.
- The change position of the prompt determines which latent caches can be reused. The segments after the changed prompt must be rebuilt, and the downstream nodes use the same processing logic.
- ignore_latent_hash, ignore the hash value check of the input port input_latent. Practical scenarios: some latent processing nodes will change the latent judgment information (such as: Minimax H3 Latent Upscaler (3D) node), causing the slight change of latent to make the latent cache unusable, wasting inference time. In this case, it is recommended to set it to true. I only tested the Minimax_H3-LatentUpscaler_Adv node in my another repository github.com/supElement/ComfyUI_Element_easy extension, similar nodes have not been tested. When using latent processing nodes that do not change the latent noise features, the ignore_latent_hash parameter can be set to false.

V0.5.8
- Improved hash value detection parameters, solved the error of tensor mismatch caused by the change of parameter parameters in the upstream node of the sampler.
- Minimax_H3_Seam_Correction node, removed the lens detection model, and replaced it with PySceneDetect method (pure CPU, no potential pollution).

## 📖 Directory

- [Node List](#nodes)
- [Core Features](#features)
- [Installation](#install)
- [Node Parameters](#params)
- [Output](#output)
- [Secondary Sampling and SplitSigmas High and Low Frequencies](#second-pass)
- [Seam Correction Node](#seam)
- [Prompt Writing Examples](#prompt-examples)
- [Prompt Considerations (Node Limitations)](#limitations)

## <a id="nodes"></a> 🧩 Node List

| Node | Description |
|------|------|
| **Minimax_H3_AutoContext_parameter** | Parameter group node: centrally manage prompt/segment/resolution/audio, output `parameter`, and preview "expected segment" in real time |
| **Minimax_H3_AutoContext_Sampler** | Main node: segmented inference + anchoring + sampling (commonly used by one sample and secondary sample) |
| **Minimax_H3_Seam_Correction** | Seam correction node: pixel domain correction for segment junctions of decoded video |

> Usage: `parameter node --parameter--> main node`. Prompts are filled in the parameter node, and the main node receives `parameter` (required) through `parameter`.

## <a id="features"></a> ✨ Core Features

### 🧩 Segmented Inference

- Divided into multiple segments according to `total_frames` / `chunk_frames` (frame unit), frame number suggestions: 5, 22, 39, 56, 73, 90...
- The last segment is automatically extended to make up for it, avoiding too short tail segments
- `fps` is only used for audio synchronization and prompt second conversion

### 🔗 Inter-segment Continuation

- **Superposition Enhancement**: The non-first segment automatically "relays" the end of the previous segment, and the new content is naturally extended from the end of the previous segment, eliminating stalling or position jump at the junction.
- The end of the previous segment is transmitted as a motion reference to the current segment to help continue the direction and speed of motion.
- The audio of the previous segment is also transmitted as "previous content" to help the sound continue naturally.
- The audio between segments is smoothly faded out, aligned with the number of video frames

> Frame number rules: `total_frames` / `chunk_frames` / `context_frames` all take 5, 22, 39, 56, 73, 90... (17 times the multiple plus 5), the node will automatically align, generally no need to calculate manually.

### ⏱️ Prompt Timeline

| Mode | Description |
|------|------|
| **Clip_Tag** | Slice prompts according to user-defined tags (such as `Segment 1`/`Segment 2`), each tag corresponds to an independent video segment; segment duration is determined by the prompt content (tag duration > time mark within segment > `total_frames/fps` as a floor). |
| **timeline** | Slice prompts according to explicit time marks (such as `0-2s`/`2-6s`), each time interval corresponds to a video segment; segment duration = interval length × `fps` and automatically吸附 to a legal grid; **ignore `total_frames` and `chunk_frames`**. The global segment (`【Global】`) remains in the original position and will not be extracted. |
| **sequential** | Distribute prompts uniformly along the entire video timeline according to the order of reading, do not slice prompts themselves; video segmentation is still performed according to `chunk_frames`. | 
| **global** | The entire prompt is used for all video segments (strip `【Global】` tag), video segmentation is performed according to `chunk_frames`. |

> In `Clip_Tag` and `timeline` modes, `total_frames` and `chunk_frames` parameters are ignored (segment length is determined by the prompt), only when the mode is downgraded (such as no tag/time mark detected) will these values be used.

### 🏷️ Clip_Tag Tag Segmentation Mode

- Slice prompts according to user-defined tags (such as `Segment 1`/`Segment 2`/`Segment 3`), each segment = one chunk = all prompts in this segment
- Segment duration is determined by the prompt content (three levels of priority):
  1. Duration following the tag line (such as `Segment 1:0-5 seconds` → 5 seconds; `Segment 1:3-8 seconds` → 5 seconds)
  2. Maximum end value of time mark within the segment (such as `【0-2 seconds】`+`【2-5 seconds】` → 5 seconds)
  3. `total_frames / fps` default value as a floor (total frame number fits `total_frames` for a single segment)
- Segment time marks are **relative time** (each segment starts from 0), not global absolute time
- An additional segment of overlapping frames is generated for non-first segment to facilitate connection, and the generated frames are automatically cropped off
- Total duration is automatically aligned with the target total frame number, as close as possible to the expected duration
- The tag itself is removed during inference, and the rest of the prompt content is output according to `prompt_format`

### 🎯 Reference Intelligent Filtering (Image / Video / Audio)

- Automatically identify the referenced images/videos/audio used in each prompt, and only transmit the materials mentioned to the segment
- The video is bound with its matching audio track to avoid interference between image and sound

### 🖌️ Secondary Sampling (Secondary Sample)

- The main node `latent_input` connects to the one sample latent (or expanded by latent) to enter the secondary sampling mode
- The resolution of secondary sampling is **based on the input latent** (ignore width/height), realizing low-quality one sample → high-quality secondary sample
- `denoise`
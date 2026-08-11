# ComfyUI_MinimaxH3_AutoContext
初步实现了Minimax_H3的长视频拆分为多段独立推理，但还是有很多问题，音视频参考还没调试，先放到这里。


# 说明

一键式 MiniMax H3 长视频自动化生成节点：**分段推理 + 段间续接锚定 + 提示词时间轴切片**。

在显存有限的情况下，将长视频拆分为多段独立推理，通过多帧 keyframe 锚定实现段间无缝衔接，同时按时间轴自动切分提示词，让每段生成内容与提示词节奏对齐。

## 核心特性

### 分段推理
- 自动将目标时长按 `chunk_seconds` 拆分为多段，每段帧数吸附到 H3 的 17n+5 VAE 时序网格
- 末段优先加长吸收锚定亏空（≤30% 上限），避免产生微小尾段
- 帧数账目精确：生成帧数 = 用户指定时长 × fps

### 段间续接锚定
- 从上一段尾部提取 N 个 latent 帧作为独立 keyframe，锚定到当前段对应时间坐标
- 模型看到真实运动序列而非单张静帧，正确延续运动方向与速度
- 上下文音频通过 ref 通道传入（不插入 `<Audio j>` 标签），模型视为"之前的内容"而非参考素材
- 段间音频 cosine 交叉淡化，与视频帧数严格对齐

### 提示词时间轴
- **auto**：段落含时间标记（如 `0-5s`）自动切分，无标记段落（风格/音效/禁止项）拼入每个窗口
- **timeline**：强制按时间标记切分
- **global**：整段提示词用于所有窗口（适合全程同质动作的一镜到底）
- **sequential**：按句读顺序铺到时间轴，以 `全局:`/`[全局]` 开头的段落拼入每个窗口

### 参考图智能过滤
- 解析每段提示词中的 `imageN` 引用，只传被引用的参考图给当前段
- 编号自动重映射为连续序号，匹配模型的 `<Image 0>` 标签
- 避免所有参考图同时传入导致的画面串扰

### 动态端口（Autogrow）
- 默认只显示 `_0` 端口，连接后才显示 `_1`，以此类推
- 支持：参考图片（0-9）、参考视频（0-3）、参考视频配对音轨（0-3）、独立参考音频（0-3）
- `first_frame` / `last_frame` 始终显示（用于 FL2VA 首尾帧锚定模式）

### 其他
- 节点进度条与预览图实时显示
- PackedLayout 时间坐标修正：keyframe 的 `cond_t` 基于视频段实际起始坐标，而非 text_len
- extra_conds 拼接修复：keyframe cond rows 与 ref rows 拼接而非覆写

## 安装

将 `ComfyUI_MinimaxH3_AutoContext` 文件夹放入 ComfyUI 的 `custom_nodes` 目录，重启 ComfyUI 即可。

依赖：`torchaudio`（用于音频重采样，可选）。

## 节点参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| model / vae / audio_vae / clip | — | MiniMax H3 模型组件 |
| first_frame | 可选 | 首帧锚定（FL2VA 模式） |
| last_frame | 可选 | 尾帧锚定（FL2VA 模式） |
| long_prompt | — | 长提示词，支持时间标记分段 |
| prompt_mode | auto | 提示词时间轴模式 |
| width × height | 960×544 | 生成分辨率 |
| total_seconds | 15.0 | 总时长（秒） |
| fps | 24 | 帧率 |
| chunk_seconds | 5.0 | 每段时长（秒） |
| context_frames | 22 | 段间续接帧数，值越大连续性越强 |
| steps / cfg / sampler / scheduler | 30 / 1.0 / euler / simple | 采样参数 |
| seed | 0 | 随机种子 |
| ref_image_N | 可选 | 参考图片（提示词中用 `image0`/`image1` 引用） |
| ref_video_N | 可选 | 参考视频帧（24fps, 2-15s） |
| ref_video_audio_N | 可选 | 同编号参考视频的配对音轨 |
| ref_audio_N | 可选 | 独立参考音频 |

## 提示词写法示例

### 时间轴模式（auto / timeline）

```
0-5s: 镜头跟拍，场景过渡为中式园林
5-10s: 镜头穿过园林，...入画

integrated_multimodal_description（多模态整体画面描述）
你的提示词

overall_soundscape（整体环境音效）
你的提示词

```

含 `0-5s` 标记的段落按时间切分，无标记段落（风格/音效/禁止项）自动拼入每个窗口。

### 全局模式（global）

整段提示词用于所有分段，适合全程同质动作的一镜到底。

## 输出

- **video_frames**：拼接后的完整视频帧序列
- **audio**：拼接后的音频（段间 cosine 交叉淡化）

## 技术细节

### 17n+5 VAE 时序网格

H3 的 VAE 要求帧数满足 `17n+5` 网格（5, 22, 39, 56, 73, 90, 107, 124, ...）。节点自动将每段帧数吸附到最近的网格点，并动态计算分段策略。

### 多帧 keyframe 锚定

续接时从上一段尾部提取 N 个 latent 帧作为独立 keyframe。每个 keyframe 锚定到当前段的 `pixel_index` 对应时间坐标：

```
cond_t = video_t0 + FRAME_RESCALE × pixel_index
```

其中 `video_t0` 是视频段在 packed sequence 中的实际起始位置（考虑 ref 块的 cursor 偏移）。

### Monkey-patch

节点运行时自动应用两个 patch：

1. **PackedLayout.\_\_init\_\_**：接受任意 `resolved_frame_index`
2. **MiniMaxH3.extra_conds**：将 keyframe latent 与 ref latent 拼接

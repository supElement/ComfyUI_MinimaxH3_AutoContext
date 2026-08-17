# ComfyUI_MinimaxH3_AutoContext

一键式 MiniMax H3 长视频自动化生成节点：**分段推理 + 段间续接锚定 + 提示词时间轴切片**。

在显存有限的情况下，将长视频拆分为多段独立推理，通过多帧 keyframe 锚定实现段间无缝衔接，同时按时间轴自动切分提示词，让每段生成内容与提示词节奏对齐。

## 核心特性

### 分段推理
- 按 `total_frames` / `chunk_frames`（帧单位）拆分为多段，参数步幅为 17（可选 5, 22, 39, 56, 73, 90, ...）
- 每段帧数吸附到 H3 的 17n+5 VAE 时序网格
- 末段优先加长吸收锚定亏空（≤30% 上限），避免产生微小尾段
- `fps` 仅用于音频同步和提示词内秒数换算

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

### Clip_Tag 标签分段模式
- 按用户自定义标签（如 `段1`/`段2`/`段3`）切分提示词，每段 = 一个 chunk = 该段全部提示词
- 段时长由提示词内容决定（三层优先级）：
  1. 标签行紧跟的时长（如 `段1:0-5秒` → 5 秒；`段1:3-8秒` → 5 秒）
  2. 段内时间标记的最大结束值（如 `【0-2秒】`+`【2-5秒】` → 5 秒）
  3. `chunk_frames / fps` 默认值兜底
- 段内时间标记是**相对时间**（每段从 0 开始），不是全局绝对时间
- 续接锚定补偿：非首段多生成 `context_frames` 帧用于头部锚定，生成后自动裁掉，保证段间连续
- 推理时标签本身会被去除，其余提示词内容根据 `prompt_format` 输出：
  - `official`/`legacy`：时间坐标自动转为段内相对
  - `raw`：去标签后原样输出，不做任何转换

### 参考智能过滤（图 / 视频 / 音频）
- 解析每段提示词中的引用（`image1`/`image 1`/`图像1`/`图片1`、`视频1`、`音频1`/`audio 1`，以及原生 `<Picture N>`/`<Video N>`/`<Audio N>`，1 基），只传被引用的参考给当前段
- 编号自动重映射为连续序号，与模型原生 `<Picture N>`/`<Video N>`/`<Audio N>` 标签对齐
- 视频与其配对音轨绑定；音轨与独立音频按官方规则统一计数为 `<Audio N>`（音轨在前）
- 避免所有参考同时传入导致的画面/声音串扰

### 动态端口（Autogrow）
- 默认只显示 `_0` 端口，连接后才显示 `_1`，以此类推
- 支持：参考图片（0-9）、参考视频（0-3）、参考视频配对音轨（0-3）、独立参考音频（0-3）
- `first_frame` / `last_frame` 始终显示（用于 FL2VA 首尾帧锚定模式）

### 音频驱动（Audio Drive，参考音频驱动视频）
- 新增 `drive_audio`（AUDIO，可选）输入 + `audio_drive`（disable/enable，默认 disable）开关
- 开启后：把 `drive_audio` 编码后锁进 latent 的音频半区，并设 `noise_mask`（视频=1 正常去噪、音频=0 不重新生成），
  让视频生成被这条音频"驱动"（口型/节奏对齐），同时**输出音频 = `drive_audio` 本身**（无损，绕过 VAE 往返）
  —— `ref_audio_0` 负责语义驱动，`drive_audio` 负责锁定输出
- 每段按输出时间轴切片源音频（含非首段续接重演头），段间拼接后与视频逐帧对齐
- 源音频短于目标时长时自动补零，长于时自动截断

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
| clip_mode | Clip_Frame | 分段模式：Clip_Frame（按帧均匀切）/ Clip_Tag（按自定义标签切分） |
| clip_tag | 段1 | Clip_Tag 模式的分割标签模板，必须以数字序号结尾（如 `段1`/`A01`/`[片段001]`） |
| prompt_mode | auto | 提示词时间轴模式（仅 Clip_Frame 生效） |
| prompt_format | official | 提示词输出格式：official / legacy / raw（raw 为 Clip_Tag 专用，原样输出） |
| crop_mode | stretch | 参考图/首尾帧/参考视频的缩放裁剪：center（等比缩放到最短边并中心裁剪）/ stretch（直接拉伸）/ none（保持原始分辨率，仅 32 对齐，高级用户） |
| ref_sync_mode | global | 参考视频/音频是否按段切片：global（每段用完整参考）/ segmented（每段取对应时间片段，用于口型同步） |
| decode_output | disable | 是否内部解码：disable（默认，只输出 latent）/ enable（输出 image+audio+latent） |
| width × height | 960×544 | 生成分辨率 |
| total_frames | 362 | 生成总帧数（需满足 17n+5，约 15s@24fps），Clip_Tag 模式下被各段之和覆盖 |
| fps | 24 | 帧率，仅用于音频同步和提示词内秒数换算 |
| chunk_frames | 90 | 每段生成帧数（需满足 17n+5，约 3.75s@24fps），Clip_Tag 模式下作为时长兜底默认值 |
| context_frames | 22 | 段间续接帧数，值越大连续性越强 |
| steps / cfg / sampler / scheduler | 30 / 1.0 / euler / simple | 采样参数 |
| seed | 0 | 随机种子 |
| ref_image_N | 可选 | 参考图片（提示词中用 `image1`/`image 1`/`图像1`/`图片1` 或 `<Picture N>` 引用，1 基） |
| ref_video_N | 可选 | 参考视频帧 |
| ref_video_audio_N | 可选 | 同编号参考视频的配对音轨 |
| ref_audio_N | 可选 | 独立参考音频 |
| drive_audio | 可选 | 音频驱动源（要锁定的源音频），配合 `audio_drive=enable` 使用 |
| audio_drive | disable | 音频驱动开关：enable 时锁定 `drive_audio`（noise_mask=0），输出音频=源音频本身 |

## 提示词写法示例

### 时间轴模式（auto / timeline）

```
0-5s: ...
5-10s: ...

integrated_multimodal_description
....

overall_soundscape

```

含 `0-5s` 标记的段落按时间切分，无标记段落（风格/音效/禁止项）自动拼入每个窗口。

### 全局模式（global）

整段提示词用于所有分段，适合全程同质动作的一镜到底。

### Clip_Tag 模式（按标签分段）

`clip_mode` 设为 `Clip_Tag`，`clip_tag` 填写标签模板（必须以数字序号结尾）。

**标签模板示例**：
- `段1` → 匹配提示词中的 `段1`/`段2`/`段3`（前缀"段"+数字）
- `A01` → 匹配 `A01`/`A02`/`A03`（前缀"A"+数字）
- `[片段001]` → 匹配 `[片段001]`/`[片段002]`（前缀"[片段"+数字+后缀"]"）

**标签写法**：标签独占一行作为分割点，标签后推荐换行。不换行时也能处理（跳过分隔符取段内容）：
```
段1:0-3s
视频：
0-3秒
...
音频设计：
0-3秒：...。


段2:3-8s
视频：
0-2秒
...
2-5秒
...
音频设计：
0-5秒：...

```

**段时长规则**（三层优先级）：
1. 标签行紧跟的时长：`段1:0-5秒` → 5 秒；`段1:3-8秒` → 5 秒（时长标记会从提示词中去除）
2. 段内时间标记 0 基：`【0-2秒】`+`【2-5秒】` → 5 秒
3. 都没有 → `chunk_frames / fps` 兜底

**prompt_format 选择**：
- `official`/`legacy`：段内时间标记自动转为段内相对坐标渲染
- `raw`：去标签后原样输出，时间标记保持不变（适合用大模型生成的结构化提示词）

**续接补偿**：非首段多生成 `context_frames` 帧用于头部锚定（重演上一段尾部），生成后自动裁掉，保证段间连续无卡顿。实际输出时长 = 各段有效新增之和，运行日志会打印实际值。

## 输出

- **video_frames**：拼接后的完整视频帧序列（内部解码，段间裁剪对齐）
- **audio**：拼接后的音频（段间 cosine 交叉淡化）
- **latent**：音视频共用 latent（NestedTensor），裁掉段间重演区后拼接，可用 ComfyUI 原生 VAE Decode 解码。推荐用法：latent → VAE Decode 得到视频，配合 audio 端口输出音频（波形层面交叉淡化，连续性更好）


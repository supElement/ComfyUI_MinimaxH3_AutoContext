"""
nodes.py - H3 Auto Context Sampler 节点定义

使用 io.ComfyNode 新 API，支持 io.Autogrow 动态端口：
- ref_image_0 ~ ref_image_9 (参考图片)
- ref_video_0 ~ ref_video_3 (参考视频)
- ref_video_audio_0 ~ ref_video_audio_3 (参考视频配对音轨)
- ref_audio_0 ~ ref_audio_3 (独立参考音频)

默认只显示 _0 端口，连接后才显示 _1，以此类推。
"""

from comfy_api.latest import io
from . import h3_sampler

_SAMPLERS = [
    "euler", "euler_ancestral", "euler_cfg_pp",
    "res_multistep", "res_multistep_cfg_pp", "dpmpp_2m", "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde", "dpmpp_3m_sde", "uni_pc", "uni_pc_bh2",
    "ddpm", "lms", "heun", "dpm_2", "dpm_2_ancestral",
]
_SCHEDULERS = ["simple", "normal", "karras", "exponential",
               "sgm_uniform", "beta", "linear_quadratic"]

_PROMPT_MODE_TOOLTIP = (
    "提示词时间轴模式 (仅 Clip_Frame 模式生效，Clip_Tag 模式下自动忽略)。"
    "auto: 段落含时间标记(如0-5s)自动切分，无标记的段落(风格/音效/禁止项)自动拼入每个窗口; "
    "timeline: 强制按时间标记切分; global: 整段提示词用于所有窗口(适合全程同质动作); "
    "sequential: 按句读顺序铺到时间轴，以 '全局:'/'[全局]'/'[global]' 开头的段落不平铺、拼入每个窗口"
)
_CONTEXT_FRAMES_TOOLTIP = (
    "段间续接用的上一段尾部帧数。值越大连续性越强。建议 22 以上防硬切。有效网格点: 5,22,39,56,73,90,107,124"
)
_CLIP_MODE_TOOLTIP = (
    "分段模式。Clip_Frame: 按 chunk_frames 均匀切分视频时间轴，提示词按时间标记匹配各窗口 (原有逻辑); "
    "Clip_Tag: 按用户自定义标签切分提示词，每段=一个 chunk=该段全部提示词，段时长由提示词内容决定"
)
_CLIP_TAG_TOOLTIP = (
    "Clip_Tag 模式的分割标签模板 (仅 Clip_Tag 模式生效)。必须以数字序号结尾，"
    "如 '段1' (前缀'段'+1位数字)、'A01' (前缀'A'+2位数字)、'[片段001]' (前缀'[片段'+3位数字+后缀']')。"
    "提示词中标签独占一行才作为分割点，标签后推荐换行，不换行时跳过分隔符 (:：，,。；; —–- 空格) 取段内容。"
    "推理时标签本身会被去除。提示词内时间写法保持秒不变"
)
_PROMPT_FORMAT_TOOLTIP = (
    "提示词输出格式。official: [Shot N] At MM:SS.mmm + <Picture N>/<Video N>/<Audio N> 标签 + 官方字段名 (MiniMax H3 官方训练格式，推荐); "
    "legacy: 【0-3秒】+ 引用保持原写法(1基) + 块标题 (旧格式); "
    "raw: 完全不处理引用，去标签后原样输出，不做任何时间标记转换 (Clip_Tag 模式专用，保留段内相对时间原样)"
)
_CROP_MODE_TOOLTIP = (
    "参考图/首尾帧/参考视频的缩放裁剪模式。center: 等比例缩放并中心裁剪到目标尺寸; "
    "stretch: 直接拉伸到目标尺寸; "
    "none: 保持原始分辨率，仅做 32 对齐 (高级用户选项，ref2va 场景使用，尺寸问题用户自行处理)"
)
_REF_SYNC_MODE_TOOLTIP = (
    "参考视频/音频是否按生成段的时间范围切片。global: 每段使用完整参考视频/音频 (默认); "
    "segmented: 每段只取参考视频/音频中对应时间片段，用于替换人物并保持口型同步等场景"
)


class H3AutoContextSampler(io.ComfyNode):
    """一键式 MiniMax H3 长视频自动化生成节点 (分段推理 + 续接锚定)"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3AutoContextSampler",
            display_name="Minimax_H3_AutoContext_Sampler",
            category="MinimaxH3_AutoContext",
            description="MiniMax H3 长视频分段推理：自动分段、段间续接锚定、提示词时间轴切片",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Clip.Input("clip"),

                io.Image.Input("first_frame", optional=True,
                    tooltip="首帧锚定 (FL2VA 模式)"),
                io.Image.Input("last_frame", optional=True,
                    tooltip="尾帧锚定 (FL2VA 模式)"),

                io.String.Input("long_prompt", multiline=True, dynamic_prompts=True,
                    socketless=False, default="一镜到底，平滑移动"),
                io.Combo.Input("clip_mode",
                    options=["Clip_Frame", "Clip_Tag"],
                    default="Clip_Frame", tooltip=_CLIP_MODE_TOOLTIP),
                io.String.Input("clip_tag", multiline=False, dynamic_prompts=True,
                    socketless=False, default="段1",
                    tooltip=_CLIP_TAG_TOOLTIP),
                io.Combo.Input("prompt_mode",
                    options=["auto", "timeline", "global", "sequential"],
                    default="auto", tooltip=_PROMPT_MODE_TOOLTIP),
                io.Combo.Input("prompt_format",
                    options=["official", "legacy", "raw"],
                    default="official",
                    tooltip=_PROMPT_FORMAT_TOOLTIP),
                io.Combo.Input("crop_mode",
                    options=["center", "stretch", "none"],
                    default="stretch",
                    tooltip=_CROP_MODE_TOOLTIP),
                io.Combo.Input("ref_sync_mode",
                    options=["global", "segmented"],
                    default="global",
                    tooltip=_REF_SYNC_MODE_TOOLTIP),
                io.Combo.Input("decode_output",
                    options=["disable", "enable"],
                    default="disable",
                    tooltip="是否在节点内部解码输出 video_frames 和 audio。disable (默认): 只输出 latent，用 ComfyUI 原生 VAE Decode 解码（画面连续性更好）; enable: 节点内部解码，输出 image+audio+latent"),
                io.Int.Input("width", default=960, min=64, max=4096, step=32),
                io.Int.Input("height", default=544, min=64, max=4096, step=32),
                io.Int.Input("total_frames", default=362, min=5, max=2880, step=17,
                    tooltip="生成总帧数，需满足 17n+5 (5,22,39,56,73,90,...)。提示词内时间仍按秒解析"),
                io.Int.Input("fps", default=24, min=8, max=60, step=1,
                    tooltip="帧率，仅用于音频同步和提示词内秒数换算"),
                io.Int.Input("chunk_frames", default=90, min=5, max=2880, step=17,
                    tooltip="每段生成帧数，需满足 17n+5 (5,22,39,...)。设为 ≥ total_frames 时不拆分，整个视频作为一段生成"),
                io.Int.Input("context_frames", default=22, min=5, max=124, step=1,
                    tooltip=_CONTEXT_FRAMES_TOOLTIP),
                io.Int.Input("steps", default=30, min=1, max=100, step=1),
                io.Float.Input("cfg", default=1.0, min=0.0, max=30.0, step=0.1),
                io.Combo.Input("sampler_name", options=_SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=_SCHEDULERS, default="simple"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                    control_after_generate=True),

                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image",
                            tooltip="参考图片 (提示词中用 image1/image 1/图像1/图片1 或 <Picture N> 引用, 1 基)"),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video",
                            tooltip="参考视频帧 (24fps, 2-15s)"),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio",
                            tooltip="同编号参考视频的配对音轨"),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio",
                            tooltip="独立参考音频"),
                        prefix="ref_audio_", min=0, max=3)),
                io.Audio.Input("drive_audio", optional=True,
                    tooltip="音频驱动源 (Audio Drive)。接要锁定的源音频，开启 audio_drive 后 "
                            "输出音频=这条音频本身 (口型/节奏由它驱动)。可与 ref_audio_0 接同一条音频"),
                io.Combo.Input("audio_drive",
                    options=["disable", "enable"],
                    default="disable",
                    tooltip="音频驱动开关。enable: 把 drive_audio 编码后锁进 latent (noise_mask=0，"
                            "不重新生成音频)，视频照它生成；输出音频=drive_audio 本身。"
                            "disable (默认): 与之前行为一致"),
            ],
            outputs=[
                io.Image.Output(display_name="video_frames"),
                io.Audio.Output(display_name="audio"),
                io.Latent.Output(display_name="latent"),
            ],
        )

    @classmethod
    def execute(cls, model, vae, audio_vae, clip,
                first_frame=None, last_frame=None,
                long_prompt="一镜到底，平滑移动",
                clip_mode="Clip_Frame", clip_tag="段1",
                prompt_mode="auto",
                prompt_format="official",
                crop_mode="stretch",
                ref_sync_mode="global",
                decode_output="disable",
                width=960, height=544,
                total_frames=362, fps=24, chunk_frames=90,
                context_frames=22, steps=30, cfg=1.0,
                sampler_name="euler", scheduler="simple", seed=0,
                ref_images=None, ref_videos=None,
                ref_video_audios=None, ref_audios=None,
                drive_audio=None, audio_drive="disable") -> io.NodeOutput:

        if clip_mode != "Clip_Tag":
            if chunk_frames <= 0:
                chunk_frames = total_frames
        width = max(32, (width // 32) * 32)
        height = max(32, (height // 32) * 32)

        def _autogrow_to_list(ag_dict, prefix, max_count):
            """将 autogrow dict 转为有序 list，按 ref_xxx_N 的 N 排序"""
            if not ag_dict:
                return []
            result = []
            for i in range(max_count):
                key = f"{prefix}{i}"
                val = ag_dict.get(key)
                if val is not None:
                    result.append(val)
            return result

        ref_image_list = _autogrow_to_list(ref_images, "ref_image_", 10)

        ref_video_audios = ref_video_audios or {}
        ref_video_list = []
        for i in range(4):
            vkey = f"ref_video_{i}"
            vval = (ref_videos or {}).get(vkey)
            if vval is None:
                continue
            akey = f"ref_video_audio_{i}"
            soundtrack = ref_video_audios.get(akey)
            ref_video_list.append({"video": vval, "audio": soundtrack})

        ref_audio_list = _autogrow_to_list(ref_audios, "ref_audio_", 4)

        frames, audio, latent = h3_sampler.run_auto_context_generation(
            model=model, vae=vae, audio_vae=audio_vae, clip=clip,
            first_frame=first_frame, last_frame=last_frame,
            ref_images=ref_image_list, ref_videos=ref_video_list,
            ref_audios=ref_audio_list,
            long_prompt=long_prompt, width=width, height=height,
            total_frames=total_frames, fps=fps,
            chunk_frames=chunk_frames, context_frames=int(context_frames),
            steps=steps, cfg=cfg, sampler_name=sampler_name,
            scheduler=scheduler, seed=seed, prompt_mode=prompt_mode,
            prompt_format=prompt_format,
            clip_mode=clip_mode, clip_tag=clip_tag,
            crop_mode=crop_mode, ref_sync_mode=ref_sync_mode,
            decode_output=(decode_output == "enable"),
            drive_audio=drive_audio, audio_drive=(audio_drive == "enable"),
        )
        return io.NodeOutput(frames, audio, latent)

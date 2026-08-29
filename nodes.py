"""
nodes.py - H3 Auto Context Sampler 节点定义

使用 io.ComfyNode 新 API，支持 io.Autogrow 动态端口：
- ref_image_0 ~ ref_image_9 (参考图片)
- ref_video_0 ~ ref_video_3 (参考视频)
- ref_video_audio_0 ~ ref_video_audio_3 (参考视频配对音轨)
- ref_audio_0 ~ ref_audio_3 (独立参考音频)

默认只显示 _0 端口，连接后才显示 _1，以此类推。

参数组 (分段/分辨率/音频等) 已拆分到 Minimax_H3_AutoContext_parameter 节点，
通过 parameter (Dict) 单端口传入本节点；主节点只保留采样/提示词/参考素材相关输入。
"""

import comfy.utils
import os
import folder_paths
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

_CONTEXT_FRAMES_TOOLTIP = (
    "段间续接用的上一段尾部帧数。值越大连续性越强。建议 22 以上防硬切。有效网格点: 5,22,39,56,73,90,107,124"
)
_CLIP_MODE_TOOLTIP = (
    "提示词映射到视频段的方式：Clip_Tag按自定义标签分段；timeline按时间标记分段；sequential按句读顺序平铺；global整段用于所有段。"
    "Clip_Tag和timeline模式下，total_frames和chunk_frames无效，段长由提示词决定。"
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


class H3ParameterNode(io.ComfyNode):
    """H3 参数组节点：集中管理分段/分辨率/音频等参数，输出 parameter dict，并预览预计分段。"""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3Parameter",
            display_name="Minimax_H3_AutoContext_parameter",
            category="MinimaxH3_AutoContext",
            description="H3 参数组：集中管理分段/分辨率/音频等参数，输出 parameter 供主节点使用，并预览预计分段",
            inputs=[
                io.String.Input("long_prompt", multiline=True, dynamic_prompts=True,
                    socketless=False, default="",
                    tooltip="提示词 (传给主节点推理，同时用于「预计分段」预览)"),
                io.Combo.Input("clip_mode",
                    options=["Clip_Tag", "timeline", "sequential", "global"],
                    default="Clip_Tag",
                    tooltip=_CLIP_MODE_TOOLTIP),
                io.String.Input("clip_tag", multiline=False, dynamic_prompts=True,
                    socketless=False, default="段1",
                    tooltip=_CLIP_TAG_TOOLTIP),
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
                    default="segmented",
                    tooltip=_REF_SYNC_MODE_TOOLTIP),
                io.Int.Input("width", default=960, min=64, max=4096, step=32),
                io.Int.Input("height", default=544, min=64, max=4096, step=32),
                io.Int.Input("total_frames", default=362, min=5, max=2880, step=17,
                    tooltip="生成总帧数，需满足 17n+5 (5,22,39,56,73,90,...)。提示词内时间仍按秒解析。"
                            "在 Clip_Tag 和 timeline 模式下，该值被忽略，由提示词内容自动计算。"),
                io.Int.Input("fps", default=24, min=8, max=60, step=1,
                    tooltip="帧率，仅用于音频同步和提示词内秒数换算"),
                io.Int.Input("chunk_frames", default=90, min=5, max=2880, step=17,
                    tooltip="每段生成帧数，需满足 17n+5 (5,22,39,...)。设为 ≥ total_frames 时不拆分，整个视频作为一段生成。"
                            "在 Clip_Tag 和 timeline 模式下，该值被忽略，由提示词内容自动计算。"),
                io.Int.Input("context_frames", default=22, min=5, max=124, step=1,
                    tooltip=_CONTEXT_FRAMES_TOOLTIP),
                io.Boolean.Input("lock_audio",
                    default=True,
                    tooltip="二采时锁定音频区 (noise_mask audio=0)：只重新采样视频、保持一采音频不变。仅在连接 latent_input 时生效"),
                io.Boolean.Input("audio_drive",
                    default=False,
                    tooltip="音频驱动开关。勾选后把 drive_audio 编码后锁进 latent (noise_mask=0，"
                            "不重新生成音频)，视频照它生成。"
                            "注意：本节点不输出音频，请把同一条源音频直接接到视频合成节点 "
                            "(这样也避免了 VAE 有损往返)。不勾选 (默认): 音频照常生成"),
                
                io.Combo.Input("video_guide",
                    options=["none", "pre_guide", "post_guide", "pre_post_guide"],
                    default="none",
                    tooltip="利用 ref_video 端口输入的视频进行多帧强锚定。\n"
                            "none: 常规生成（默认），即使 ref_video 有输入也只作普通参考；\n"
                            "pre_guide: 用 ref_video_0 的尾部锚定开头（视频续写）；\n"
                            "post_guide: 用 ref_video_0 的头部锚定结尾（视频前推）；\n"
                            "pre_post_guide: 用 ref_video_0 尾部锚定开头 + ref_video_1 头部锚定结尾（中间衔接）。\n"
                            "锚定帧数由 context_frames 决定。"),
            ],
            outputs=[
                io.Dict.Output(display_name="parameter"),
            ],
        )

    @classmethod
    def execute(cls, long_prompt="", clip_mode="Clip_Tag", clip_tag="段1",
                prompt_format="official", crop_mode="stretch", ref_sync_mode="segmented",
                width=960, height=544, total_frames=362, fps=24, chunk_frames=90,
                context_frames=22, lock_audio=True, audio_drive=False,
                video_guide="none",
                ) -> io.NodeOutput:
        parameter = {
            "long_prompt": long_prompt,
            "clip_mode": clip_mode,
            "clip_tag": clip_tag,
            "prompt_format": prompt_format,
            "crop_mode": crop_mode,
            "ref_sync_mode": ref_sync_mode,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "fps": fps,
            "chunk_frames": chunk_frames,
            "context_frames": context_frames,
            "lock_audio": lock_audio,
            "audio_drive": audio_drive,
            "video_guide": video_guide,
        }
        return io.NodeOutput(parameter)


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

                io.Dict.Input("parameter",
                    tooltip="参数组输入 (必选，来自 Minimax_H3_AutoContext_parameter 节点)。提示词与分段/分辨率/音频参数均由此传入；info 中的分段参数优先于本参数组"),
                io.Sampler.Input("sampler", optional=True,
                    tooltip="外部采样器对象 (SAMPLER, 可选)。接入后覆盖内置 sampler_name/scheduler，与 SamplerCustom 同款接法"),
                io.Sigmas.Input("sigmas", optional=True,
                    tooltip="自定义 sigma 序列 (SIGMAS, 优先级最高, 与 SamplerCustomAdvanced 接法一致)。接入后接管采样 sigma；denoise≠1 时 final_sigmas=sigmas*denoise"),
                io.Latent.Input("latent_input", optional=True,
                    tooltip="二采输入 latent (可选)。接上一节点或 latent 放大节点输出的 latent 开启二次采样；空间分辨率以该 latent 为准，忽略 width/height"),
                io.Dict.Input("info", optional=True,
                    tooltip="参数继承输入 (来自上一个同款节点的 info 输出)。info 中存在的分段参数覆盖 parameter/本节点同名值，保证多节点分段一致 (二采/多采串联)"),
                io.Image.Input("first_frame", optional=True,
                    tooltip="首帧锚定 (FL2VA 模式)"),
                io.Image.Input("last_frame", optional=True,
                    tooltip="尾帧锚定 (FL2VA 模式)"),

                io.Float.Input("video_context_denoise",
                    default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="段间续接 overlap 头去噪强度 (仅非首段生效)。0=精确冻结上一段尾部 (消除接缝停顿/错位)，"
                            "1=完全重绘 (旧行为)，中间值=软混合。二采接 SplitSigmas 时建议设 1 避免花屏"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                    control_after_generate=True),
                io.Int.Input("steps", default=30, min=1, max=100, step=1),
                io.Float.Input("cfg", default=1.0, min=0.0, max=30.0, step=0.1),
                io.Combo.Input("sampler_name", options=_SAMPLERS, default="euler"),
                io.Combo.Input("scheduler", options=_SCHEDULERS, default="simple"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="重绘强度。1.0=全量重采样，越小保留越多原结构。连接 sigmas 且 denoise≠1 时 final_sigmas=sigmas*denoise"),
                
                io.Boolean.Input("enable_cache",
                    default=True,
                    tooltip="是否启用分段缓存。开启后，采样结果会保存到 cache_dir，后续运行时若参数一致则直接加载。\n"
                            "更换模型/LoRA/加速节点后请手动删除缓存文件。"
                ),
                io.Boolean.Input("clear_cache",
                    default=False,
                    tooltip="删除该节点的所有缓存文件。用于更换模型/LoRA/加速器后强制重新生成。"
                ),

                io.Boolean.Input("ignore_latent_hash",
                    default=False,
                    tooltip="⚠️ 高级选项：禁用二采输入 Latent 指纹校验。\n"
                            "仅当latent放大节点导致哈希不稳定时启用。\n"
                ),

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
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Latent.Output(display_name="denoised_latent"),
                io.Dict.Output(display_name="info"),
            ], 
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, vae, audio_vae, clip,
            parameter=None,
            sigmas=None, latent_input=None, info=None,
            first_frame=None, last_frame=None,
            denoise=1.0,
            steps=30, cfg=1.0,
            sampler_name="euler", scheduler="simple",
            sampler=None,
            video_context_denoise=0.0,
            seed=0,
            enable_cache=True,
            clear_cache=False,
            ignore_latent_hash=False,
            ref_images=None, ref_videos=None,
            ref_video_audios=None, ref_audios=None,
            drive_audio=None) -> io.NodeOutput:
                
        
        try:
            unique_id = cls.hidden.unique_id
        except AttributeError:
            unique_id = None
            
        p = parameter or {}
        video_guide = p.get("video_guide", "none")
        long_prompt = p.get("long_prompt", "")
        clip_mode = p.get("clip_mode", "Clip_Tag")
        clip_tag = p.get("clip_tag", "段1")
        prompt_format = p.get("prompt_format", "official")
        crop_mode = p.get("crop_mode", "stretch")
        ref_sync_mode = p.get("ref_sync_mode", "segmented")
        width = int(p.get("width", 960))
        height = int(p.get("height", 544))
        total_frames = int(p.get("total_frames", 362))
        fps = int(p.get("fps", 24))
        chunk_frames = int(p.get("chunk_frames", 90))
        context_frames = int(p.get("context_frames", 22))
        lock_audio = p.get("lock_audio", True)
        audio_drive = p.get("audio_drive", False)
        decode_output = False

        # info 参数继承：info 中存在的分段参数覆盖 parameter 解包值
        if info:
            _inherited = {}
            for _k in ("total_frames", "chunk_frames", "context_frames",
                       "fps", "clip_mode", "clip_tag"):
                if _k in info and info[_k] is not None:
                    _inherited[_k] = info[_k]
            total_frames = int(_inherited.get("total_frames", total_frames))
            chunk_frames = int(_inherited.get("chunk_frames", chunk_frames))
            context_frames = int(_inherited.get("context_frames", context_frames))
            fps = int(_inherited.get("fps", fps))
            clip_mode = _inherited.get("clip_mode", clip_mode)
            clip_tag = _inherited.get("clip_tag", clip_tag)
            if _inherited:
                print(f"[H3-Auto] info 参数继承: {list(_inherited.keys())}")

        out_info = {
            "total_frames": int(total_frames),
            "chunk_frames": int(chunk_frames),
            "context_frames": int(context_frames),
            "fps": int(fps),
            "clip_mode": str(clip_mode),
            "clip_tag": str(clip_tag),
        }

        # 参数归一化
        width = max(32, (width // 32) * 32)
        height = max(32, (height // 32) * 32)
        if isinstance(audio_drive, str):
            audio_drive = (audio_drive == "enable")
        if isinstance(lock_audio, str):
            lock_audio = (lock_audio == "enable")

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

        # ===== 强制清除缓存 =====
        if clear_cache and unique_id is not None:
            try:
                output_dir = folder_paths.get_output_directory()
                target_cache_dir = os.path.join(output_dir, "cache", f"node_{unique_id}")
                if os.path.exists(target_cache_dir):
                    import shutil
                    shutil.rmtree(target_cache_dir)
                    print(f"[H3-Auto] 🗑️  缓存已清除: {target_cache_dir}")
                else:
                    print("[H3-Auto] ℹ️  缓存目录不存在，无需清除")
            except Exception as e:
                print(f"[H3-Auto] ⚠️  清除缓存失败: {e}")

        # 自动缓存目录逻辑
        cache_dir = ""
        
        if enable_cache and unique_id is not None:
            try:
                output_dir = folder_paths.get_output_directory()
                cache_dir = os.path.join(output_dir, "cache", f"node_{unique_id}")
                out_info["cache_dir"] = cache_dir
                print(f"[H3-Auto] 缓存目录: {cache_dir}")
            except Exception as e:
                print(f"[H3-Auto] 缓存目录生成失败: {e}，已禁用缓存")
                enable_cache = False
        elif enable_cache:
            print("[H3-Auto] 警告: 无法获取节点ID，缓存功能不可用，请升级 ComfyUI")
            enable_cache = False

        _frames, _audio, latent, denoised_latent, seam_info = h3_sampler.run_auto_context_generation(
            model=model, vae=vae, audio_vae=audio_vae, clip=clip,
            first_frame=first_frame, last_frame=last_frame,
            ref_images=ref_image_list, ref_videos=ref_video_list,
            ref_audios=ref_audio_list,
            long_prompt=long_prompt, width=width, height=height,
            total_frames=total_frames, fps=fps,
            chunk_frames=chunk_frames, context_frames=int(context_frames),
            steps=steps, cfg=cfg, sampler_name=sampler_name,
            scheduler=scheduler, seed=seed,
            clip_mode=clip_mode,
            prompt_format=prompt_format,
            clip_tag=clip_tag,
            crop_mode=crop_mode, ref_sync_mode=ref_sync_mode,
            decode_output=decode_output,
            drive_audio=drive_audio, audio_drive=audio_drive,
            latent_input=latent_input, sigmas=sigmas,
            denoise=denoise, lock_audio=lock_audio,
            video_context_denoise=video_context_denoise,
            sampler=sampler,
            enable_cache=enable_cache,
            cache_dir=cache_dir,
            ignore_latent_hash=ignore_latent_hash,
            info=info,
            video_guide=video_guide,
        )
        if seam_info:
            out_info.update(seam_info)

        return io.NodeOutput(latent, denoised_latent, out_info)
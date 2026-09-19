"""h3_face_resample.py — 修脸第 2 步: 画布重采样 (v5)

v5: sigma/steps 两个 widget 合并为 sigmas 输入端口 (SIGMAS, 与
SamplerCustomAdvanced 的 sigmas 同类型同源——BasicScheduler / SplitSigmasDenoise
等调度器节点直接可接)。精修 σ 阶梯完全由外部调度器决定, 节点内不再构造。
其余与 v4 一致: 裁剪序列→lanczos放大→编码→H3重采样→解码, 输出原生 res² 画布
(一切缩放由 ③ 完成), bbox 原样透传。
"""

import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import comfy.nested_tensor
from comfy_api.latest import io

try:
    from . import h3_conditioning
    from . import h3_facefix as h3ff
    from . import h3_sampler
except ImportError:
    import h3_conditioning
    import h3_facefix as h3ff
    import h3_sampler

CFG = 1.0


def _pick_prompt(fc, boundaries, seg_prompts, long_prompt):
    if not seg_prompts:
        return long_prompt, "global prompt / 全局提示词"
    idx = min(sum(1 for b in (boundaries or []) if b <= int(fc) // 2),
              len(seg_prompts) - 1)
    return seg_prompts[idx], f"segment {idx + 1} prompt / 段{idx + 1}提示词"


def _normalize_sigmas(sigmas, device):
    """SIGMAS 端口 → 精修阶梯: 校验递减/首值>0, 末值非0自动补0 (完整去噪)。"""
    if sigmas is None:
        raise ValueError("[H3-FaceResample] sigmas port not connected — connect a scheduler output such as "
                         "BasicScheduler / SplitSigmasDenoise(low_sigmas)"
                         "\n[H3-FaceResample] sigmas 端口未连接 — 请接 "
                         "BasicScheduler / SplitSigmasDenoise(low_sigmas) 等调度器输出")
    s = sigmas.detach().clone().flatten().to(device=device, dtype=torch.float32)
    if s.numel() < 2:
        raise ValueError(f"[H3-FaceResample] sigmas needs at least 2 values (first σ>0, last σ=0), got {s.numel()}"
                         f"\n[H3-FaceResample] sigmas 至少 2 个值 (首σ>0, 末σ=0), 收到 {s.numel()} 个")
    if not bool(torch.all(s[1:] <= s[:-1])):
        raise ValueError("[H3-FaceResample] sigmas must be monotonically non-increasing: "
                         f"{[round(float(x), 4) for x in s]}"
                         "\n[H3-FaceResample] sigmas 必须单调不增: "
                         f"{[round(float(x), 4) for x in s]}")
    if float(s[0]) <= 0.0:
        raise ValueError("[H3-FaceResample] first σ must be > 0"
                         "\n[H3-FaceResample] 首 σ 必须 > 0")
    if float(s[-1]) != 0.0:
        s = torch.cat([s, torch.zeros(1, device=device)])
        print(f"[H3-FaceResample] last sigma is not 0, appended 0 (full denoise): "
              f"{[round(float(x), 4) for x in s]}\n"
              f"[H3-FaceResample] sigmas 末值非 0, 自动补 0 (完整去噪): "
              f"{[round(float(x), 4) for x in s]}")
    return s


class H3FaceResample(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3FaceResample",
            display_name="Minimax_H3_Face_Resample",
            category="MinimaxH3_AutoContext/FaceFix",
            description="Face fix step 2: upscale and encode the crop sequence -> H3 resample -> native canvas output. "
                        "The σ schedule comes from the sigmas input.\n"
                        "修脸第 2 步: 裁剪序列放大编码→H3重采样→原生画布输出。σ阶梯由 sigmas 端口决定。",
            inputs=[
                io.Image.Input("crop_images", tooltip="crop_images from H3FaceCut\n来自 H3FaceCut 的 crop_images"),
                io.Dict.Input("face_pack", tooltip="face_pack from H3FaceCut\n来自 H3FaceCut 的 face_pack"),
                io.Dict.Input("info", tooltip="info output of the sampler node (must contain h3_runtime)\n"
                                              "采样节点的 info 输出 (需含 h3_runtime)"),
                io.Sigmas.Input("sigmas", tooltip="Refinement σ schedule (monotonically decreasing, last value 0). "
                                "SplitSigmasDenoise low_sigmas = refine from that σ down to 0\n"
                                "精修 σ 阶梯 (单调递减, 末值0)。"
                                "SplitSigmasDenoise 的 low_sigmas = 从该σ精修到 0"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff,
                             control_after_generate=True),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Dict.Output(display_name="bbox"),
            ],
        )

    @classmethod
    def execute(cls, crop_images, face_pack, info, sigmas, seed=0) -> io.NodeOutput:
        pack = face_pack or {}
        if int(pack.get("version") or 0) != 3:
            raise ValueError("[H3-FaceResample] pack version mismatch — rerun H3FaceCut (v4)"
                             "\n[H3-FaceResample] pack 版本不符 — 请重跑 H3FaceCut (v4)")
        S = int(pack.get("crop_size") or 0)
        a_lat = pack.get("a_lat")
        if S <= 0 or a_lat is None or crop_images is None or crop_images.dim() != 4:
            print("[H3-FaceResample] no face empty pack, crop_images passed through unchanged\n"
                  "[H3-FaceResample] 无脸空包, crop_images 原样透传")
            return io.NodeOutput(crop_images, pack)

        rt = info.get("h3_runtime") if isinstance(info, dict) else None
        if not rt:
            raise ValueError("[H3-FaceResample] info does not contain h3_runtime"
                             "\n[H3-FaceResample] info 中没有 h3_runtime")
        model = rt["model"]; clip = rt["clip"]; vae = rt["vae"]
        audio_vae = rt.get("audio_vae")
        device = comfy.model_management.get_torch_device()

        base_px = crop_images.detach().float()          # [F,S,S,C]
        F_ = int(base_px.shape[0])
        if (F_ - 5) % 17 != 0:
            raise ValueError(f"[H3-FaceResample] frame count {F_} is not on the 17k+5 grid — pack and latent do not match"
                             f"\n[H3-FaceResample] 帧数 {F_} 不在 17k+5 网格上 — pack 与 latent 不匹配")
        T_exp = 5 * ((F_ - 5) // 17) + 2

        res = int((pack.get("meta") or {}).get("res", 512))
        boundaries = info.get("boundaries") or []
        seg_prompts = info.get("segment_prompts") or []
        long_prompt = rt.get("long_prompt") or ""
        crop_mode = rt.get("crop_mode", "stretch")
        ref_fps = int(rt.get("fps", 24))

        # ---- σ 阶梯: 完全来自 sigmas 端口 ----
        fix_sig = _normalize_sigmas(sigmas, device)
        k = int(fix_sig.numel()) - 1
        print(f"[H3-FaceResample] fix_sig={[round(float(s), 3) for s in fix_sig]} ({k} steps), "
              f"canvas {res}x{res} (S={S}, {float(res) / S:.2f}x), T={T_exp}\n"
              f"[H3-FaceResample] fix_sig={[round(float(s), 3) for s in fix_sig]} ({k} 步), "
              f"画布 {res}x{res} (S={S}, {float(res) / S:.2f}x), T={T_exp}")

        # ---- 裁剪序列 → 画布 latent (1E) ----
        base_lat = h3ff.upscale_and_encode(vae, base_px, res, res,
                                           want_t=T_exp, tag="[crop canvas / 裁剪画布]")
        if base_lat is None:
            raise RuntimeError("[H3-FaceResample] crop canvas encoding failed (see the [H3-Enc] log)"
                               "\n[H3-FaceResample] 裁剪画布编码失败 (见 [H3-Enc] 日志)")
        base_lat = base_lat.to(device=device, dtype=torch.float32)

        # ---- 参考素材按画布分辨率重编码 ----
        ri = (h3_sampler._prepare_ref_images(rt.get("ref_images"), vae, device, res, res, crop_mode)
              if rt.get("ref_images") else [])
        rv = (h3_sampler._prepare_ref_videos(rt.get("ref_videos"), vae, audio_vae,
                                             device, res, res, ref_fps, crop_mode)
              if rt.get("ref_videos") else [])
        ra = (h3_sampler._prepare_ref_audios(rt.get("ref_audios"), audio_vae, device)
              if rt.get("ref_audios") else [])
        print(f"[H3-FaceResample] reference assets re-encoded to canvas {res}x{res} "
              f"(images {len(ri)}/videos {len(rv)}/audios {len(ra)})\n"
              f"[H3-FaceResample] 参考素材按画布 {res}x{res} 重编码 "
              f"(图{len(ri)}/视频{len(rv)}/音频{len(ra)})")

        prompt, src_name = _pick_prompt(F_, boundaries, seg_prompts, long_prompt)
        payload = h3_conditioning.build_conditioning_payload(
            seed=int(rt.get("seed", 0)) + 1, frame_count=int(F_),
            ref_img_data=ri, ref_vid_data=rv, ref_aud_latents=[a["latent"] for a in ra],
            fps=ref_fps)
        positive = h3_conditioning.encode_text_with_references(
            clip, prompt, payload["ref_items_for_clip"], device,
            images_for_clip=payload.get("images_for_clip"))
        positive = h3_conditioning.inject_conditioning_data(positive, payload)
        print(f"[H3-FaceResample] prompt source: {src_name}\n"
              f"[H3-FaceResample] prompt 来源: {src_name}")

        # ---- 重采样: mask 全放开 ----
        audio = a_lat.to(device=device, dtype=torch.float32)
        try:
            latent_fix = comfy.nested_tensor.NestedTensor((base_lat, audio))
        except Exception:
            latent_fix = (base_lat, audio)
        s = int(seed) + 1
        noise = comfy.sample.prepare_noise(latent_fix, s)
        try:
            if rt.get("sampler_obj") is not None:
                out_s = comfy.sample.sample_custom(
                    model, noise, CFG, rt["sampler_obj"], fix_sig, positive, [],
                    latent_fix, noise_mask=None, disable_pbar=False, seed=s)
            else:
                ks = comfy.samplers.KSampler(
                    model, steps=k, device=model.load_device,
                    sampler=rt.get("sampler_name", "euler"),
                    scheduler=rt.get("scheduler", "simple"), denoise=1.0,
                    model_options=model.model_options)
                out_s = ks.sample(noise, positive, [], cfg=CFG, latent_image=latent_fix,
                                  denoise_mask=None, sigmas=fix_sig, callback=None,
                                  disable_pbar=False, seed=s, force_full_denoise=True)
            v_fix, _ = h3_conditioning.unpack_nested_latent({"samples": out_s})
        except Exception as e:
            import traceback; traceback.print_exc()
            raise RuntimeError(f"[H3-FaceResample] resample failed: {e}"
                               f"\n[H3-FaceResample] 重采样失败: {e}")
        if v_fix is None or v_fix.dim() != 5 or int(v_fix.shape[2]) != T_exp:
            raise RuntimeError(f"[H3-FaceResample] unexpected output shape: "
                               f"{None if v_fix is None else tuple(v_fix.shape)}"
                               f"\n[H3-FaceResample] 输出形状异常: "
                               f"{None if v_fix is None else tuple(v_fix.shape)}")

        # ---- 生成画布解码 (1D) → 原生分辨率输出, 一切缩放由 ③ 完成 ----
        px = vae.decode(v_fix)
        if px.dim() == 4:
            px = px.unsqueeze(1)
        px = px[0].clamp(0.0, 1.0).float()
        print(f"[H3-FaceResample] done: images {tuple(px.shape)} "
              f"(native canvas, scaling/blend-back done by ③)\n"
              f"[H3-FaceResample] 完成: images {tuple(px.shape)} "
              f"(原生画布, 缩放/贴回由 ③ 完成)")
        return io.NodeOutput(px.contiguous(), dict(pack))

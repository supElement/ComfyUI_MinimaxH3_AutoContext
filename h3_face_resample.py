"""h3_face_resample.py — 修脸第 2 步: 画布重采样 (v9, 逐子轨, 输入已归一化 res²)

crop_images 来自 ① v10: 行序=子轨序、已归一化 res², 本节点不再缩放, 直接分块编码
(首块 ≤73 帧 ctx=0, 其后 17m 帧 + ctx=22 头部冻结 noise_mask=0), 采样→解码→裁掉
ctx 头, 行 1:1 替换, 输出画布行结构与 crop_images 完全一致 (canvas_off=crop_off)。
σ 阶梯由 sigmas 端口决定; skip 子轨无行; 输出契约 (canvas IMAGE + bbox) 不变。
"""

import torch
import comfy.sample
import comfy.samplers
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


def _plan_blocks(K):
    """子轨内分块 [(b0, b1, ctx)]: 首块 new=min(73,K) ctx=0; 其后 new=68/余量 ctx=22。
    enc = ctx+new 恒 ≡5 (mod 17)。"""
    if K < 5:
        return []
    if K <= 73:
        return [(0, K, 0)]
    out = [(0, 73, 0)]
    rem = K - 73
    while rem > 0:
        take = 68 if rem > 68 else rem
        b0 = out[-1][1]
        out.append((b0, b0 + take, 22))
        rem -= take
    return out


def _pick_prompt(fc, boundaries, seg_prompts, long_prompt):
    if not seg_prompts:
        return long_prompt, "global prompt / 全局提示词"
    idx = min(sum(1 for b in (boundaries or []) if b <= fc), len(seg_prompts) - 1)
    return seg_prompts[idx], f"segment {idx + 1} prompt / 段{idx + 1}提示词"


def _normalize_sigmas(sigmas, device):
    if sigmas is None:
        raise ValueError("[H3-FaceResample] sigmas port not connected\n"
                         "[H3-FaceResample] sigmas 端口未连接 — 请接调度器输出")
    s = sigmas.detach().clone().flatten().to(device=device, dtype=torch.float32)
    if s.numel() < 2:
        raise ValueError(f"[H3-FaceResample] sigmas needs >= 2 values, got {s.numel()}"
                         f"\n[H3-FaceResample] sigmas 至少 2 个值, 收到 {s.numel()} 个")
    if not bool(torch.all(s[1:] <= s[:-1])):
        raise ValueError(f"[H3-FaceResample] sigmas must be non-increasing: "
                         f"{[round(float(x), 4) for x in s]}"
                         f"\n[H3-FaceResample] sigmas 必须单调不增: "
                         f"{[round(float(x), 4) for x in s]}")
    if float(s[0]) <= 0.0:
        raise ValueError("[H3-FaceResample] first sigma must be > 0"
                         "\n[H3-FaceResample] 首 σ 必须 > 0")
    if float(s[-1]) != 0.0:
        s = torch.cat([s, torch.zeros(1, device=device)])
        print(f"[H3-FaceResample] last sigma != 0, appended 0: {[round(float(x), 4) for x in s]}\n"
              f"[H3-FaceResample] sigmas 末值非 0, 自动补 0: {[round(float(x), 4) for x in s]}")
    return s


class H3FaceResample(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="H3FaceResample",
            display_name="Minimax_H3_Face_Resample",
            category="MinimaxH3_AutoContext/FaceFix",
            description="Face fix step 2: per-subtrack resample, crops already normalized to res^2 by step 1.\n"
                        "修脸第 2 步: 逐子轨重采样, 裁剪已由第 1 步归一化 res²。",
            inputs=[
                io.Image.Input("crop_images", tooltip="crop_images from H3FaceCut v10 (uniform res^2, "
                               "row order = subtrack order)\n来自 H3FaceCut v10 的 crop_images "
                               "(统一 res², 行序=子轨序)"),
                io.Dict.Input("face_pack", tooltip="face_pack from H3FaceCut (v5)\n来自 H3FaceCut 的 face_pack (v5)"),
                io.Dict.Input("info", tooltip="info output of the sampler node (h3_runtime)\n"
                              "采样节点的 info 输出 (需含 h3_runtime)"),
                io.Sigmas.Input("sigmas", tooltip="Refinement sigma schedule (decreasing, last=0)\n"
                                "精修 σ 阶梯 (单调递减, 末值0)"),
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
        if int(pack.get("version") or 0) != 5:
            raise ValueError("[H3-FaceResample] pack version mismatch — rerun H3FaceCut (v10)"
                             "\n[H3-FaceResample] pack 版本不符 — 请重跑 H3FaceCut (v10)")
        a_lat = pack.get("a_lat")
        subs_in = pack.get("subtracks") or []
        subs = [dict(st) for st in subs_in]
        todo = [st for st in subs if not st.get("skip") and st.get("crop_off") is not None]
        if not todo or a_lat is None:
            print("[H3-FaceResample] no samplable subtrack, empty canvas passed\n"
                  "[H3-FaceResample] 无可采样子轨, 输出空画布")
            return io.NodeOutput(torch.zeros(1, 64, 64, 3),
                                 {"version": 5, "subtracks": subs, "a_lat": a_lat,
                                  "meta": pack.get("meta", {})})

        rt = info.get("h3_runtime") if isinstance(info, dict) else None
        if not rt:
            raise ValueError("[H3-FaceResample] info does not contain h3_runtime"
                             "\n[H3-FaceResample] info 中没有 h3_runtime")
        model = rt["model"]; clip = rt["clip"]; vae = rt["vae"]
        audio_vae = rt.get("audio_vae")
        device = comfy.model_management.get_torch_device()
        res = int((pack.get("meta") or {}).get("res", 512))
        if crop_images is None or crop_images.dim() != 4 \
                or int(crop_images.shape[1]) != res or int(crop_images.shape[2]) != res:
            raise ValueError(f"[H3-FaceResample] crop_images must be [N,{res},{res},3] from H3FaceCut v10\n"
                             f"[H3-FaceResample] crop_images 必须是 [N,{res},{res},3] (来自 H3FaceCut v10)")
        if crop_images.shape[0] != int(pack.get("n_crop_rows") or 0):
            raise ValueError("[H3-FaceResample] crop_images rows do not match pack accounting\n"
                             "[H3-FaceResample] crop_images 行数与 pack 账目不符 — 请重新运行 H3FaceCut")
        boundaries = info.get("boundaries") or []
        seg_prompts = info.get("segment_prompts") or []
        long_prompt = rt.get("long_prompt") or ""
        crop_mode = rt.get("crop_mode", "stretch")
        ref_fps = int(rt.get("fps", 24))
        Ta = int(a_lat.shape[-1])

        fix_sig = _normalize_sigmas(sigmas, device)
        k = int(fix_sig.numel()) - 1
        print(f"[H3-FaceResample] fix_sig ({k} steps), canvas {res}x{res}, "
              f"{len(todo)}/{len(subs)} subtracks sampled\n"
              f"[H3-FaceResample] fix_sig ({k} 步), 画布 {res}x{res}, "
              f"{len(todo)}/{len(subs)} 条子轨待采样")

        ri = (h3_sampler._prepare_ref_images(rt.get("ref_images"), vae, device, res, res, crop_mode)
              if rt.get("ref_images") else [])
        rv = (h3_sampler._prepare_ref_videos(rt.get("ref_videos"), vae, audio_vae,
                                             device, res, res, ref_fps, crop_mode)
              if rt.get("ref_videos") else [])
        ra = (h3_sampler._prepare_ref_audios(rt.get("ref_audios"), audio_vae, device)
              if rt.get("ref_audios") else [])

        canvas_parts = []
        gi = 0
        for st in subs:
            if st.get("skip") or st.get("crop_off") is None:
                st["canvas_off"] = None
                continue
            off, f0, f1, S = int(st["crop_off"]), int(st["f0"]), int(st["f1"]), int(st["S"])
            K = f1 - f0
            if (K - 5) % 17 != 0:
                raise RuntimeError(f"[H3-FaceResample] subtrack [{f0},{f1}) length {K} off 17n+5 grid\n"
                                   f"[H3-FaceResample] 子轨 [{f0},{f1}) 长度 {K} 不在 17n+5 网格上")
            rows = crop_images[off:off + K].float()          # 本子轨的 crop 行
            st["canvas_off"] = off                            # 行 1:1 替换, 偏移不变
            for bi, (b0, b1, ctx) in enumerate(_plan_blocks(K)):
                comfy.model_management.throw_exception_if_processing_interrupted()
                L = (b1 - b0) + ctx
                T_blk = h3ff.video_latent_frames(L)
                enc_px = rows[b0 - ctx:b1]
                enc0 = f0 + b0 - ctx
                prompt, src_name = _pick_prompt(f0 + (b0 + b1) // 2, boundaries,
                                                seg_prompts, long_prompt)
                base_lat = h3ff.encode_frames_adaptive(vae, enc_px.contiguous(), want_t=T_blk,
                                                       tag=f"[子轨{f0}-{f1} 块{bi + 1}]")
                if base_lat is None:
                    raise RuntimeError("[H3-FaceResample] canvas encoding failed\n"
                                       "[H3-FaceResample] 画布编码失败")
                base_lat = base_lat.to(device=device, dtype=torch.float32)
                a0 = max(0, min(Ta - 1, int(round(enc0 / ref_fps * h3ff.AUDIO_LATENTS_PER_SEC))))
                a1 = max(a0 + 1, min(Ta, int(round((f0 + b1) / ref_fps * h3ff.AUDIO_LATENTS_PER_SEC))))
                audio = a_lat[..., a0:a1].to(device=device, dtype=torch.float32)
                try:
                    latent_i = comfy.nested_tensor.NestedTensor((base_lat, audio))
                except Exception:
                    latent_i = (base_lat, audio)
                mask = None
                if ctx > 0:
                    vh = min(h3ff.video_latent_frames(ctx), T_blk - 1)
                    ah = min(max(1, int(round(ctx / ref_fps * h3ff.AUDIO_LATENTS_PER_SEC))),
                             max(1, int(audio.shape[-1]) - 1))
                    vm = torch.ones_like(base_lat); vm[:, :, :vh] = 0.0
                    am = torch.ones_like(audio); am[..., :ah] = 0.0
                    try:
                        mask = comfy.nested_tensor.NestedTensor((vm, am))
                    except Exception:
                        mask = (vm, am)

                payload = h3_conditioning.build_conditioning_payload(
                    seed=int(rt.get("seed", 0)) + 1 + gi, frame_count=b1 - b0,
                    ref_img_data=ri, ref_vid_data=rv,
                    ref_aud_latents=[a["latent"] for a in ra], fps=ref_fps)
                positive = h3_conditioning.encode_text_with_references(
                    clip, prompt, payload["ref_items_for_clip"], device,
                    images_for_clip=payload.get("images_for_clip"))
                positive = h3_conditioning.inject_conditioning_data(positive, payload)

                s = int(seed) + 1 + gi
                gi += 1
                noise = comfy.sample.prepare_noise(latent_i, s)
                try:
                    if rt.get("sampler_obj") is not None:
                        out_s = comfy.sample.sample_custom(
                            model, noise, CFG, rt["sampler_obj"], fix_sig, positive, [],
                            latent_i, noise_mask=mask, disable_pbar=False, seed=s)
                    else:
                        ks = comfy.samplers.KSampler(
                            model, steps=k, device=model.load_device,
                            sampler=rt.get("sampler_name", "euler"),
                            scheduler=rt.get("scheduler", "simple"), denoise=1.0,
                            model_options=model.model_options)
                        out_s = ks.sample(noise, positive, [], cfg=CFG, latent_image=latent_i,
                                          denoise_mask=mask, sigmas=fix_sig, callback=None,
                                          disable_pbar=False, seed=s, force_full_denoise=True)
                    v_i, _ = h3_conditioning.unpack_nested_latent({"samples": out_s})
                except Exception as e:
                    import traceback; traceback.print_exc()
                    raise RuntimeError(f"[H3-FaceResample] block failed: {e}"
                                       f"\n[H3-FaceResample] 块重采样失败: {e}")
                if v_i is None or v_i.dim() != 5 or int(v_i.shape[2]) != T_blk:
                    raise RuntimeError(f"[H3-FaceResample] bad output shape: "
                                       f"{None if v_i is None else tuple(v_i.shape)}"
                                       f"\n[H3-FaceResample] 输出形状异常: "
                                       f"{None if v_i is None else tuple(v_i.shape)}")
                px_i = vae.decode(v_i)
                if px_i.dim() == 4:
                    px_i = px_i.unsqueeze(1)
                px_i = px_i[0].clamp(0.0, 1.0).float()[ctx:].cpu()
                canvas_parts.append(px_i)
                print(f"[H3-FaceResample] subtrack [{f0},{f1}) block {bi + 1}: enc [{enc0},{f0 + b1}) "
                      f"(ctx {ctx}) keep {int(px_i.shape[0])}, prompt: {src_name}\n"
                      f"[H3-FaceResample] 子轨 [{f0},{f1}) 块 {bi + 1}: 编码 [{enc0},{f0 + b1}) "
                      f"(上下文 {ctx}) 保留 {int(px_i.shape[0])}, 提示词: {src_name}")
                del base_lat, audio, latent_i, noise, out_s, v_i, px_i
                comfy.model_management.soft_empty_cache()

        px = torch.cat(canvas_parts, dim=0)
        if int(px.shape[0]) != int(pack.get("n_crop_rows") or 0):
            raise RuntimeError("[H3-FaceResample] canvas row accounting mismatch\n"
                               "[H3-FaceResample] 画布行数账目不符")
        print(f"[H3-FaceResample] done: canvas {tuple(px.shape)} ({int(px.shape[0])} rows, "
              f"same row structure as crop_images)\n"
              f"[H3-FaceResample] 完成: 画布 {tuple(px.shape)} "
              f"({int(px.shape[0])} 行, 与 crop_images 行结构一致)")
        return io.NodeOutput(px.contiguous(), {"version": 5, "subtracks": subs,
                                               "a_lat": a_lat, "meta": pack.get("meta", {})})

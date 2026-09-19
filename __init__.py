""" ComfyUI-H3-Auto-Context 一键式 MiniMax H3 长视频自动化生成节点 """
from .nodes import H3AutoContextSampler, H3ParameterNode

try:
    from .seam_correction import H3SeamCorrection
except ImportError:
    H3SeamCorrection = None
    print("[H3-AutoContext] 警告：H3SeamCorrection 加载失败，请检查依赖（如 transnetv2-pytorch）。")

try:
    from .h3_tst_patch import H3TSTPatch
except ImportError:
    H3TSTPatch = None
    print("[H3-AutoContext] 警告：H3TSTPatch 加载失败，TST 注意力校正不可用"
          "（需要较新版本 ComfyUI：依赖 comfy_api.latest 与 comfy.patcher_extension）。")

NODE_CLASS_MAPPINGS = {
    "H3AutoContextSampler": H3AutoContextSampler,
    "H3Parameter": H3ParameterNode,
}

if H3SeamCorrection is not None:
    NODE_CLASS_MAPPINGS["H3SeamCorrection"] = H3SeamCorrection
if H3TSTPatch is not None:
    NODE_CLASS_MAPPINGS["H3TSTPatch"] = H3TSTPatch

try:
    from .h3_face_cut import H3FaceCut
    from .h3_face_resample import H3FaceResample
    from .h3_face_blend import H3FaceBlend
    NODE_CLASS_MAPPINGS["H3FaceCut"] = H3FaceCut
    NODE_CLASS_MAPPINGS["H3FaceResample"] = H3FaceResample
    NODE_CLASS_MAPPINGS["H3FaceBlend"] = H3FaceBlend
except Exception as _e:
    print(f"[H3-AutoContext] 警告：修脸三节点加载失败: {_e}")


WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]

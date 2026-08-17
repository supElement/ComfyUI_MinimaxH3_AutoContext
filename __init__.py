"""
ComfyUI-H3-Auto-Context
一键式 MiniMax H3 长视频自动化生成节点
"""

from .nodes import H3AutoContextSampler

NODE_CLASS_MAPPINGS = {
    "H3AutoContextSampler": H3AutoContextSampler,
}


WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]
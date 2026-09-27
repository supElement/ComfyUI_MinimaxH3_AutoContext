"""sec_engine — 内置于 ComfyUI_MinimaxH3_AutoContext 的 SeC-4B 推理引擎。

整个目录内容原样取自 Comfyui-SecNodes 的 inference/ 包 (上游为 OpenIXCLab/SeC):
- configuration_*/modeling_*  : SeC 主模型 + InternViT/InternLM2/Phi3 主干定义
- sam2/                       : SAM2 grounding encoder (含 configs/ 下 hydra yaml)
- model_config/               : SeC LVLM config + tokenizer (放在本目录内, 或扩展根目录亦可)

本文件仅做导入适配: 将本目录以包名 "inference" 注册进 sys.modules 后再加载,
使上游文件内部无论使用 `from inference.sam2 import ...` 绝对导入还是相对导入
都能原样工作, 上游源码零修改。
"""
import importlib.util
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG = "inference"          # 与上游原始包名保持一致

# model_config 中必须存在的关键文件 (用于放置错误的清晰报错)
_REQUIRED_CONFIG_FILES = ("config.json",)


def _load_core():
    if _PKG in sys.modules:
        return sys.modules[_PKG]
    init_py = os.path.join(_DIR, "__init__.py")
    if not os.path.isfile(init_py):
        raise RuntimeError(
            f"[H3-FaceFix] sec_engine incomplete: {init_py} missing\n"
            f"[H3-FaceFix] sec_engine 不完整: 缺少 {init_py} "
            f"(应原样拷贝 Comfyui-SecNodes/inference/ 整个目录)")
    spec = importlib.util.spec_from_file_location(
        _PKG, init_py, submodule_search_locations=[_DIR])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[_PKG] = pkg          # 先注册再执行, 保证内部绝对/相对导入均可用
    spec.loader.exec_module(pkg)
    return pkg


def _core_classes():
    pkg = _load_core()
    cfg = getattr(pkg, "SeCConfig", None)
    mdl = getattr(pkg, "SeCModel", None)
    if cfg is None or mdl is None:
        # 上游 inference/__init__.py 可能不 re-export, 从子模块显式取
        cfg_mod = importlib.import_module(f"{_PKG}.configuration_sec")
        mdl_mod = importlib.import_module(f"{_PKG}.modeling_sec")
        cfg, mdl = cfg_mod.SeCConfig, mdl_mod.SeCModel
    return cfg, mdl


def is_available():
    try:
        _core_classes()
        return True
    except Exception:
        return False


def import_error():
    try:
        _core_classes()
        return ""
    except Exception as e:
        return str(e)


def core():
    """返回 (SeCConfig, SeCModel)。不可用时抛出带原因的异常。"""
    try:
        return _core_classes()
    except Exception as e:
        raise RuntimeError(
            "[H3-FaceFix] built-in SeC engine import failed: " + str(e) + "\n"
            "[H3-FaceFix] 内置 SeC 引擎导入失败: " + str(e)) from e


def config_dir():
    """定位 model_config 目录 (SeC LVLM config + tokenizer)。
    兼容两种放置方式, 均为随扩展分发的内置文件:
      1) sec_engine/model_config/    ← 本扩展推荐 (与引擎同目录)
      2) 扩展根目录/model_config/
    """
    candidates = [
        os.path.join(_DIR, "model_config"),                     # sec_engine/model_config
        os.path.join(os.path.dirname(_DIR), "model_config"),    # 扩展根/model_config
    ]
    for d in candidates:
        if not os.path.isdir(d):
            continue
        missing = [f for f in _REQUIRED_CONFIG_FILES
                   if not os.path.isfile(os.path.join(d, f))]
        if missing:
            found = sorted(os.listdir(d))
            raise RuntimeError(
                f"[H3-FaceFix] model_config dir found but incomplete: {d}\n"
                f"  missing: {missing}\n  found: {found}\n"
                f"[H3-FaceFix] model_config 目录不完整: {d}\n"
                f"  缺少: {missing}\n  现有: {found}\n"
                f"(应含 config.json / tokenizer.json / vocab.json / merges.txt 等全部 8 个文件)")
        return d
    raise RuntimeError(
        "[H3-FaceFix] model_config directory not found, searched:\n  "
        + "\n  ".join(candidates)
        + "\n[H3-FaceFix] 未找到 model_config 目录, 已搜索:\n  "
        + "\n  ".join(candidates)
        + "\n(应拷贝 Comfyui-SecNodes/model_config/ 到 sec_engine/ 内或扩展根目录)")

import os
import threading
import queue
import torch
import hashlib
import numpy as np
from comfy.nested_tensor import NestedTensor

def compute_input_hash(input_latent):
    """
    计算输入 Latent 的指纹哈希（仅取视频部分的前1024个元素）。
    用于检测一采结果是否变化，从而决定二采缓存是否失效。
    """
    if input_latent is None:
        return None
    v_lat = None
    if hasattr(input_latent, "is_nested") and input_latent.is_nested:
        tensors = list(input_latent.unbind())
        for t in tensors:
            if t.dim() == 5:
                v_lat = t
                break
    elif isinstance(input_latent, dict):
        samples = input_latent.get("samples")
        if samples is not None:
            if hasattr(samples, "is_nested") and samples.is_nested:
                for t in samples.unbind():
                    if t.dim() == 5:
                        v_lat = t
                        break
            elif isinstance(samples, torch.Tensor):
                v_lat = samples
    else:
        v_lat = input_latent

    if v_lat is None:
        return None
    flat = v_lat.flatten()
    if flat.numel() == 0:
        return None
    sample = flat[:1024].cpu().numpy().tobytes()
    return hashlib.md5(sample).hexdigest()


def normalize_cache_dir(raw_dir):
    """将路径统一为系统标准格式，支持正/反斜杠"""
    if not raw_dir:
        return ""
    normalized = os.path.normpath(raw_dir.replace('\\', '/'))
    return normalized

# ------------------ 内部工具函数 ------------------
def _unwrap_nested(tensor):
    """将 NestedTensor 解包为普通张量列表，便于存储"""
    if isinstance(tensor, NestedTensor):
        return [t.cpu() for t in tensor.unbind()]
    if isinstance(tensor, (list, tuple)):
        return [t.cpu() for t in tensor]
    return tensor.cpu()

def _wrap_nested(tensors):
    """将张量列表重新打包为 NestedTensor"""
    if isinstance(tensors, (list, tuple)) and len(tensors) > 0:
        if len(tensors) == 1 and not isinstance(tensors[0], (list, tuple)):
            return tensors[0]
        try:
            return NestedTensor(tuple(tensors))
        except:
            return tensors
    return tensors

# ------------------ 缓存文件操作 ------------------
def get_cache_path(cache_dir, seg_idx):
    """生成段缓存文件路径"""
    if not cache_dir:
        return None
    normalized_dir = normalize_cache_dir(cache_dir)
    os.makedirs(normalized_dir, exist_ok=True)
    return os.path.join(normalized_dir, f"seg_{seg_idx:04d}.pt")

def save_segment_latent_sync(cache_dir, seg_idx, samples, x0, metadata=None):
    """同步保存一段 latent（供异步线程调用）"""
    if not cache_dir:
        return
    path = get_cache_path(cache_dir, seg_idx)
    if not path:
        return

    if isinstance(samples, dict):
        samples_tensor = samples.get("samples")
    else:
        samples_tensor = samples

    samples_unwrapped = _unwrap_nested(samples_tensor)
    x0_unwrapped = _unwrap_nested(x0) if x0 is not None else None

    data = {
        "samples": samples_unwrapped,
        "x0": x0_unwrapped,
        "metadata": metadata or {},
    }
    temp_path = path + ".tmp"
    try:
        torch.save(data, temp_path)
        os.replace(temp_path, path)  
    except Exception as e:
        print(f"[H3-Cache] Failed to save segment {seg_idx}: {e}\n[H3-Cache] 保存段 {seg_idx} 失败: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def load_segment_latent(cache_dir, seg_idx, current_metadata=None):
    """
    加载一段缓存，若提供了 current_metadata 则校验关键参数。
    返回 (samples_dict, x0) 或 (None, None)
    """
    if not cache_dir:
        return None, None
    path = get_cache_path(cache_dir, seg_idx)
    if not os.path.exists(path):
        return None, None

    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[H3-Cache] Failed to load segment {seg_idx}: {e}\n[H3-Cache] 加载段 {seg_idx} 失败: {e}")
        return None, None

    saved_meta = data.get("metadata", {})

    if current_metadata:
        sensitive_keys = [
            "seed", "steps", "cfg", "sampler_name", "scheduler",
            "denoise", "video_context_denoise", "sigmas_hash", "input_slice_hash",
            "latent_w", "latent_h",
            "seg_frames", "context_frames", "fps",
            "lock_audio", "audio_drive",
            "window_prompt_hash",
            "conditions_hash",
            "segment_fingerprint",      
            "upstream_global_hash",
            "video_guide", "tst_tau",
        ]

        mismatch = False
        for k in current_metadata.keys():
            if k in sensitive_keys:
                if saved_meta.get(k) != current_metadata[k]:
                    print(f"   Mismatch on key: {k}, saved={saved_meta.get(k)}, current={current_metadata[k]}"
                          f"\n   键 {k} 不一致: 已保存={saved_meta.get(k)}, 当前={current_metadata[k]}")
                    mismatch = True
                    break

        if mismatch:
            print("\033[33m" + f"[H3-Cache] Segment {seg_idx+1} parameters or input fingerprint changed, deleting stale cache\n[H3-Cache] 段 {seg_idx+1} 参数或输入指纹变更，删除旧缓存" + "\033[0m")
            try:
                os.remove(path)
            except:
                pass
            return None, None

    samples_unwrapped = data["samples"]
    x0_unwrapped = data.get("x0")

    samples_tensor = _wrap_nested(samples_unwrapped)
    x0_tensor = _wrap_nested(x0_unwrapped) if x0_unwrapped is not None else None

    samples_dict = {"samples": samples_tensor}
    return samples_dict, x0_tensor

# ------------------ 异步保存队列 ------------------
_save_queue = queue.Queue()
_save_thread_started = False

def _save_worker():
    """后台线程工作函数"""
    while True:
        try:
            cache_dir, seg_idx, samples, x0, metadata = _save_queue.get(timeout=1)
            if cache_dir is None:  
                break
            save_segment_latent_sync(cache_dir, seg_idx, samples, x0, metadata)
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[H3-Cache] Async save thread error: {e}\n[H3-Cache] 异步保存线程异常: {e}")

def start_save_thread():
    """启动后台保存线程（只启动一次）"""
    global _save_thread_started
    if not _save_thread_started:
        thread = threading.Thread(target=_save_worker, daemon=True)
        thread.start()
        _save_thread_started = True

def save_segment_latent_async(cache_dir, seg_idx, samples, x0, metadata=None):
    """异步保存一段 latent（将任务放入队列）"""
    if not cache_dir:
        return
    start_save_thread()
    _save_queue.put((cache_dir, seg_idx, samples, x0, metadata))

def flush_save_queue():
    """等待所有保存任务完成（可选，在程序退出前调用）"""
    _save_queue.join()
    
# ================= 通用缓存块 (FaceCut / FaceResample 用, 与段缓存同规则) =================
_blob_thread_started = False
_blob_queue = queue.Queue()


def save_blob_sync(cache_dir, name, payload, metadata=None):
    """保存任意 torch.save 可序列化结构 (dict/tensor/list/str/...) 为单个缓存块 (原子写)。"""
    if not cache_dir or not name:
        return
    directory = normalize_cache_dir(cache_dir)
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception:
        return
    path = os.path.join(directory, name)
    tmp = path + ".tmp"
    try:
        torch.save({"payload": payload, "metadata": metadata or {}}, tmp)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[H3-Cache] Failed to save blob {name}: {e}\n[H3-Cache] 缓存块 {name} 保存失败: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)


def load_blob(cache_dir, name, current_metadata=None, sensitive_keys=()):
    """加载缓存块；任一敏感键与当前值不一致 → 删除旧缓存并返回 None (与段缓存同规则)。"""
    if not cache_dir or not name:
        return None
    path = os.path.join(normalize_cache_dir(cache_dir), name)
    if not os.path.exists(path):
        return None
    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[H3-Cache] Failed to load blob {name}: {e}\n[H3-Cache] 缓存块 {name} 加载失败: {e}")
        return None
    if current_metadata:
        saved = data.get("metadata", {})
        for k in sensitive_keys:
            if k in current_metadata and saved.get(k) != current_metadata[k]:
                print("\033[33m" + f"[H3-Cache] blob {name} mismatch on '{k}', deleting stale cache\n"
                      f"[H3-Cache] 缓存块 {name} 键 '{k}' 不一致，删除旧缓存" + "\033[0m")
                try:
                    os.remove(path)
                except Exception:
                    pass
                return None
    return data.get("payload")


def clear_cache_dir(cache_dir):
    """删除整个缓存目录 (节点 clear_cache 开关用)。返回是否删除了内容。"""
    if not cache_dir:
        return False
    d = normalize_cache_dir(cache_dir)
    if os.path.isdir(d):
        import shutil
        shutil.rmtree(d, ignore_errors=True)
        return True
    return False


def _blob_worker():
    while True:
        try:
            item = _blob_queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            if item is None:
                break
            save_blob_sync(*item)
        except Exception as e:
            print(f"[H3-Cache] Async blob thread error: {e}\n[H3-Cache] 异步缓存线程异常: {e}")


def save_blob_async(cache_dir, name, payload, metadata=None):
    """异步保存缓存块 (后台线程, 与段缓存异步保存同风格)。"""
    global _blob_thread_started
    if not cache_dir:
        return
    if not _blob_thread_started:
        threading.Thread(target=_blob_worker, daemon=True).start()
        _blob_thread_started = True
    _blob_queue.put((cache_dir, name, payload, metadata))

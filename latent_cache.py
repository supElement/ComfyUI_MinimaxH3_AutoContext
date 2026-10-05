import os
import re 
import time
import threading
import queue
import torch
import hashlib
import numpy as np
from comfy.nested_tensor import NestedTensor


# ================= 内容指纹工具 (全流水线共用) =================
# 设计原则: 有界成本 + 可控灵敏度。
# 全量哈希 (GB 级 md5) 在每次运行时同步阻塞数秒, 只用于最终产物;
# 缓存键/输入变化检测一律用 strided 采样签名, 成本 O(采样数) 而非 O(数据量)。

def tensor_sig(t, max_elems=8192):
    """张量内容签名: shape + 等距采样元素 md5。
    任何密度高于采样步长的局部改动都会改变签名 — 对视频/画布这类
    "改动至少波及一整帧/一行" 的输入, 单帧改动也几乎必然命中。
    非 tensor 输入退化为类型名 (保持稳定)。"""
    if not torch.is_tensor(t):
        return "n:" + str(type(t).__name__)
    n = t.numel()
    if n == 0:
        return f"{tuple(int(x) for x in t.shape)}:empty"
    step = max(1, n // max_elems)
    s = t.detach().reshape(-1)[::step][:max_elems]
    try:
        b = s.to(torch.float32).cpu().numpy().tobytes()
    except Exception:
        try:
            b = s.detach().cpu().numpy().tobytes()
        except Exception:
            return f"{tuple(int(x) for x in t.shape)}:unhashable"
    return f"{tuple(int(x) for x in t.shape)}:{hashlib.md5(b).hexdigest()}"


def _iter_tensors(x, out):
    """递归收集嵌套结构里的全部 tensor (LoRA patch 条目格式多变, 统一展平搜索)。"""
    if torch.is_tensor(x):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            _iter_tensors(y, out)
    elif isinstance(x, dict):
        for y in x.values():
            _iter_tensors(y, out)


def _fn_id(fn):
    """函数的稳定标识 (跨重启不变): module.qualname。functools.partial 展开 内函数+关键字。
    绝不用 str(fn)/id(fn) — 前者含内存地址 (每次重启变, 缓存永远失效), 后者同。"""
    import functools
    try:
        if isinstance(fn, functools.partial):
            inner = _fn_id(fn.func)
            kw = ",".join(f"{k}={_stable_repr(v, 4)}" for k, v in
                          sorted((str(k), v) for k, v in (fn.keywords or {}).items())[:16])
            return f"partial({inner}|{kw})"
        mod = getattr(fn, "__module__", None) or "?"
        qn = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
        if qn is None:
            return "callable:" + type(fn).__name__
        return f"{mod}.{qn}"
    except Exception:
        return "fn:err"


def _stable_repr(x, depth=0):
    """任意对象的稳定字符串表示 (有界): 供模型指纹使用。
    dict 排序键; callable 用 _fn_id; tensor 用小预算内容签名; 未知类型只记类型名。
    深度/条目封顶, 防病态结构。"""
    try:
        if depth > 6:
            return "…"
        if x is None or isinstance(x, (bool, int, float, str, bytes)):
            return repr(x)[:128]
        if torch.is_tensor(x):
            return tensor_sig(x, 128)
        if callable(x):
            return "fn:" + _fn_id(x)
        if isinstance(x, dict):
            items = []
            for k, v in sorted(list(x.items())[:64], key=lambda kv: str(kv[0])):
                items.append(f"{k}:{_stable_repr(v, depth + 1)}")
            return "{" + ",".join(items) + "}"
        if isinstance(x, (list, tuple, set, frozenset)):
            vals = [_stable_repr(v, depth + 1) for v in list(x)[:64]]
            return "[" + ",".join(vals) + "]"
        return "t:" + type(x).__name__
    except Exception:
        return "repr:err"


def model_fingerprint(model):
    """扩散模型权重指纹 — 缓存键的模型部分。
    覆盖: checkpoint sha256 / LoRA patch 集合及其内容采样 / 函数级补丁 (object_patches,
    以 module.qualname 标识 — 覆盖 KJ SageAttention / Sol-Attn / Low VRAM Attention 等
    attention 补丁节点) / model_options 全量稳定表示 (覆盖 Model Attention Backend 等
    走 transformer_options/model_options 的节点) / model_sampling 对象 (覆盖
    ModelSamplingMiniMaxH3 — shift/v-shift 等属性直接入指纹) / 计算精度。
    换加速 LoRA、换 checkpoint、改挂载补丁 → 指纹变化 → 旧缓存自动失效。
    任何一步失败都不抛错 (软失败, 返回尽力而为的指纹)。"""
    parts = []
    try:
        m = getattr(model, "model", None)
        info = getattr(m, "sd_checkpoint_info", None) if m is not None else None
        if info is not None:
            sha = str(getattr(info, "sha256", "") or "")
            fn = str(getattr(info, "filename", "") or "")
            parts.append("ckpt:" + (sha or fn or str(info))[:128])
    except Exception:
        parts.append("ckpt:err")
    try:
        dt = str(model.model_dtype())
    except Exception:
        try:
            dt = str(getattr(getattr(model, "model", None), "dtype", None))
        except Exception:
            dt = "?"
    parts.append("dtype:" + dt)
    try:
        patches = getattr(model, "patches", None) or {}
        keys = sorted(patches.keys(), key=lambda k: str(k))
        n_hashed = 0
        for k in keys:
            if n_hashed >= 256:  
                parts.append(f"P:truncated:{len(keys)}")
                break
            for entry in patches[k]:
                ts = []
                _iter_tensors(entry, ts)
                sig = ",".join(tensor_sig(t, 2048) for t in ts)
                entry_str = re.sub(r"0x[0-9a-fA-F]+", "0xX", str(entry))[:96]
                parts.append("P:" + str(k) + "|" + entry_str + "|" + sig)
                n_hashed += 1
    except Exception:
        parts.append("patches:err")
    try:
        # model_sampling 对象: ModelSamplingMiniMaxH3 等 sigma↔t 映射节点。
        # sigmas 端口值不变但 shift 变时, 输出会变 — 必须入指纹 (否则假命中)。
        ms = getattr(getattr(model, "model", None), "model_sampling", None)
        if ms is not None:
            try:
                attrs = {k: v for k, v in vars(ms).items()
                         if isinstance(v, (bool, int, float, str, bytes)) or torch.is_tensor(v)}
                parts.append("MS:" + _stable_repr(attrs, 2))
            except Exception:
                parts.append("MS:" + type(ms).__name__)
    except Exception:
        pass
    try:
        to = getattr(model, "transformer_options", None)
        if isinstance(to, dict) and len(to) > 0:
            parts.append("TO:" + ",".join(sorted((str(k) for k in to.keys()))))
    except Exception:
        pass
    return hashlib.md5("|".join(parts).encode("utf-8", "ignore")).hexdigest()

# ================= 图指纹: 用户配置镜像 (C1-v3) =================
# 数据源: ComfyUI hidden prompt — 整张工作流的 API 序列化 dict:
#   {node_id: {"class_type": str, "inputs": {端口: 控件值 或 [源node_id, 槽位]}}}
# 纯 JSON 结构: 无内存地址、无 set 迭代序、不依赖任何节点的内部实现。
# 哈希内容 = 权重链沿途每个节点的 (class_type + 控件值 + 拓扑); 节点 id 只作
# 遍历去重, 不入哈希。效果: 用户不动上游控件/连线 → 图指纹不变 → 缓存命中;
# 补丁节点内部往 model_options 塞什么、对象是否每次新建 → 完全无关。

_WEIGHT_PORT_TYPES = {"MODEL", "VAE", "CLIP"}
_VALUE_PORT_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
_GRAPH_MAX_NODES = 256


def _node_class_mapping():
    try:
        from nodes import NODE_CLASS_MAPPINGS as m
        return m
    except Exception:
        try:
            from server import NODE_CLASS_MAPPINGS as m
            return m
        except Exception:
            return {}


def _is_link(v):
    """API prompt 里的连线形式为 [源node_id(str/int), 槽位int]"""
    return (isinstance(v, (list, tuple)) and len(v) == 2
            and isinstance(v[1], int) and not isinstance(v[1], bool)
            and isinstance(v[0], (str, int)))


def graph_weight_fingerprint(prompt, node_id, ports=("model", "vae", "audio_vae", "clip")):
    """权重链用户配置指纹。prompt=hidden prompt(整图 dict), node_id=本节点 id。
    从本节点的 model/vae/audio_vae/clip 端口沿 MODEL/VAE/CLIP 类型边向上遍历,
    哈希沿途 (class_type + 控件值 + 拓扑)。拿不到图返回 None (调用方回退)。"""
    if not isinstance(prompt, dict) or node_id is None:
        return None
    root = prompt.get(str(node_id))
    if not isinstance(root, dict):
        return None
    mapping = _node_class_mapping()

    def port_type(class_type, port):
        cls = mapping.get(class_type)
        if cls is None or not hasattr(cls, "INPUT_TYPES"):
            return None
        try:
            it = cls.INPUT_TYPES()
            for sec in ("required", "optional"):
                spec = (it.get(sec) or {}).get(port)
                if spec is not None:
                    return spec[0] if isinstance(spec, (list, tuple)) and spec else spec
        except Exception:
            pass
        return None

    def setnode_ids(varname):
        out = []
        for oid, nd in prompt.items():
            try:
                if (str(nd.get("class_type", "")).lower() == "setnode"
                        and str((nd.get("inputs") or {}).get("varname", "")) == varname):
                    out.append(str(oid))
            except Exception:
                pass
        return out

    h = hashlib.md5()
    visited = set()
    stack = []
    root_inputs = root.get("inputs") or {}
    for p in ports:
        v = root_inputs.get(p)
        if _is_link(v):
            stack.append(str(v[0]))
    while stack:
        if len(visited) >= _GRAPH_MAX_NODES:
            h.update(b"|truncated")  # 同一工作流同一拓扑 → 遍历顺序确定 → 仍稳定
            break
        nid = stack.pop()
        node = prompt.get(nid)
        if not isinstance(node, dict) or nid in visited:
            continue
        visited.add(nid)
        ct = str(node.get("class_type", ""))
        inputs = node.get("inputs") or {}

        # KJNodes GetNode: 无连线, 按 varname 配对跳到 SetNode 继续沿权重链向上
        if ct.lower() == "getnode":
            var = str(inputs.get("varname", ""))
            h.update(b"get\x1f" + var.encode("utf-8", "ignore") + b"\x1e")
            stack.extend(setnode_ids(var))
            continue

        h.update(ct.encode("utf-8", "ignore") + b"\x1e")
        for k in sorted(inputs.keys(), key=str):
            v = inputs[k]
            if _is_link(v):
                t = port_type(ct, str(k))
                # 只沿权重边/控件值边继续向上; 类型查不到的未知节点保守跟进
                if t is None or t in _WEIGHT_PORT_TYPES or t in _VALUE_PORT_TYPES:
                    stack.append(str(v[0]))
                # IMAGE/LATENT/AUDIO 等数据边不影响权重链, 到此为止
            else:
                h.update(str(k).encode("utf-8", "ignore") + b"\x1f"
                         + repr(v).encode("utf-8", "ignore") + b"\x1f")
    return h.hexdigest()


def vae_fingerprint(vae):
    """VAE 权重指纹 — 解码/编码结果直接受 VAE 权重影响, 缓存键必须包含。
    从 state_dict 中抽样 6 个张量做内容签名 (有界成本)。"""
    try:
        sd = getattr(vae, "sd", None)
        if not sd:
            return None
        parts = [f"n={len(sd)}"]
        for i, k in enumerate(sorted(sd.keys(), key=str)):
            if i >= 6:
                break
            parts.append(tensor_sig(sd[k], 2048))
        return hashlib.md5(";".join(parts).encode("utf-8")).hexdigest()
    except Exception:
        return None


def clip_fingerprint(clip):
    """文本编码器指纹 (Qwen3-VL 等) — CLIP 权重变化 → 文本条件变化。"""
    try:
        p = getattr(clip, "patcher", None)
        if p is None:
            return None
        return model_fingerprint(p)
    except Exception:
        return None


def compute_input_hash(input_latent):
    """
    计算输入 Latent 的指纹哈希。
    v2: 由"仅前 1024 个元素"改为 全量等距采样 8192 元素 + 形状 + 音频形状 —
    一采中后段的改动不再漏检 (旧实现对只改视频后半段的输入永远命中旧缓存)。
    用于检测一采结果是否变化，从而决定二采缓存是否失效。
    """
    if input_latent is None:
        return None
    v_lat = None
    a_lat = None
    if hasattr(input_latent, "is_nested") and input_latent.is_nested:
        tensors = list(input_latent.unbind())
        for t in tensors:
            if t.dim() == 5 and v_lat is None:
                v_lat = t
            elif a_lat is None:
                a_lat = t
    elif isinstance(input_latent, dict):
        samples = input_latent.get("samples")
        if samples is not None:
            if hasattr(samples, "is_nested") and samples.is_nested:
                for t in samples.unbind():
                    if t.dim() == 5 and v_lat is None:
                        v_lat = t
                    elif a_lat is None:
                        a_lat = t
            elif isinstance(samples, torch.Tensor):
                v_lat = samples
    else:
        v_lat = input_latent

    if v_lat is None:
        return None
    h = hashlib.md5()
    h.update(tensor_sig(v_lat, 8192).encode())
    if a_lat is not None:
        h.update(b"|a|")
        h.update(tensor_sig(a_lat, 1024).encode())
    return h.hexdigest()


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
        enforce_cache_budget(cache_dir)
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
            "semantic_bridge", "semantic_bridge_adapter", "semantic_bridge_alpha", "semantic_bridge_magnitude",
            "model_fp", "vae_fp", "clip_fp",
            "graph_fp",
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
    
# ================= 磁盘预算 (目录级 LRU, 防缓存无限膨胀) =================
# 环境变量 H3_CACHE_MAX_GB 控制单个缓存目录的总预算 (默认 32 GB, 设 0 关闭)。
# 超预算时按 mtime 从旧到新删除; 永不动最近 10 分钟内写入的文件 (当前运行)。

_BUDGET_ENV = "H3_CACHE_MAX_GB"
_DEFAULT_BUDGET_GB = 32.0
_BUDGET_CHECK_INTERVAL = 60.0   # 同一目录至多每 60s 巡检一次 (后台线程内执行)
_last_budget_check = {}


def _cache_budget_bytes():
    try:
        v = os.environ.get(_BUDGET_ENV, "").strip()
        if v:
            return max(0.0, float(v)) * (1024 ** 3)
    except Exception:
        pass
    return _DEFAULT_BUDGET_GB * (1024 ** 3)


def enforce_cache_budget(cache_dir):
    """目录级 LRU 清理。在保存线程中调用, 不阻塞采样主线程。"""
    if not cache_dir:
        return
    budget = _cache_budget_bytes()
    if budget <= 0:
        return
    directory = normalize_cache_dir(cache_dir)
    if not os.path.isdir(directory):
        return
    now = time.time()
    if now - _last_budget_check.get(directory, 0.0) < _BUDGET_CHECK_INTERVAL:
        return
    _last_budget_check[directory] = now

    files = []   # (mtime, size, path)
    total = 0
    try:
        for root, _dirs, names in os.walk(directory):
            for fn in names:
                p = os.path.join(root, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                # 残留 tmp (>24h, 崩溃遗留) 直接清; 新 tmp 跳过 (可能正在写)
                if fn.endswith(".tmp"):
                    if now - st.st_mtime > 86400:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
                    continue
                files.append((st.st_mtime, st.st_size, p))
                total += st.st_size
    except OSError:
        return
    if total <= budget:
        return

    files.sort()  # mtime 升序 = 最旧优先
    freed = 0
    removed = 0
    for mtime, size, p in files:
        if total <= budget or now - mtime < 600.0:
            break
        try:
            os.remove(p)
        except OSError:
            continue
        total -= size
        freed += size
        removed += 1
    if removed:
        print("\033[33m" + f"[H3-Cache] budget: removed {removed} old cache file(s), "
              f"freed {freed / (1024**3):.2f} GB (cap {budget / (1024**3):.0f} GB, "
              f"set {_BUDGET_ENV}=0 to disable)\n"
              f"[H3-Cache] 预算清理: 删除 {removed} 个旧缓存文件, "
              f"释放 {freed / (1024**3):.2f} GB (上限 {budget / (1024**3):.0f} GB, "
              f"设 {_BUDGET_ENV}=0 关闭)" + "\033[0m")


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
        enforce_cache_budget(cache_dir)
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

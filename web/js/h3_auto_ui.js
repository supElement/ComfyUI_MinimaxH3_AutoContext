import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "H3.AutoContextSampler",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3AutoContextSampler") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            this.infoWidget = this.addWidget(
                "text", "预计分段", "-", () => {
                    setTimeout(() => this._updateChunkInfo(), 0);
                }, { serialize: false }
            );

            const tryReadOnly = () => {
                if (this.infoWidget && this.infoWidget.inputEl) {
                    this.infoWidget.inputEl.readOnly = true;
                    this.infoWidget.inputEl.style.cursor = "default";
                    this.infoWidget.inputEl.style.opacity = "0.85";
                } else {
                    requestAnimationFrame(tryReadOnly);
                }
            };
            requestAnimationFrame(tryReadOnly);

            setTimeout(() => {
                this._updateChunkInfo();
                this._updatePromptModeVisibility();
                this._watchUpstream();
            }, 100);
            return r;
        };

        const onWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name, value, old_value) {
            if (onWidgetChanged) onWidgetChanged.apply(this, arguments);
            const watched = ["total_frames", "chunk_frames", "fps", "context_frames",
                             "clip_mode", "clip_tag", "long_prompt", "prompt_format"];
            if (watched.includes(name)) {
                setTimeout(() => this._updateChunkInfo(), 0);
            }
            if (name === "clip_mode") {
                setTimeout(() => this._updatePromptModeVisibility(), 0);
            }
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, slot, connected, link_info, input) {
            if (onConnectionsChange) onConnectionsChange.apply(this, arguments);
            if (type === LiteGraph.INPUT) {
                scheduleAutogrowReconcile(this);
                const name = this.inputs[slot]?.name;
                if (name === "total_frames" || name === "chunk_frames" || name === "fps" || name === "context_frames") {
                    setTimeout(() => {
                        this._updateChunkInfo();
                        this._watchUpstream();
                    }, 0);
                }
            }
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (onRemoved) onRemoved.apply(this, arguments);
            if (this._upstreamHooks) {
                for (const h of this._upstreamHooks) {
                    if (h.widget) h.widget.callback = h.origCb;
                }
                this._upstreamHooks = [];
            }
        };

        nodeType.prototype._updatePromptModeVisibility = function () {
            if (!this.widgets) return;
            const clipModeWidget = this.widgets.find(w => w.name === "clip_mode");
            const promptModeWidget = this.widgets.find(w => w.name === "prompt_mode");
            if (!clipModeWidget || !promptModeWidget) return;
            const isTag = clipModeWidget.value === "Clip_Tag";
            promptModeWidget.hidden = isTag;
            if (isTag && promptModeWidget.inputEl) {
                promptModeWidget.inputEl.style.display = "none";
            } else if (!isTag && promptModeWidget.inputEl) {
                promptModeWidget.inputEl.style.display = "";
            }
            if (this.setSize && this.graph) {
                this.setSize(this.size);
            }
        };

        nodeType.prototype._updateChunkInfo = function () {
            if (!this.widgets) return;

            const clipMode = readWidget(this, "clip_mode", "Clip_Frame");

            if (clipMode === "Clip_Tag") {
                const tagR = resolveInputValue(this, "clip_tag");
                const promptR = resolveInputValue(this, "long_prompt");
                const tagVal = tagR.value !== undefined ? String(tagR.value) : "段1";
                const promptVal = promptR.value !== undefined ? String(promptR.value) : "";
                const pat = parseTagPattern(tagVal);
                let text = "Clip_Tag 模式";
                if (pat) {
                    const n = countTagSegments(promptVal, pat.prefix, pat.suffix);
                    text = n > 0 ? `Clip_Tag: ${n} 段 (详见运行日志)` : "Clip_Tag: 未找到标签";
                }
                if (this.infoWidget) this.infoWidget.value = text;
                return;
            }

            const totalR = resolveInputValue(this, "total_frames");
            const chunkR = resolveInputValue(this, "chunk_frames");
            const ctxR = resolveInputValue(this, "context_frames");

            if ((totalR.connected && totalR.value === undefined) ||
                (chunkR.connected && chunkR.value === undefined) ||
                (ctxR.connected && ctxR.value === undefined)) {
                if (this.infoWidget) this.infoWidget.value = "已连接上游(无法预估)，以运行日志为准";
                return;
            }

            const totalFrames = parseInt(totalR.value !== undefined ? totalR.value : 362);
            const chunkFramesInput = parseInt(chunkR.value !== undefined ? chunkR.value : 90);
            const ctxFrames = parseInt(ctxR.value !== undefined ? ctxR.value : 22);

            let sizes, effContext = 0;
            if (chunkFramesInput <= 0) {
                sizes = [snapToGridUp(Math.max(5, totalFrames))];
            } else {
                const result = computeChunksJS(totalFrames, chunkFramesInput, ctxFrames);
                sizes = result.segSizes;
                effContext = result.effContext;
            }

            let text;
            if (sizes.length === 1) {
                text = `1 段 (${sizes[0]}帧, ${sizes[0] > totalFrames ? "超出" + (sizes[0] - totalFrames) + "帧" : "精准"})`;
            } else {
                const newFrames = [sizes[0]];
                for (let i = 1; i < sizes.length; i++) {
                    newFrames.push(sizes[i] - effContext);
                }
                const totalNew = newFrames.reduce((a, b) => a + b, 0);
                text = `${sizes.length} 段 [${sizes.join(", ")}] (新增${totalNew}帧, 目标${totalFrames})`;
            }

            if (this.infoWidget) {
                this.infoWidget.value = text;
            }
        };

        nodeType.prototype._watchUpstream = function () {
            if (this._upstreamHooks) {
                for (const h of this._upstreamHooks) {
                    if (h.widget) h.widget.callback = h.origCb;
                }
                this._upstreamHooks = [];
            }

            const watched = ["total_frames", "chunk_frames", "context_frames"];
            const hooks = [];
            const seenWidgets = new Set();
            const self = this;

            for (const name of watched) {
                if (!this.inputs) break;
                const slotIdx = this.inputs.findIndex(i => i.name === name);
                if (slotIdx < 0 || this.inputs[slotIdx].link == null) continue;
                const graph = getGraph(this);
                const origin = traceOrigin(graph, this.inputs[slotIdx].link);
                if (!origin || !origin.node || !origin.node.widgets) continue;
                for (const w of origin.node.widgets) {
                    if (!w || seenWidgets.has(w)) continue;
                    seenWidgets.add(w);
                    const origCb = w.callback;
                    w.callback = function (...args) {
                        if (origCb) origCb.apply(this, args);
                        setTimeout(() => self._updateChunkInfo(), 0);
                    };
                    hooks.push({ widget: w, origCb });
                }
            }
            this._upstreamHooks = hooks;
        };
    },
});

// ---------------------------------------------------------------------------
// Autogrow 动态端口对齐修复
// ---------------------------------------------------------------------------
// ComfyUI 自带 Autogrow 在“断开靠后组 + 同时连接靠前组”时（例如断开
// ref_video_0 改连 ref_image_0），断开的收缩处理被 requestAnimationFrame 延迟，
// 而靠前组会同步插入新端口使槽位下标整体后移，导致延迟回调按旧 slot 序号
// 操作了错误的输入。结果：ref_video_0/ref_audio_0 残留指向已删除 link 的 .link，
// 且多余尾端口（ref_video_1）不消失，排队时报
// “No link found in parent graph for id [...] slot [...] ref_videos.ref_video_0”。
// 这里在连接变化后按图里实际 link 状态重新对齐每个 autogrow 组：
//   1) 清除指向已不存在 link 的脏 .link
//   2) 把组内 link 向前补齐空洞（与官方收缩的 bubble 逻辑一致）
//   3) 删除组尾多余的空端口
// 该过程幂等，可安全重复执行。

function scheduleAutogrowReconcile(node) {
    if (!node || app.configuringGraph) return;
    // 官方收缩回调用一层 requestAnimationFrame 延迟，且可能再链式触发一层；
    // 用三层 rAF 确保跑在官方回调之后，看到的是最终稳定状态。
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            requestAnimationFrame(() => autogrowReconcile(node));
        });
    });
}

function autogrowReconcile(node) {
    if (!node || app.configuringGraph) return;
    const ag = node.comfyDynamic && node.comfyDynamic.autogrow;
    const graph = node.graph;
    if (!ag || !graph || !node.inputs) return;

    const resolveLink = (id) => {
        if (id == null) return null;
        if (typeof graph.getLink === "function") return graph.getLink(id) || null;
        if (graph.links && typeof graph.links.get === "function") return graph.links.get(id) || null;
        return null;
    };

    let changed = false;
    for (const groupName of Object.keys(ag)) {
        const cfg = ag[groupName];
        if (!cfg) continue;
        const min = typeof cfg.min === "number" ? cfg.min : 1;
        const stride = (cfg.inputSpecs && cfg.inputSpecs.length) || 1;

        // 按数组顺序收集本组输入槽下标
        const indices = [];
        for (let i = 0; i < node.inputs.length; i++) {
            const name = node.inputs[i] && node.inputs[i].name;
            if (typeof name === "string" && name.startsWith(groupName + ".")) {
                indices.push(i);
            }
        }
        if (indices.length === 0) continue;

        // 1) 清除脏 link（link id 已不在图中）
        for (const idx of indices) {
            const inp = node.inputs[idx];
            if (inp.link != null && !resolveLink(inp.link)) {
                inp.link = null;
                changed = true;
            }
        }

        // 2) 向前补齐空洞：每个 column 内把非空 link 压缩到前面
        for (let c = 0; c < stride; c++) {
            const colIdx = [];
            for (let p = c; p < indices.length; p += stride) colIdx.push(indices[p]);
            const links = colIdx.map((idx) => node.inputs[idx].link);
            const nonNull = links.filter((l) => l != null);
            for (let k = 0; k < colIdx.length; k++) {
                const inp = node.inputs[colIdx[k]];
                const want = k < nonNull.length ? nonNull[k] : null;
                if (inp.link !== want) {
                    inp.link = want;
                    changed = true;
                }
                if (want != null) {
                    const lk = resolveLink(want);
                    if (lk && lk.target_slot !== colIdx[k]) {
                        lk.target_slot = colIdx[k];
                        changed = true;
                    }
                }
            }
        }

        // 3) 删除组尾多余的空端口
        let highest = -1;
        for (let p = 0; p < indices.length; p++) {
            if (node.inputs[indices[p]].link != null) {
                highest = Math.max(highest, Math.floor(p / stride));
            }
        }
        const keepSlots = (Math.max(min, highest + 1) + 1) * stride;
        if (indices.length > keepSlots) {
            const removeIdx = indices.slice(keepSlots);
            for (let k = removeIdx.length - 1; k >= 0; k--) {
                node.removeInput(removeIdx[k]);
                changed = true;
            }
        }
    }

    if (changed) {
        if (typeof node.setSize === "function" && node.size) node.setSize(node.size);
        if (typeof node.setDirtyCanvas === "function") node.setDirtyCanvas(true, true);
    }
}

function getGraph(node) {
    return node.graph || (typeof app !== "undefined" ? app.graph : null);
}

function getLink(graph, linkId) {
    if (!graph || linkId == null) return null;
    if (graph.links && typeof graph.links.get === "function") return graph.links.get(linkId);
    return graph.links ? graph.links[linkId] : null;
}

function getNodeById(graph, id) {
    if (!graph || id == null) return null;
    if (typeof graph.getNodeById === "function") return graph.getNodeById(id);
    return graph._nodes_by_id ? graph._nodes_by_id[id] : null;
}

function traceOrigin(graph, linkId, visited) {
    if (linkId == null) return null;
    if (!visited) visited = new Set();
    if (visited.has(linkId)) return null;
    visited.add(linkId);
    const link = getLink(graph, linkId);
    if (!link) return null;
    const origin = getNodeById(graph, link.origin_id);
    if (!origin) return null;
    if ((origin.type === "Reroute" || origin.comfyClass === "Reroute") && origin.inputs && origin.inputs.length) {
        return traceOrigin(graph, origin.inputs[0].link, visited);
    }
    return { node: origin, slot: link.origin_slot };
}

function readOriginValue(node, slot) {
    if (!node || !node.widgets || !node.widgets.length) return undefined;
    // 1) 直接取输出槽位对应的 widget (primitive/常量节点)
    const w = node.widgets[slot];
    if (w && typeof w.value !== "undefined" && w.value !== null && w.value !== "") return w.value;
    // 2) 单输出 + 单 widget 的常量节点
    const outCount = node.outputs ? node.outputs.length : 0;
    if (outCount === 1 && node.widgets.length === 1) {
        const w0 = node.widgets[0];
        if (w0 && typeof w0.value !== "undefined" && w0.value !== null && w0.value !== "") return w0.value;
    }
    return undefined;
}

// 读取一个参数的实际值：未连接 -> 本地 widget；已连接 -> 追溯上游。
// 返回 { connected, value }，connected 且 value === undefined 表示连了上游但无法解析。
function resolveInputValue(node, name) {
    if (!node || !node.inputs) return { connected: false, value: undefined };
    const slotIdx = node.inputs.findIndex(i => i.name === name);
    const widget = node.widgets ? node.widgets.find(w => w.name === name) : null;
    if (slotIdx < 0 || node.inputs[slotIdx].link == null) {
        return { connected: false, value: widget ? widget.value : undefined };
    }
    const graph = getGraph(node);
    const origin = traceOrigin(graph, node.inputs[slotIdx].link);
    if (!origin) return { connected: true, value: undefined };
    return { connected: true, value: readOriginValue(origin.node, origin.slot) };
}

function readWidget(node, name, def) {
    const w = node.widgets ? node.widgets.find(w => w.name === name) : null;
    return w && w.value !== undefined ? w.value : def;
}

function snapToGrid(n) {
    if (n < 5) return Math.max(1, n);
    return 17 * Math.floor((n - 5) / 17) + 5;
}

function snapToGridUp(n) {
    if (n <= 5) return 5;
    return 17 * Math.floor((n + 11) / 17) + 5;
}

function snapToGridNearest(n) {
    if (n <= 5) return 5;
    const down = snapToGrid(n);
    const up = snapToGridUp(n);
    if (up - n <= n - down) return up;
    return down;
}

function computeChunksJS(totalFrames, chunkFrames, contextFrames) {
    let chunk = snapToGridNearest(chunkFrames);
    if (chunk < 5) chunk = 5;
    let context = Math.max(0, Math.min(contextFrames, chunk - 5));

    if (totalFrames <= chunk) {
        const actual = snapToGridUp(Math.max(5, totalFrames));
        return { segSizes: [actual], chunk, effContext: context };
    }

    const maxLast = Math.floor(chunk * 1.3);

    let n = Math.ceil(totalFrames / chunk);
    while (n > 1) {
        const cov = (n - 2) * chunk + maxLast - (n - 2) * context;
        if (cov >= totalFrames) {
            n -= 1;
        } else {
            break;
        }
    }

    let last;
    while (true) {
        const covered = n >= 2 ? (n - 1) * chunk - (n - 2) * context : 0;
        const needLast = totalFrames - covered + (n >= 2 ? context : 0);
        last = snapToGridUp(Math.max(5, needLast));
        if (last <= maxLast) break;
        n += 1;
    }

    let sizes = new Array(n - 1).fill(chunk);
    sizes.push(last);

    while (sizes.length >= 2 && sizes[sizes.length - 1] - context < 22) {
        const merged = snapToGridUp(sizes[sizes.length - 2] + sizes[sizes.length - 1] - context);
        if (merged > maxLast) break;
        sizes[sizes.length - 2] = merged;
        sizes.pop();
    }

    const minSeg = Math.min(...sizes);
    const maxContext = minSeg - 5;
    context = Math.max(5, Math.min(contextFrames, maxContext));

    return { segSizes: sizes, chunk, effContext: context };
}

function parseTagPattern(tagInput) {
    const s = tagInput.trim();
    let i = s.length - 1;
    while (i >= 0 && !/\d/.test(s[i])) i--;
    if (i < 0) return null;
    const numEnd = i + 1;
    while (i >= 0 && /\d/.test(s[i])) i--;
    const numStart = i + 1;
    return { prefix: s.slice(0, numStart), suffix: s.slice(numEnd) };
}

function countTagSegments(prompt, prefix, suffix) {
    const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(
        '^[ \\t]*' + esc(prefix) + '(\\d+)' + esc(suffix),
        'gm');
    const matches = prompt.match(re);
    return matches ? matches.length : 0;
}

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "H3.AutoContextSampler",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3Parameter") return;

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
    
            if ((totalR.connected && totalR.value === undefined) || (chunkR.connected && chunkR.value === undefined) || (ctxR.connected && ctxR.value === undefined)) {
                if (this.infoWidget) this.infoWidget.value = "已连接上游(无法预测)，以运行日志为准";
                return;
            }
    
            const totalFrames = parseInt(totalR.value !== undefined ? totalR.value : 362, 10);
            const chunkFramesInput = parseInt(chunkR.value !== undefined ? chunkR.value : 90, 10);
            const ctxFrames = parseInt(ctxR.value !== undefined ? ctxR.value : 22, 10);
    
            if (!Number.isFinite(totalFrames) || !Number.isFinite(chunkFramesInput) || !Number.isFinite(ctxFrames)) {
                if (this.infoWidget) this.infoWidget.value = "已连接上游(无法预测)，以运行日志为准";
                return;
            }
    
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
                    // 跳过文本编辑控件（如 Math Expression 的 expression）：
                    if (w.type === "customtext" || w.type === "text" || w.type === "string") continue;
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

function scheduleAutogrowReconcile(node) {
    if (!node || app.configuringGraph) return;
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

        const indices = [];
        for (let i = 0; i < node.inputs.length; i++) {
            const name = node.inputs[i] && node.inputs[i].name;
            if (typeof name === "string" && name.startsWith(groupName + ".")) {
                indices.push(i);
            }
        }
        if (indices.length === 0) continue;

        for (const idx of indices) {
            const inp = node.inputs[idx];
            if (inp.link != null && !resolveLink(inp.link)) {
                inp.link = null;
                changed = true;
            }
        }

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
    const hasValue = (v) => typeof v !== "undefined" && v !== null && v !== "";
    if (node.type === "PrimitiveNode") {
        const w = node.widgets[0];
        return (w && hasValue(w.value)) ? w.value : undefined;
    }

    const valueWidget = node.widgets.find(w => w && w.name === "value" && hasValue(w.value));
    if (valueWidget) return valueWidget.value;
    const isTextWidget = (w) => w && (w.type === "customtext" || w.type === "text" || w.type === "string");
    const valued = node.widgets.filter(w => w && hasValue(w.value) && !isTextWidget(w));
    if (valued.length === 1) {
        const v = valued[0].value;
        if (typeof v === "number" && Number.isFinite(v)) return v;
        if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) return Number(v);
        return undefined;
    }
    return undefined;
}


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
    if (!Number.isFinite(totalFrames)) totalFrames = 0;
    if (!Number.isFinite(chunkFrames)) chunkFrames = 0;
    if (!Number.isFinite(contextFrames)) contextFrames = 0;

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
        // NaN/Infinity 保险：NaN 参与比较恒为 false 会导致此循环永不退出卡死页面；
        // 有限数字下 needLast 每轮至少减 5 (chunk-context>=5)，必然收敛，无需帧数上限
        if (!Number.isFinite(n) || !Number.isFinite(last)) break;
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

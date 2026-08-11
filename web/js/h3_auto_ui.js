import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "H3.AutoContextSampler",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3AutoContextSampler") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            this.infoWidget = this.addWidget("text", "预计分段", "-", () => {}, { serialize: false });
            setTimeout(() => this._updateChunkInfo(), 50);
            return r;
        };

        const onPropertyChanged = nodeType.prototype.onPropertyChanged;
        nodeType.prototype.onPropertyChanged = function (property, value) {
            if (onPropertyChanged) onPropertyChanged.apply(this, arguments);
            setTimeout(() => this._updateChunkInfo(), 0);
        };

        nodeType.prototype._updateChunkInfo = function () {
            if (!this.widgets) return;
            const get = (name, def) => {
                const w = this.widgets.find(w => w.name === name);
                return w ? w.value : def;
            };
            const totalSec = get("total_seconds", 15);
            const chunkSec = get("chunk_seconds", 5);
            const fps = get("fps", 24);
            const totalFrames = Math.floor(totalSec * fps);
            const chunkFrames = Math.floor(chunkSec * fps);
            const numChunks = chunkFrames > 0 ? Math.ceil(totalFrames / chunkFrames) : 0;
            if (this.infoWidget) {
                this.infoWidget.value = `约 ${numChunks} 段 (每段${chunkFrames}帧)`;
            }
        };
    },
});

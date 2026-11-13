import { app } from "../../scripts/app.js";

/* ComfyUI's addDOMWidget owns positioning and resizing. SVGs are decoded
 * before DOM insertion; the SVG output socket passes through as it is. */
function makeTextPreview(node) {
    const field = document.createElement("textarea");
    field.readOnly = true;
    field.spellcheck = false;
    field.wrap = "off";
    field.placeholder = "SVG source appears here after execution";
    field.setAttribute("aria-label", "SVG source preview");
    field.style.cssText = [
        "box-sizing:border-box", "display:block", "width:100%", "height:270px",
        "resize:none", "overflow:auto", "padding:10px", "border:1px solid #666",
        "border-radius:6px", "background:#171717", "color:#eee",
        "font:12px/1.45 monospace"
    ].join(";");
    const widget = node.addDOMWidget("svg_text", "svg-text-preview", field, {
        hideOnZoom: false, getMinHeight: () => 270, getHeight: () => 270
    });
    widget.serialize = false;
    widget.options.serialize = false;
    node.__starvectorPreview = { kind: "text", field };
}

function makeImagePreview(node) {
    // Match ComfyUI Preview Image: the preview widget appears once media has
    // loaded.
    node.__starvectorPreview = {
        kind: "image", frame: null, widget: null,
        urls: [], pending: new Set(), generation: 0, disposed: false
    };
}

function releasePreview(state) {
    state.generation++;
    for (const url of [...state.urls, ...state.pending]) {
        URL.revokeObjectURL(url);
    }
    state.urls = [];
    state.pending.clear();
    if (state.frame) state.frame.replaceChildren();
}

function ensureImageWidget(node, state) {
    if (state.frame) return state.frame;

    // A DOM widget hosts the preview; it and its background appear only after
    // a decoded SVG is available.
    const frame = document.createElement("div");
    frame.setAttribute("aria-label", "StarVector SVG preview");
    frame.style.boxSizing = "border-box";
    frame.style.width = "100%";
    frame.style.height = "100%";
    frame.style.minWidth = "0";
    frame.style.minHeight = "0";
    frame.style.display = "grid";
    frame.style.overflow = "hidden";
    frame.style.alignContent = "stretch";

    const widget = node.addDOMWidget("svg_image", "svg-image-preview", frame, {
        hideOnZoom: false,
        getMinHeight: () => 190
    });
    widget.serialize = false;
    widget.options.serialize = false;
    const originalRemove = widget.onRemove;
    widget.onRemove = function (...args) {
        releasePreview(state);
        state.frame = null;
        state.widget = null;
        originalRemove?.apply(this, args);
    };
    state.frame = frame;
    state.widget = widget;

    // Reserve widget height on the first successful preview, keeping the
    // user's node width and letting the images scale.
    const minimumHeight = node.computeSize?.()?.[1] ?? 260;
    if ((node.size?.[1] ?? 0) < minimumHeight) {
        node.setSize([node.size?.[0] ?? 210, minimumHeight]);
    }
    return frame;
}

function fitImagesToNode(frame, images) {
    const columns = Math.ceil(Math.sqrt(images.length));
    const rows = Math.ceil(images.length / columns);
    frame.style.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
    frame.style.gridTemplateRows = `repeat(${rows}, minmax(0, 1fr))`;
    for (const image of images) {
        image.alt = "StarVector SVG";
        image.draggable = false;
        image.style.width = "100%";
        image.style.height = "100%";
        image.style.minWidth = "0";
        image.style.minHeight = "0";
        image.style.maxWidth = "100%";
        image.style.maxHeight = "100%";
        image.style.display = "block";
        image.style.objectFit = "contain";
    }
}

function setBackground(element, background) {
    const colors = {
        white: "#ffffff", black: "#000000",
        gray: "#808080", transparent: "transparent"
    };
    element.style.backgroundColor = colors[background] ?? "#ffffff";
    element.style.backgroundImage = background === "transparent"
        ? "conic-gradient(#bbb 25%, #fff 0 50%, #bbb 0 75%, #fff 0)"
        : "none";
    element.style.backgroundSize = background === "transparent" ? "24px 24px" : "";
}

function watchBackground(node, state) {
    // Repaint a shown preview as soon as the background widget changes.
    const widget = node.widgets?.find(w => w.name === "background");
    if (!widget) return;
    const originalCallback = widget.callback;
    widget.callback = function (...args) {
        originalCallback?.apply(this, args);
        if (state.frame && state.urls.length) {
            setBackground(state.frame, widget.value);
            node.graph?.setDirtyCanvas(true, true);
        }
    };
}

function showPreview(node, message) {
    const state = node.__starvectorPreview;
    if (!state) return;
    const documents = message?.svg;
    if (!Array.isArray(documents) || documents.some(x => typeof x !== "string")) return;

    if (state.kind === "text") {
        state.field.value = documents.map((source, index) => (
            documents.length === 1
                ? source
                : "<!-- SVG " + (index + 1) + " / " + documents.length + " -->\n" + source
        )).join("\n\n");
        state.field.scrollTop = 0;
        return;
    }

    if (state.disposed) return;
    const background = message.background?.[0] ?? "white";
    const generation = ++state.generation;
    const sources = documents.filter(source => /<svg(?=[\s/>])/i.test(source));

    const pending = sources.map(source => {
        const url = URL.createObjectURL(
            new Blob([source], { type: "image/svg+xml;charset=utf-8" })
        );
        state.pending.add(url);
        return new Promise(resolve => {
            // Attach only after the browser decodes the SVG successfully,
            // just like the normal Preview Image node.
            const image = new Image();
            image.onload = () => resolve({ image, url });
            image.onerror = () => resolve(null);
            image.src = url;
        }).then(result => {
            state.pending.delete(url);
            if (!result || generation !== state.generation) {
                URL.revokeObjectURL(url);
                return null;
            }
            return result;
        });
    });

    Promise.all(pending).then(results => {
        if (generation !== state.generation) return;
        const valid = results.filter(Boolean);
        for (const url of state.urls) URL.revokeObjectURL(url);
        state.urls = valid.map(result => result.url);

        if (valid.length) {
            // No placeholder image, background, or widget until decoding succeeds.
            const frame = ensureImageWidget(node, state);
            const images = valid.map(result => result.image);
            fitImagesToNode(frame, images);
            setBackground(frame, background);
            frame.replaceChildren(...images);
            node.graph?.setDirtyCanvas(true, true);
        } else if (state.frame) {
            state.frame.replaceChildren();
            state.frame.style.backgroundColor = "";
            state.frame.style.backgroundImage = "";
            state.frame.style.backgroundSize = "";
            node.graph?.setDirtyCanvas(true, true);
        }
    });
}

app.registerExtension({
    name: "ComfyUI.StarVectorGGUF.SVGPreviews",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const kind = {
            StarVectorGGUFTextPreview: "text",
            StarVectorGGUFPreviewSVG: "image"
        }[nodeData.name];
        if (!kind) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            originalCreated?.apply(this, args);
            if (kind === "text") makeTextPreview(this);
            else {
                // Clear ComfyUI's generic image state: the preview shows
                // executed SVGs only.
                this.imgs = undefined;
                makeImagePreview(this);
                watchBackground(this, this.__starvectorPreview);
            }
            if (kind === "text") {
                this.setSize([
                    Math.max(this.size?.[0] ?? 0, 400),
                    Math.max(this.size?.[1] ?? 0, 360)
                ]);
            }
            // Graphic preview: preserve ComfyUI's default initial node size.
            // The preview area is only added after the first successful SVG.
        };

        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            originalExecuted?.apply(this, [message]);
            showPreview(this, message);
        };

        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) {
            const state = this.__starvectorPreview;
            if (state?.kind === "image") {
                state.disposed = true;
                releasePreview(state);
            }
            originalRemoved?.apply(this, args);
        };
    }
});

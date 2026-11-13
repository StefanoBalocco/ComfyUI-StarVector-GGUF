"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(
    path.join(__dirname, "..", "js", "svg_previews.js"), "utf8"
).replace(/^import [^\n]+\n/gm, "");

function createNode({ autoLoad = true, widgets = [] } = {}) {
    let extension;
    let imagesConstructed = 0;
    let hostsCreated = 0;
    let nextUrl = 0;
    const revoked = [];
    const pendingImages = [];
    const blobsByUrl = new Map();
    const frame = {
        style: {}, children: [],
        setAttribute() {},
        replaceChildren(...children) { this.children = children; }
    };
    const host = {
        el: frame,
        get images() { return frame.children; }
    };
    class FakeImage {
        constructor() {
            imagesConstructed++;
            this.style = {};
            this.naturalWidth = 20;
            this.naturalHeight = 30;
            pendingImages.push(this);
        }
        set src(url) {
            this.url = url;
            if (autoLoad) Promise.resolve().then(() => this.finish());
        }
        finish() {
            if (blobsByUrl.get(this.url)?.data.join("").includes("INVALID")) {
                this.onerror?.();
            } else {
                this.onload?.();
            }
        }
    }
    vm.runInNewContext(source, {
        app: { registerExtension(value) { extension = value; } },
        document: {
            createElement(tag) {
                if (tag === "div") {
                    hostsCreated++;
                    return frame;
                }
                return { style: {}, setAttribute() {} };
            }
        },
        URL: {
            createObjectURL(blob) {
                const url = "blob:test-" + (++nextUrl);
                blobsByUrl.set(url, blob);
                return url;
            },
            revokeObjectURL(url) { revoked.push(url); }
        },
        Image: FakeImage,
        Blob: class FakeBlob {
            constructor(data, opts) { this.data = data; this.opts = opts; }
        }
    });
    const nodeType = { prototype: {} };
    extension.beforeRegisterNodeDef(nodeType, { name: "StarVectorGGUFPreviewSVG" });
    const node = Object.create(nodeType.prototype);
    node.size = [240, 140];
    node.setSize = function(size) { this.size = size; };
    node.addDOMWidget = function(name, kind, el, options) {
        this.widgets.push({ name, kind, el, options });
        const widget = this.widgets.at(-1);
        this.widget = widget;
        return widget;
    };
    node.widgets = [...widgets];
    node.graph = { setDirtyCanvas() {} };
    node.onNodeCreated();
    return {
        node, host, revoked, pendingImages,
        hostsCreated: () => hostsCreated,
        imagesConstructed: () => imagesConstructed,
    };
}

function settle() {
    return new Promise(resolve => setImmediate(resolve));
}

test("new node has no preview widget, broken image, white background, or forced size", async () => {
    const h = createNode();
    assert.equal(h.node.widgets.length, 0);
    assert.equal(h.hostsCreated(), 0);
    assert.equal(h.imagesConstructed(), 0);
    assert.equal(h.host.el.style.backgroundColor, undefined);
    assert.equal(h.node.size[0], 240);
    assert.equal(h.node.size[1], 140);

    h.node.onExecuted({ svg: [], background: ["white"] });
    await settle();
    h.node.onExecuted({ svg: ["not SVG"], background: ["white"] });
    await settle();
    assert.equal(h.node.widgets.length, 0);
    assert.equal(h.hostsCreated(), 0);
    assert.equal(h.imagesConstructed(), 0);
});

test("first valid decoded SVG creates exactly one native ComfyUI preview", async () => {
    const h = createNode();
    h.node.onExecuted({
        svg: ['<svg xmlns="http://www.w3.org/2000/svg"/>'],
        background: ["white"]
    });
    assert.equal(h.hostsCreated(), 0);
    assert.equal(h.node.widgets.length, 0);
    await settle();
    assert.equal(h.hostsCreated(), 1);
    assert.equal(h.node.widgets.length, 1);
    assert.equal(h.node.widget.options.getMinHeight(), 190);
    assert.equal(h.node.widget.options.hideOnZoom, false);
    assert.equal(h.node.widget.el, h.host.el);
    assert.equal(h.host.images.length, 1);
    assert.equal(h.host.el.children.length, 1);
    assert.equal(h.host.el.style.backgroundColor, "#ffffff");
    assert.ok(h.node.size[1] >= 260);
    assert.equal(h.node.size[0], 240);
    assert.equal(h.node.widget.options.getMinHeight(), 190);
    assert.equal(h.host.el.style.display, "grid");
    assert.equal(h.host.images[0].style.objectFit, "contain");
    assert.equal(h.host.images[0].style.width, "100%");
    assert.equal(h.host.images[0].style.height, "100%");
});

test("SVG batches and background changes reuse the existing preview host", async () => {
    const h = createNode();
    h.node.onExecuted({ svg: ["<svg/>"], background: ["black"] });
    await settle();
    assert.equal(h.hostsCreated(), 1);
    assert.equal(h.host.el.style.backgroundColor, "#000000");

    h.node.onExecuted({
        svg: ["<svg/>", "<svg/>"],
        background: ["transparent"]
    });
    await settle();
    assert.equal(h.hostsCreated(), 1);
    assert.equal(h.host.images.length, 2);
    assert.equal(h.host.el.children.length, 2);
    assert.equal(h.host.el.style.backgroundColor, "transparent");
    assert.equal(h.host.el.style.gridTemplateColumns, "repeat(2, minmax(0, 1fr))");
    assert.equal(h.host.el.style.gridTemplateRows, "repeat(1, minmax(0, 1fr))");
    assert.ok(h.host.images.every(img => img.style.objectFit === "contain"));
    assert.ok(h.revoked.includes("blob:test-1"));
});

test("background widget change repaints the shown preview at once", async () => {
    const background = { name: "background", value: "white" };
    const h = createNode({ widgets: [background] });
    background.value = "black";
    background.callback("black");
    assert.equal(h.hostsCreated(), 0);

    h.node.onExecuted({ svg: ["<svg/>"], background: ["black"] });
    await settle();
    background.value = "gray";
    background.callback("gray");
    assert.equal(h.host.el.style.backgroundColor, "#808080");
    assert.equal(h.host.images.length, 1);
});

test("invalid SVG never creates a broken-image preview", async () => {
    const h = createNode();
    h.node.onExecuted({ svg: ["<svg INVALID>"], background: ["white"] });
    await settle();
    assert.equal(h.imagesConstructed(), 1);
    assert.equal(h.hostsCreated(), 0);
    assert.equal(h.node.widgets.length, 0);
    assert.equal(h.host.el.style.backgroundColor, undefined);
    assert.ok(h.revoked.includes("blob:test-1"));
});

test("removed node never creates a preview from a late image load", async () => {
    const h = createNode({ autoLoad: false });
    h.node.onExecuted({ svg: ["<svg/>"], background: ["white"] });
    assert.equal(h.pendingImages.length, 1);
    assert.equal(h.hostsCreated(), 0);
    h.node.onRemoved();
    h.pendingImages[0].finish();
    await settle();
    assert.equal(h.hostsCreated(), 0);
    assert.equal(h.node.widgets.length, 0);
    assert.ok(h.revoked.includes("blob:test-1"));
});

test("removing a populated node releases all preview object URLs", async () => {
    const h = createNode();
    h.node.onExecuted({ svg: ["<svg/>", "<svg/>"] });
    await settle();
    assert.equal(h.host.images.length, 2);
    h.node.onRemoved();
    assert.equal(h.host.el.children.length, 0);
    assert.ok(h.revoked.includes("blob:test-1"));
    assert.ok(h.revoked.includes("blob:test-2"));
});

test("preview scales with user-controlled node size and handles a square batch", async () => {
    const h = createNode();
    h.node.onExecuted({ svg: ["<svg/>", "<svg/>", "<svg/>", "<svg/>"] });
    await settle();
    assert.equal(h.host.el.style.gridTemplateColumns, "repeat(2, minmax(0, 1fr))");
    assert.equal(h.host.el.style.gridTemplateRows, "repeat(2, minmax(0, 1fr))");
    for (const image of h.host.images) {
        assert.equal(image.style.width, "100%");
        assert.equal(image.style.height, "100%");
        assert.equal(image.style.objectFit, "contain");
    }
    h.node.setSize([540, 620]);
    assert.equal(h.node.size[0], 540);
    assert.equal(h.node.size[1], 620);
    // Layout adapts through CSS, with no fixed pixel width or height.
    assert.equal(h.host.el.style.width, "100%");
    assert.equal(h.host.el.style.height, "100%");
    assert.equal(h.host.el.style.overflow, "hidden");
});

test("no node widget or background until the first successful SVG decode", async () => {
    const h = createNode({ autoLoad: false });
    h.node.onExecuted({ svg: ["<svg/>"], background: ["black"] });
    assert.equal(h.node.widgets.length, 0);
    assert.equal(h.hostsCreated(), 0);
    h.pendingImages[0].finish();
    await settle();
    assert.equal(h.node.widgets.length, 1);
    assert.equal(h.host.el.style.backgroundColor, "#000000");
    assert.equal(h.host.images.length, 1);
});

test("failed replacement clears prior preview and background", async () => {
    const h = createNode();
    h.node.onExecuted({ svg: ["<svg/>"], background: ["black"] });
    await settle();
    assert.equal(h.host.images.length, 1);
    h.node.onExecuted({ svg: ["<svg INVALID>"], background: ["white"] });
    await settle();
    assert.equal(h.host.images.length, 0);
    assert.equal(h.host.el.style.backgroundColor, "");
    assert.equal(h.host.el.style.overflow, "hidden");
    assert.ok(h.revoked.includes("blob:test-1"));
});

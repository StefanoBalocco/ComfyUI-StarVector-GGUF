# ComfyUI-StarVector-GGUF

Convert images to SVG in ComfyUI with StarVector 1B (CLIP + GPTBigCode) and 8B (SigLIP + StarCoder2) models stored as local GGUF files. The models run on a native PyTorch and Triton runtime. When StarVector leaves an SVG incomplete, the node traces the input image with VTracer or repairs the markup, and a LIVE-style refine fits the SVG's shapes to the input image by gradient descent.

With the same weights, the native decoders and the SigLIP encoder produce the same outputs as StarVector's reference implementation. On an RTX 3060 with real GGUF weights, the 1B and 8B decoders (Q4_0, Q5_0, Q8_0) agree with StarVector's FP16/BF16 checkpoints on 99-100% of top-1 tokens over thousands of forced SVG tokens.

## Install

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/StefanoBalocco/ComfyUI-StarVector-GGUF.git
# Install with ComfyUI's own Python interpreter:
python -m pip install -r ComfyUI-StarVector-GGUF/requirements.txt
```

The extension runs on ComfyUI's PyTorch. With Triton installed for the same CUDA stack, the decoder uses Triton kernels for its quantized weights. The browser renders the SVG previews.

## Models

ComfyUI finds the models through its `folder_paths` registry: in `ComfyUI/models/starvector/` and, first, in the `starvector` paths of `extra_model_paths.yaml`. `base_path` applies to the categories listed in its own section, so list `starvector` there:

```yaml
shared_models:
  base_path: /data/comfy-models
  starvector: models/starvector
```

ComfyUI resolves this to `/data/comfy-models/models/starvector`. A list of paths works in the usual multi-line YAML form.

A StarVector GGUF holds:

- the vision encoder, the projector and the decoder;
- the model configuration, in `starvector.config_json`, so 1B and 8B files share a directory;
- the tokenizer vocabulary and configuration, in `starvector.embedded_file.*`. The loader checks that the tokenizer matches the decoder vocabulary and raises a `ValueError` on missing or invalid tokenizer metadata.

Ready-made 1B and 8B GGUF files are on Hugging Face (links under Technologies). The loader indexes the GGUF once and builds the model from its memory-mapped tensors. Log lines prefixed `StarVector:` report the loading, encoding and generation times.

## Nodes

The four nodes appear under `StarVector/GGUF`.

1. **Load StarVector GGUF** loads a local GGUF and outputs a `STARVECTOR` socket.
2. **Image to SVG** converts a ComfyUI `IMAGE` batch into a ComfyUI SVG batch, one document per input image, in order.
3. **SVG Text Preview** shows the XML read-only and passes the SVG batch through.
4. **SVG Image Preview** renders the SVGs with ComfyUI's image preview layout, which follows the node's size and arranges a batch as a grid. The `background` option (`white` by default; also `black`, `gray`, `transparent`) applies to the preview as soon as it changes. The preview appears with the first SVG the browser decodes and shows the valid SVGs of the batch. The node passes the SVG batch through unchanged.

ComfyUI's **Save SVG** node saves the results, with filename masks, subfolders, counters and metadata.

```text
Load StarVector GGUF ─┐
                      ├─> Image to SVG ─> SVG Text Preview ─> SVG Image Preview ─> Save SVG (ComfyUI)
Load Image ───────────┘                         SVG                   SVG             file.svg
```

A batch runs in two stages. The vision stage encodes the images one by one and holds the compact embeddings in system RAM; the decoder stage then generates the SVGs one by one. ComfyUI loads each stage when it runs and can evict the other.

The progress bar advances with each encoded image, decoded token and refine step. After generation, the log reports the token count and tokens per second.

## Options

Image to SVG samples as StarVector's reference implementation does: `do_sample=True`, `top_k=50`, `early_stopping=True`, `repetition_penalty=3.1` and `length_penalty=-1`. With the same model and seed, it returns the same tokens. For each input image, the node returns the best SVG.

- `max_new_tokens` (default 4096): upper limit on new tokens, capped by the model's remaining context.
- `temperature` (default 1.5): randomness of sampling, above zero.
- `top_p` (default 0.9): nucleus sampling threshold.
- `num_beams` (default 2, range 1-8): 1 samples one sequence; 2-8 use beam sampling, whose KV-cache memory and computation grow with each beam.
- `seed` (default 0): makes runs reproducible; the calling process keeps its RNG state.
- `loop_guard` (default on): on inputs outside its training data, such as shaded illustrations, StarVector can repeat one element until `max_new_tokens`. The guard stops generation when the last 64 tokens have already occurred 3 times and passes the incomplete SVG to `on_failure`.
- `on_failure` (default `vtracer`): the action when the SVG is incomplete (loop guard, `max_new_tokens`) or malformed.
  - `none` raises an error.
  - `recover` repairs the partial markup.
  - `vtracer` traces the input image with VTracer, a color tracer that takes about 0.1 s and handles shaded illustrations.
- `refine_steps` (default 100): the gradient steps of each refine round; 0 turns refine off. See Refine.
- `refine_paths` (default on): adds paths where the SVG still differs most from the image. See Refine.

The node validates the generated XML before it reaches the SVG socket. It logs every incomplete or malformed output and saves it unmodified in ComfyUI's temporary directory as `starvector-invalid-*.svg.txt`.

`recover` follows StarVector's reference implementation: Beautiful Soup reconstruction, CairoSVG `svg2svg()` normalization, XML validation. When reconstruction fails, the node returns an empty `<svg></svg>` and logs an error. Repair can drop an unfinished attribute or path, and CairoSVG can turn shapes into paths. CairoSVG runs with `unsafe=False`, which blocks external files and network resources.

## Refine

Refine fits the SVG to the input image, as LIVE (Ma et al., CVPR 2022) does. CairoSVG resolves the SVG (CSS, `<use>`, transforms, arcs, clip paths) into cubic Bézier paths. A differentiable rasterizer renders them at 512 px on the longest side: Triton kernels on a CUDA GPU, PyTorch on the CPU and other devices, with the same coverage and gradients. Adam moves the control points and adjusts fill and stroke colors and stroke widths to minimize the squared error against the input image composited on white. Straight segments (polygons, rectangles, `L`, `H`, `V`) stay straight: only their endpoints move, and they are written back as `L`.

`refine_steps` sets the steps of each fit round. The first round fits the existing shapes. With `refine_paths`, each added path starts a further round. As in LIVE, each path goes on top, starting from the largest connected region of one quantized image color that the SVG still renders wrong, traced with VTracer. Paths keep coming while such a region of at least 16 pixels exists, up to 16 paths: about 10 on a shaded illustration, 0-1 on a flat icon.

Refine applies to complete StarVector SVGs and to VTracer's traces. When refining, VTracer fits fewer, longer curves (`corner_threshold=90`, `length_threshold=8`, `splice_threshold=60`): the trace is about 20% smaller and, after refining, as accurate as the default tracing refined.

The refined SVG is flat: paths in 512 px units, with transforms and CSS resolved and metadata dropped. Strokes keep miter joins (with the SVG miter limit), round joins, and butt, square and round caps; bevel joins become round. Documents with gradients, patterns, text, images, masks, filters or dashed strokes come back unchanged, with a warning.

On six flat icons, 100 steps leave 1-1.5% of the squared error of the StarVector or VTracer SVG, and 300 steps leave 0.5-0.8%, as much as 1000 steps. Added paths pay off where the SVG misses detail: on VTracer's trace of a shaded cinnamon illustration, 100 steps alone leave 65% of the error, and the 10 added paths leave 9%.

With the defaults (100 steps, `refine_paths` on), refine takes 0.4-0.9 s per flat icon and 8 s for the cinnamon on an RTX 3060, and 2.5-3 s and 3 minutes on 5 CPU threads. Set `refine_steps` to 300 for a closer fit, or turn `refine_paths` off for a single round.

## Attention

Every attention operation goes through ComfyUI's `optimized_attention` with `low_precision_attention=False`, so the ComfyUI installation and launch flags choose the backend (FlashAttention, xFormers, PyTorch SDPA or a memory-optimized fallback), and ComfyUI picks a compatible backend for each shape, dtype and mask.

With `--use-sage-attention`, StarVector runs on PyTorch attention while other models use SageAttention: SageAttention's INT8 Q/K raised the 1B image-prefix error from about 1% to 16-19% and the decoder KL from 0.0001 to 0.01-0.03 against the FP16 model.

The CLIP/SigLIP encoders and the decoder prefill call `optimized_attention` directly. Cached decode steps run inside CUDA graphs and attend the fixed KV cache with PyTorch batched matrix products, as ComfyUI's own LLM decoding does.

## Creating GGUF files

`tools/starvector_safetensors_to_gguf.py` converts a StarVector `starvector-{1b,8b}-im2svg` Safetensors directory into a self-contained GGUF holding the weights, `starvector.config_json` and the tokenizer assets. It runs on the CPU with `gguf`, `safetensors`, `numpy`, `torch` and `tokenizers`.

```sh
python tools/starvector_safetensors_to_gguf.py --model-dir starvector-1b-im2svg \
  --output starvector-1b-im2svg-full-q4_0.gguf --family 1b --quantization q4_0
```

`--quantization` accepts `q4_0`, `q5_0` and `q8_0` (default `q8_0`), the three types the runtime loads.

Normalization parameters, biases and tensors whose width is not block-aligned keep their F16/F32/BF16 type. Below 8 bits, the vision encoder, projection adapter and token/position embeddings are exported as Q8_0, because Q4_0 distorted the 1B image prefix by 15-20%. The token embedding is also the tied output head.

## Tools

Run these from the extension directory with ComfyUI's Python.

- `tools/starvector_safetensors_to_gguf.py` creates GGUF files (see above).
- `tools/compare_original_1b.py` measures how far 1B GGUF files are from StarVector's FP16 checkpoint. It loads the checkpoint's Safetensors at full precision into this runtime, samples an SVG for an image, and teacher-forces every GGUF through the same tokens. It reports the image-prefix error, top-1 agreement and the KL divergence of the next-token distributions.

  ```sh
  python tools/compare_original_1b.py --original /path/to/starvector-1b-im2svg \
    --gguf 1b-im2svg-full-q8_0.gguf 1b-im2svg-full-q4_0.gguf --image icon.png
  ```
- `tools/tune_kernels.py` grid-searches the production SoA GEMV launch parameters `(bn, blocks, num_warps)` for Q4_0/Q5_0/Q8_0 and 1/2 decode rows on the 1B/8B matrix shapes, and compares the best configuration with `quantization.SOA_GEMV_CONFIG`. It takes about an hour on an RTX 3060, mostly Triton compilation.

## Implementation

- **GGUF Q4_0, Q5_0, Q8_0:** model construction rearranges every rank-2 linear and embedding weight on the CPU into SoA form (quantized values and FP16 scale bits), and ComfyUI transfers and offloads these two parameter arrays. For FP16 inference with 1 or 2 rows, a Triton SoA GEMV runs with fused bias; `quantization.SOA_GEMV_CONFIG` holds its launch parameters per quantization and row count (tuned by `tools/tune_kernels.py`). Prefill and vision (more rows) use a Triton SoA GEMM on the same arrays. On the CPU, in other dtypes or without Triton, the layers dequantize the SoA weights in bounded slices for PyTorch linear operations. Embeddings read the SoA layout directly, and the tied `lm_head` shares both weight arrays with its embedding.
- **F16, F32, BF16:** these tensors load directly; other types raise an error.
- **Native runtime:** CLIP, SigLIP, GPTBigCode (1B), StarCoder2 (8B) and the sampling loop are native PyTorch on the GGUF layers. The `tokenizers` library loads the embedded `tokenizer.json`.
- **Decoders:** each generation allocates a zeroed KV cache for the prompt and `max_new_tokens`, prefills once, and shares the prefill across beams. Decode steps use one CUDA graph per power-of-two cache span (at least 512 slots). Inside the graphs, Triton linears read and write static buffers that the eager warm-up of each span allocates, because Triton rejects memory allocated during capture with `cudaMallocAsync`, ComfyUI's default allocator. With 2 beams on an RTX 3060, the 1B decodes about 140-155 tokens/s (Q8_0/Q5_0/Q4_0) and the 8B about 30-40 tokens/s.
- **8B architecture:** the 8B decoder implements the StarCoder2 architecture. StarVector's config lacks the RoPE theta and the sliding window, so the runtime uses StarCoder2-7B's values (1e6 and 4096).
- **SVG refinement:** `svg_scene.py` records what CairoSVG fills, strokes and clips as device-space cubic Béziers. `svg_refine.py` flattens them with one constant sparse matrix. A Triton kernel computes, per layer and pixel, the box-filtered coverage from the distance to the outline and the winding number, over the tiles inside each layer's bounding box. A stroke is the union of the bands around its edges, cut near butt or square open ends, plus a quadrilateral per miter-joined corner from a second Triton kernel pair. Another kernel composites the layers, and gradients flow back through both. On the CPU and other devices, PyTorch computes the same values: the winding number from the edge crossings of each pixel row, and distances for the pixels within reach of each edge, where coverage and gradients change. CUDA devices with Triton use the kernels, 3-6 times faster than the PyTorch path.
- **Runtime layout:** `runtime_shared.py` holds the GGUF tensor store, the cached CUDA-graph decoding and the image-to-SVG generation shared by both architectures. `runtime_1b.py` holds the CLIP/GPTBigCode code; `runtime_8b.py` holds the SigLIP/StarCoder2 code.
- **ComfyUI VRAM:** separate ModelPatchers for the encoder/projector and the decoder allow stage-wise eviction.

## Tests

Run the offline `unittest` suite locally:

```sh
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p test_quantization.py   # one file
```

CI runs on Python 3.12. It syntax-checks every Python file, installs CPU PyTorch and the reference implementation, and runs small CPU tests of:

- the native GPTBigCode, StarCoder2 (including its sliding window) and SigLIP against the reference implementation with the same weights (prefill, cached decode, beam reordering);
- the native sampling against the reference sampling;
- quantization, generation, nodes, SVG handling and refine.

Locally, the refine tests also run on CUDA, where they check that the Triton kernels and the PyTorch rasterizer agree.

For a smoke test with a real checkpoint, connect the loader, a raster image and Image to SVG. A missing-tensor-name error points to a mismatch between the GGUF export, its configuration and its embedded tokenizer.

## License

BSD 3-Clause.

## Technologies

- [StarVector](https://github.com/joanrod/star-vector): the image-to-SVG models, as GGUF files for this extension: [1B](https://huggingface.co/StefanoBalocco/starvector-1b-im2svg-GGUF) and [8B](https://huggingface.co/StefanoBalocco/starvector-8b-im2svg-GGUF).
- [LIVE](https://ma-xu.github.io/LIVE/): the method behind refine and its added paths.
- [VTracer](https://www.visioncortex.org/vtracer/): fallback tracing and the outlines of added paths.

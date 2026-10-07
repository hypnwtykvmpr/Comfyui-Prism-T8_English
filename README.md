# Prism T8 English · ComfyUI

[Tencent Prism](https://github.com/Tencent-Hunyuan/Prism) native video and audio joint generation node. Retains the official dual-tower model, cross-modal bridge, and paired scheduler, providing seven independent component loaders, INT8 ConvRot weights, and a complete ComfyUI canvas workflow.

[Model download](https://huggingface.co/t8star/Prism-Comfy/tree/main) · [Canvas workflow](examples/README.md) · [Real 480p sample](https://github.com/T8mars/Comfyui-Prism-T8/blob/main/examples/sample-480p.mp4)

This is an English-language fork of [T8mars/Comfyui-Prism-T8](https://github.com/T8mars/Comfyui-Prism-T8). Node IDs and component formats are unchanged. The default negative prompt is translated into English; that changes model conditioning and is not output-equivalent to the upstream Chinese prompt. Verification results below are the upstream author's reports, not a qualification of this fork.

## Installation

The upstream project was submitted to Comfy Registry (`t8star/prism-t8`). This fork is not that registry release. For manual installation:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hypnwtykvmpr/Comfyui-Prism-T8_English.git
cd Comfyui-Prism-T8_English
```

Install dependencies using the Python that runs ComfyUI, then restart:

```bash
uv pip install --python /path/to/ComfyUI/python -r requirements.txt
```

Requires Python 3.10+, NVIDIA CUDA environment with BF16 support; keep ComfyUI's existing CUDA PyTorch. Saving MP4 with audio requires **ffmpeg** in PATH. Default SDPA does not need Triton; enabling native BSA additionally requires a Triton matched to PyTorch, and on Windows use `triton-windows`.

## Models

Install either this fork or upstream, not both: node identifiers are intentionally
the same so existing workflows continue to work.

Download all **seven files** from [t8star/Prism-Comfy](https://huggingface.co/t8star/Prism-Comfy) into the corresponding ComfyUI model directories. Full alpha model is about **38.13 GiB**.

| File | Directory |
| --- | --- |
| `prism_alpha_video_dit_int8_convrot.safetensors` | `models/diffusion_models/` |
| `prism_alpha_video_dit_2_int8_convrot.safetensors` | `models/diffusion_models/` |
| `prism_alpha_audio_dit_int8_convrot.safetensors` | `models/diffusion_models/` |
| `prism_alpha_dual_tower_bridge_int8_convrot.safetensors` | `models/diffusion_models/` |
| `prism_alpha_text_encoder_int8_convrot.safetensors` | `models/text_encoders/` |
| `prism_alpha_video_vae_bf16.safetensors` | `models/vae/` |
| `prism_alpha_audio_vae_bf16.safetensors` | `models/vae/` |

Text encoder is **UMT5**, video denoiser is dual **DiT**. Configurations and tokenizer are embedded in standalone files; use this plugin's loaders, no Diffusers weight directory needed. The `diffusers` library is only a code dependency for native model classes.

ConvRot quantization is used for block Linear; embedding, norm, time projection, output head, and VAE retain floating-point precision. Audio VAE is saved in BF16 and executed in FP32. All seven files must come from the same bundle; loaders check components, shapes, integrity, and quantization tags.

## Workflows

Download the [full workflow package](examples/README.md) and drag the JSON into the ComfyUI canvas. Place the included `prism_official_case5.png` into ComfyUI `input/`, or upload your own reference image in **Load Image**, then select the seven model files and run.

| Workflow | Purpose |
| --- | --- |
| [01 · I2VA](examples/01_native_i2va.json) | Recommended starting point: portable INT8 + SDPA, 848×480, 49 frames, 50 steps |
| [02 · Kitchen + BSA](examples/02_native_i2va_kitchen_bsa.json) | W8A8, native video/v2a BSA and IVPQ |
| [03 · White-frame T2VA](examples/03_native_t2va_white_reference.json) | Official white first-frame conditional experimental mode |
| [04 · 720p](examples/04_native_i2va_720p.json) | 1280×720, 205 frames, VAE tiling parameter presets |
| [05 · Kitchen control](examples/05_native_i2va_validation.json) | Kitchen INT8 + dense SDPA |

Each is in **canvas format** including node positions, groups, parameters, and connections, outputting PNG frames, 48 kHz FLAC, H.264/AAC MP4, and canvas video preview. MP4 retains all video frames, shorter audio tracks are padded with silence, longer tracks trimmed to video end. Detailed import instructions in [examples/README.md](examples/README.md).

Supports separate video/audio prompts, `<music>` / `<sfx>` / `<speech>` tags, CFG, seed, visual/audio shift, block or whole-component CPU offload, and native sparse attention parameters. Advanced parameters in [sparse_options.json](examples/sparse_options.json). Resolution is a multiple of 16; frames at least 5, satisfying `(frames-1)%4==0`. BSA's 3D block axes are powers of 2, K block at least 16 tokens; v2a audio block is a power of 2 not less than 64.

## Running and Verification

### Automatic memory admission

The sampler and component loader check system commit headroom before materializing
weights. On Windows this uses `GetPerformanceInfo` (commit limit minus committed
pages), not the free-RAM display. Insufficient or unavailable measurements raise
`MemoryError` before loading. A conservative estimate includes weight storage,
temporary conversions and output buffers, with a 16 GiB system reserve.

GPU transfers also check current CUDA free memory, transfer size, temporary INT8
dequantization and estimated inference workspace, retaining at least 4 GiB or 10%
of total VRAM, whichever is larger. Block-offload transfers recheck as they run.
The node does not unload unrelated ComfyUI models to make space. It refuses an
insufficient load rather than retrying automatically or changing user settings.

These are conservative admission estimates, not measured maximums or an OS-level
reservation. Another process can allocate memory after a check. Coordinate heavy
jobs; CUDA OOM remains possible and is not a successful generation. This fork has
not yet been locally GPU-qualified. Downloading weights is independent of these checks.

On Windows, the MP4 encoder is launched with `CREATE_NO_WINDOW` without conflicting
detached-console flags. Encoder failures and interruption still propagate; owned
temporary media files are cleaned by the existing save path.

Default `portable` is W8A16 rotation and temporary dequantization path; `kitchen` uses `comfy_kitchen.int8_linear` for dynamic W8A8. Block offload can reduce memory usage, speed affected by CPU memory and PCIe; INT8 does not reduce high-resolution activation footprint.

A real alpha INT8 sample has completed **848×480, 49 frames, 50 steps** generation and full-frame visual inspection, with complete decoding of audio and video tracks; audio not yet listened. On RTX 5090 Laptop 24 GB with block offload configuration, took about 47 minutes, PyTorch peak allocated memory about 7.76 GiB. There is slight composition drift and soft fine texture; quantization does not guarantee losslessness.

Five canvases have been actually imported and saved, regression tests all passed. Full 480p samples for 02/03/05, 720p long video, beta, and multi-GPU have not yet been verified with real samples; 320×192 sample quality is poor, recommend using 01's default settings first.

## Self-conversion

```bash
uv run --no-project --python /path/to/ComfyUI/python scripts/download_models.py --output checkpoints/official --variant alpha
uv run --no-project --python /path/to/ComfyUI/python scripts/convert_models.py --base checkpoints/official/pretrained_models/MOVA-360p --preview checkpoints/official/preview_alpha/diffusion_pytorch_model.safetensors --output models/standalone --variant alpha --device cuda:0
```

Conversion output is also auto-discovered by the plugin. Source weights about 72.35 GiB, conversion requires additional space for the final model and temporary space for the largest component. Supports `--variant beta`, `--dry-run`, and `--components`. When keeping source weights unchanged, can add `--resume` to validate and reuse completed files; `scripts/prepare_models.py` will verify existing components, restore missing manifest, and continue conversion. With a complete seven components, the preparation script does not need source weights or network. Default does not overwrite files, use a new output directory when switching recipes. Quantization recipes and file checksums are in the Hugging Face model repository.

## Source and License

### Incremental upstream updates

This fork keeps one English-and-runtime-fixes commit above its recorded upstream
base (or zero when there is no fork delta). Existing translations are retained;
upstream updates are not a request to translate the entire repository again.

For maintainers, start from a clean working tree and fetch `upstream`. Record the
old fork tip and its parent before changing anything. Compare that parent against
the new upstream commit using `git diff --unified=0 OLD_BASE NEW_UPSTREAM`.
Rebase the single overlay with `git rebase --onto NEW_UPSTREAM OLD_BASE main`.
Translate only new or changed user-facing source text from that upstream diff;
reuse unchanged English text, including moved strings. Resolve overlapping code
changes explicitly, preserving the memory-admission and windowless-launch fixes.
Rebuild the workflow ZIP only when its member files change.

Amend the single overlay after testing; do not accumulate translation commits or
merge commits. Keep the previous tip as a local recovery ref. Verify
`git rev-list --left-right --count NEW_UPSTREAM...main` reports `0 1` (or `0 0`).
When publishing a rewritten overlay, use an explicit
`--force-with-lease=refs/heads/main:OLD_REMOTE_TIP`, never an unconditional force
push. A changed remote tip means stop publishing and inspect the concurrent work.
New translation/API use and publication remain explicit maintenance actions;
there is no background translator or updater. Local update plans, receipts and
translation tooling stay excluded from the distributable node package.

Native source pinned to Tencent Prism [`883e90a5`](https://github.com/Tencent-Hunyuan/Prism/tree/883e90a5c90dc8b7044c65eba0bb64e9342cb46a), changes in [NATIVE_CHANGES.md](NATIVE_CHANGES.md). Original [LICENSE](LICENSE) and third-party attributions retained: Prism uses MIT, third-party components follow their respective licenses. This project is a community ComfyUI integration.

## T8star

[Bilibili](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [API](https://api.seedance.nz/sign-up?aff=5f4w) · [Free gallery](https://www.openzhenzhen.com) · [Online AI applications](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI bundle](https://pan.quark.cn/s/264edb7e36bd) · [Hugging Face](https://huggingface.co/t8star)

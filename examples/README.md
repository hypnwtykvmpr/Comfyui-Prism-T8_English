# Prism Full Canvas Workflows

These files are ComfyUI `version: 0.4` canvas JSON, containing node positions, connections, and parameters, which can be dragged into the canvas or opened from the workflow menu.

| File | Purpose |
| --- | --- |
| `01_native_i2va.json` | Generate video and audio jointly from reference image, INT8 portable, dense SDPA; 848×480 actual sample visuals reviewed |
| `02_native_i2va_kitchen_bsa.json` | Kitchen INT8, native video/v2a BSA, IVPQ dynamic blocking |
| `03_native_t2va_white_reference.json` | Official white-image condition text generation path, first frame is white |
| `04_native_i2va_720p.json` | 1280×720, 205 frames, 50 steps, VAE tiling presets; this configuration not yet qualified by the upstream author |
| `05_native_i2va_validation.json` | Kitchen INT8, dense SDPA, same-seed control |

1. Install the plugin and dependencies per the project root README; keep the seven standalone model files from the same conversion bundle.
2. Copy `prism_official_case5.png` to ComfyUI `input/`, or upload your own reference image in `Load Image`. The included image is from the pinned Tencent Prism official example, unmodified.
3. Import the desired JSON, check the seven model dropdowns, then click "Run". 01/02/03/05 default to 848×480, 49 frames, 24 fps, 50 steps, seed fixed at 42.
4. Full output includes frame-by-frame PNG (with embedded canvas workflow), 48 kHz FLAC, H.264/AAC MP4, and canvas video preview. Output is in ComfyUI `output/Prism/<workflow name>/`.

This fork uses an English translation of the official negative prompt. It can be modified in the sampling node. Translating prompt text changes conditioning, so outputs are not guaranteed to match the upstream Chinese-prompt reference.
`sparse_options.json` and `validation_sparse_options.json` are command-line parameter files, not canvas workflows.
A real 480p sample with the same parameters as 01 has completed full 49-frame visual review; still has slight composition drift and soft fine texture. Audio track fully decoded, not yet listened.
Full 480p samples for 02/03/05 and 720p long video for 04 have not been verified. Recommend using 01's default settings first.

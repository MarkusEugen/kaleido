# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Single-file Python CLI tool that applies a kaleidoscope effect to video clips. Decodes/encodes via ffmpeg pipes (preserving HDR), remaps frames with OpenCV or GPU, and muxes audio via ffmpeg.

## Dependencies

```bash
pip install opencv-python numpy          # CPU path (any Python)
pip install torch                        # GPU path (requires Python ≤ 3.12)
# ffmpeg + ffprobe must be on PATH (required for HDR, audio muxing)
```

## Running

```bash
# GPU path (fast, HDR-preserving — requires python3.12 + torch)
python3.12 kaleidoscope.py input.mov -o output.mp4 --segments 8

# CPU fallback (any Python, still HDR if ffmpeg available)
python3 kaleidoscope.py input.mov -o output.mp4 --segments 8

# Multiple files → output directory
python3.12 kaleidoscope.py clip1.mp4 clip2.mov -o out_dir --segments 12 --spin 15
```

## How it works

For each output pixel, it computes the angle/radius from a center point, folds the angle into `360°/segments` wedges with a mirror reflection, and samples the source frame at that folded angle via `cv2.remap` (CPU) or `F.grid_sample` (GPU). That's what produces the tiled, symmetric kaleidoscope look.

Static kaleidoscopes reuse one precomputed coordinate map for every frame (fast); any animated parameter (`--spin`, `--pan-speed`, `--seg-end`) recomputes it per frame.

### I/O pipeline

**GPU path** (`python3.12`, torch with MPS/CUDA available):
- ffmpeg decodes to `p010le` (native 10-bit YUV) — no RGB conversion
- Y and UV planes remapped separately on GPU via `F.grid_sample`
- libx265 encodes from `p010le` — direct YUV, no conversion
- Fast `-c copy` remux stamps HDR color metadata (`hevc_videotoolbox` doesn't embed it when fed via pipe)

**CPU fallback** (no torch GPU):
- ffmpeg decodes to `rgb48le` (16-bit RGB) for OpenCV compatibility
- `cv2.remap` on CPU
- Same libx265 encode + remux

**OpenCV fallback** (no ffprobe/ffmpeg):
- OpenCV `VideoCapture` + `VideoWriter` (8-bit, HDR lost)

### Encoder choice

`libx265 --preset fast` is preferred over `hevc_videotoolbox` for pipe-based encoding. VideoToolbox requires file-backed `CVPixelBuffer`s and silently falls back to slow software when fed via stdin. On Apple Silicon Macs with a file input, VideoToolbox would be faster — but we always pipe here.

## Key knobs

- `--segments` — number of mirrored wedges (6–16 is a good range); accepts floats
- `--seg-end` — ramp segment count to this value over the clip; `0` dissolves the last mirror plane back to the original image
- `--seg-end-time` / `--seg-end-frame` — when the ramp finishes (default: end of clip); mutually exclusive
- `--zoom` — zoom into the source before folding
- `--rotate` — static rotation of the mirror axes in degrees; `--spin` accumulates on top of this
- `--spin` — degrees/sec to rotate the sample angle for an animated swirl
- `--cx` / `--cy` — kaleidoscope symmetry center (0–1, fraction of frame)
- `--ox` / `--oy` — shift what source pixels are sampled without moving the symmetry center
- `--pan-speed` — cycles/sec for the center to wander (try 0.05–0.2); uses golden-ratio Lissajous path so it never repeats
- `--pan-radius` — how far the center wanders from `--cx`/`--cy` (fraction of frame, default 0.15)
- `--no-audio` — skip audio muxing

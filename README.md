# kaleido

Two standalone Python CLI tools for creative video processing — both preserve HDR (10-bit HEVC) via ffmpeg pipes and use GPU acceleration on Apple Silicon (MPS) and CUDA cards.

---

## kaleidoscope.py

Applies a mirrored radial kaleidoscope effect to video clips.  For each output pixel it computes the angle and radius from a center point, folds the angle into `360°/segments` mirror-symmetric wedges, and samples the source frame at the reflected position.  All parameters can be animated over the clip.

### Install

```bash
pip install opencv-python numpy   # required
pip install torch                 # optional — enables GPU path (requires Python ≤ 3.12)
# ffmpeg + ffprobe must be on PATH
```

### Quick start

```bash
# Static kaleidoscope, 8 segments
python3.12 kaleidoscope.py input.mov -o output.mp4 --segments 8

# Animated swirl (15 °/sec rotation)
python3.12 kaleidoscope.py input.mov -o output.mp4 --segments 8 --spin 15

# Ramp from 16 segments down to 1 (dissolves back to original image)
python3.12 kaleidoscope.py input.mov -o output.mp4 --segments 16 --seg-end 1

# Slow-wandering center + zoom
python3.12 kaleidoscope.py input.mov -o output.mp4 --segments 10 --pan-speed 0.1 --zoom 1.4

# Batch — multiple inputs → output directory
python3.12 kaleidoscope.py clip1.mp4 clip2.mov -o out_dir/ --segments 12 --spin 15
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--segments N` | 8 | Number of mirrored wedges. Floats accepted. Good range: 6–16. |
| `--seg-end N` | — | Ramp segment count to N over the clip. `0` dissolves the last mirror plane back to the original image. |
| `--seg-end-time S` | end | Time in seconds when the ramp finishes. |
| `--seg-end-frame F` | end | Frame number when the ramp finishes. Mutually exclusive with `--seg-end-time`. |
| `--zoom Z` | 1.0 | Zoom into the source image before folding. |
| `--rotate DEG` | 0 | Static rotation of the mirror axes in degrees. |
| `--spin DEG/s` | 0 | Degrees per second to rotate the sample angle — produces an animated swirl. Accumulates on top of `--rotate`. |
| `--cx / --cy` | 0.5 | Kaleidoscope symmetry center as fraction of frame width/height. |
| `--ox / --oy` | 0.0 | Shift the source sampling origin without moving the symmetry center (fraction of frame). |
| `--pan-speed C/s` | 0 | Cycles per second for the center to drift. Try 0.05–0.2 for a slow wander. Uses a golden-ratio Lissajous path so the trajectory never repeats. |
| `--pan-radius R` | 0.15 | How far the center wanders from `--cx/--cy` (fraction of frame). |
| `--no-audio` | — | Skip audio muxing. |

### How it works

**GPU path** (Python 3.12 + torch, MPS or CUDA detected automatically):
- ffmpeg decodes to `p010le` (native 10-bit YUV) — no RGB conversion overhead
- Y and UV planes remapped separately on GPU via `torch.nn.functional.grid_sample`
- libx265 encodes from `p010le` — stays in YUV throughout
- A fast `-c copy` remux pass stamps HDR colour metadata (`color_trc`, `color_primaries`, `color_space`)

**CPU fallback** (any Python, ffmpeg required for HDR):
- ffmpeg decodes to `rgb48le` (16-bit RGB)
- OpenCV `cv2.remap` on CPU
- Same libx265 encode + remux

**OpenCV fallback** (no ffmpeg/ffprobe): 8-bit `VideoCapture` + `VideoWriter`, HDR lost.

Threaded I/O queues keep the GPU busy between frames.  Static effects precompute one coordinate map and reuse it; any animated parameter (`--spin`, `--pan-speed`, `--seg-end`) recomputes the map per frame.

---

## led_pack.py

Extracts N vertical LED strips from video and packs them side-by-side into a narrower output.  Designed for footage where the arm carrying the strips is always in frame (no clean background reference shot available).

### Install

```bash
pip install numpy sam2   # sam2 pulls torch + torchvision
# ffmpeg + ffprobe must be on PATH
```

The SAM2 model weights (~39 MB, `facebook/sam2.1-hiera-tiny`) are downloaded automatically from HuggingFace Hub on first run.

### Quick start

```bash
# Default — SAM2 per-frame, full lasso mask cutouts, tracks arm movement
python3.12 led_pack.py input.mov -o packed.mp4 --gap 8

# SAM2 every 30 frames, interpolate masks between — much faster, still tracks
python3.12 led_pack.py input.mov -o packed.mp4 --gap 8 --sam2-interval 30

# SAM2 once on frame 0, fixed masks throughout — fastest, use when arm is static
python3.12 led_pack.py input.mov -o packed.mp4 --gap 8 --lock-boxes

# No SAM2 — fall back to temporal-diff column detection (rectangular crops)
python3.12 led_pack.py input.mov -o packed.mp4 --no-sam2
```

### All options

| Flag | Default | Description |
|---|---|---|
| `--gap N` | 8 | Black pixels between packed strips. |
| `--n-strips N` | 7 | Number of LED strips to detect. |
| `--lock-boxes` | — | Run SAM2 once on frame 0; reuse those pixel masks for the whole video. Fast path. |
| `--sam2-interval N` | 1 | Run SAM2 every N frames; linearly shift masks between keyframes. `1` = per-frame inline (default). |
| `--no-sam2` | — | Disable SAM2; use temporal-diff column detection instead (rectangular crops, no mask). |
| `--threshold N` | 1000 | Diff threshold for `--no-sam2` mode, on the 0–65535 scale. |
| `--no-audio` | — | Skip audio muxing. |

### How it works

1. **Temporal median background** — samples ~30 evenly-spaced frames and computes a per-pixel median.  Because the arm moves during the clip, background pixels are uncovered in enough samples for the median to converge to the true background — no clean reference frame is needed.

2. **Foreground preprocessing** — each frame is background-subtracted and contrast-stretched to a uint8 image where the LED strips appear bright against black.  This is what SAM2 sees, not the raw frame.

3. **SAM2 segmentation** — the column activity profile of the foreground image gives rough strip center x-positions, which become point prompts for SAM2.  SAM2 refines each prompt into a pixel-precise mask that follows the exact shape of the band (angled, curved, varying width) rather than a rectangular column crop.

4. **Lasso pack** — each strip is cut out of the original (full-quality, HDR) frame using its SAM2 mask: pixels outside the mask are zeroed.  The shaped cutouts are placed side-by-side with a black gap on a narrower canvas.

5. **Encode** — same libx265 + remux pipeline as `kaleidoscope.py`; HDR metadata is preserved.

**Mask tracking across frames:**
- `--lock-boxes`: one mask set, used as-is for every frame.
- `--sam2-interval N`: masks stored at keyframes; between keyframes the nearest mask is shifted horizontally by the interpolated column offset.
- Default (`--sam2-interval 1`): SAM2 runs inline for every frame — slowest but gives the tightest cutout at every frame for a fast-moving arm.

---

## Requirements summary

| | kaleidoscope.py | led_pack.py |
|---|---|---|
| Python | any (3.12 for GPU) | 3.12 recommended |
| ffmpeg / ffprobe | required for HDR | required |
| numpy | ✓ | ✓ |
| opencv-python | ✓ | — |
| torch | optional (GPU) | via sam2 |
| sam2 | — | ✓ |

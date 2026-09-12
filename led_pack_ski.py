#!/usr/bin/env python3
"""
led_pack_ski.py — extract LED strips from video using scikit-image (no SAM2).

Detection pipeline (per frame, fully adaptive):
  1. Temporal median background estimation from full clip.
  2. Background subtraction + contrast stretch → grayscale foreground.
  3. Vertical morphological closing to bridge gaps between LED dots.
  4. Otsu threshold → binary foreground.
  5. Column-profile peak detection → strip centre columns.
  6. Voronoi partition: each column assigned to the nearest strip centre.
  7. Largest connected component in each Voronoi zone → strip pixel mask.
  8. Centerline warp: per-row mask centroid is mapped to center column so
     curved strips become vertical in the output.

Usage:
    python3.12 led_pack_ski.py input.mov -o packed.mp4 --gap 8
    python3.12 led_pack_ski.py input.mov -o packed.mp4 --gap 8 --close-radius 12
    python3.12 led_pack_ski.py input.mov -o packed.mp4 --start-frame 2000 --end-frame 2200
"""

import argparse
import json
import os
import subprocess
from queue import Queue
from threading import Thread

import numpy as np


# ── Infrastructure ────────────────────────────────────────────────────────────

def _probe_video(in_path):
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json',
             '-show_streams', '-select_streams', 'v:0', in_path],
            capture_output=True, text=True, check=True,
        )
        return json.loads(r.stdout)['streams'][0]
    except Exception:
        return None


def _pick_hevc_encoder():
    try:
        r = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'],
                           capture_output=True, text=True, check=True)
        enc_list = r.stdout
    except Exception:
        return None
    if 'libx265' in enc_list:
        return ('libx265', 'yuv420p10le', ['-preset', 'fast', '-crf', '24', '-tag:v', 'hvc1'])
    if 'hevc_videotoolbox' in enc_list:
        return ('hevc_videotoolbox', 'p010le', ['-allow_sw', '1', '-tag:v', 'hvc1'])
    return None


# ── Background estimation ─────────────────────────────────────────────────────

def estimate_background(in_path, width, height, n_frames_total, n_samples=30):
    """Temporal median background from evenly-spaced frames across the FULL clip → uint8 (H, W, 3)."""
    step = max(1, n_frames_total // n_samples)
    frame_nbytes = width * height * 6

    reader = subprocess.Popen(
        ['ffmpeg', '-i', in_path,
         '-vf', f'select=not(mod(n\\,{step}))', '-vsync', '0',
         '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-v', 'error', 'pipe:1'],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    frames_u8 = []
    try:
        while len(frames_u8) < n_samples:
            raw = reader.stdout.read(frame_nbytes)
            if len(raw) < frame_nbytes:
                break
            f16 = np.frombuffer(raw, dtype=np.uint16).reshape(height, width, 3)
            frames_u8.append((f16 >> 8).astype(np.uint8))
    finally:
        reader.stdout.close()
        reader.kill()
        reader.wait()

    if not frames_u8:
        raise RuntimeError("Could not read frames for background estimation")
    stack = np.stack(frames_u8, axis=0)
    return np.median(stack, axis=0).astype(np.uint8)


# ── Strip detection (scikit-image) ────────────────────────────────────────────

def _fg_gray(frame_u16, background_u8):
    """Background-subtract, contrast-stretch → grayscale uint8 (H, W)."""
    frame_u8 = (frame_u16 >> 8).astype(np.uint8)
    fg = np.clip(frame_u8.astype(np.int16) - background_u8.astype(np.int16),
                 0, 255).astype(np.uint8)
    peak = int(fg.max())
    if peak > 0:
        fg = (fg.astype(np.float32) * (255.0 / peak)).clip(0, 255).astype(np.uint8)
    return fg.max(axis=2)   # (H, W) uint8


def _col_peaks(col_profile, n_strips):
    """Return n_strips x-positions of the highest local maxima.

    Tries progressively smaller suppression windows so closely-spaced strips
    are not merged into a single peak.
    """
    W = len(col_profile)
    k = np.ones(7, dtype=float) / 7
    s = np.convolve(col_profile.astype(float), k, mode='same')
    threshold = s.max() * 0.02
    top = []
    for divisor in (4, 6, 10, 16, 24):
        half_win = max(3, W // (n_strips * divisor))
        candidates = [i for i in range(W)
                      if s[i] == s[max(0, i - half_win):min(W, i + half_win + 1)].max()
                      and s[i] > threshold]
        top = sorted(candidates, key=lambda i: s[i], reverse=True)[:n_strips]
        if len(top) >= n_strips:
            return sorted(top)
    return sorted(top) if top else []


def _detect_strips_ski(fg_gray, n_strips, close_radius=8):
    """Detect strip pixel masks using morphology + connected components.

    Returns (boxes, masks) or (None, None) if detection fails.
      boxes  — list of n_strips (x0, x1) column spans, sorted left→right
      masks  — list of n_strips (H, W) bool arrays
    """
    from skimage.morphology import closing as ski_closing
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops

    H, W = fg_gray.shape

    # Vertical structuring element — bridges LED dot gaps along the strip
    # without ever merging horizontally adjacent strips.
    struct = np.ones((close_radius * 2 + 1, 1), dtype=bool)
    closed = ski_closing(fg_gray, struct)

    if closed.max() == 0:
        return None, None
    thresh = threshold_otsu(closed)
    binary = closed > thresh                   # (H, W) bool

    # Column profile for peak detection
    col_profile = fg_gray.max(axis=0)          # (W,) — max brightness per column
    peak_cols = _col_peaks(col_profile, n_strips)
    if len(peak_cols) < n_strips:
        return None, None

    # Voronoi: assign each column to the nearest peak
    col_arr   = np.arange(W)
    col_owner = np.abs(col_arr[:, None] - np.array(peak_cols)[None, :]).argmin(axis=1)
    strip_zone = col_owner[None, :]            # (1, W) → broadcasts to (H, W)

    boxes, masks = [], []
    for i in range(n_strips):
        zone_mask = binary & (strip_zone == i)
        labeled   = label(zone_mask)
        if labeled.max() == 0:
            return None, None
        props = regionprops(labeled)
        best  = max(props, key=lambda r: r.area)
        m     = labeled == best.label          # (H, W) bool

        cols = np.where(m.any(axis=0))[0]
        if cols.size == 0:
            return None, None
        boxes.append((int(cols.min()), int(cols.max()) + 1))
        masks.append(m)

    return boxes, masks


# ── Strip warping & packing ───────────────────────────────────────────────────

def _centerline(mask):
    """Per-row x-centroid of the mask, smoothed → float64 (H,)."""
    H = mask.shape[0]
    cx = np.full(H, np.nan)
    for y in range(H):
        cols = np.where(mask[y])[0]
        if cols.size:
            cx[y] = cols.mean()
    nans = np.isnan(cx)
    if nans.all():
        cx[:] = mask.shape[1] / 2.0
    elif nans.any():
        xs = np.arange(H)
        ok = ~nans
        cx[nans] = np.interp(xs[nans], xs[ok], cx[ok])
    k = 21
    kernel = np.ones(k) / k
    cx = np.convolve(cx, kernel, mode='same')
    half = k // 2
    cx[:half] = cx[half]
    cx[H - half:] = cx[H - half - 1]
    return cx


def straighten_strip(frame_u16, mask, strip_width):
    """Warp frame so the strip's curved centerline maps to the center column.

    Returns (H, strip_width, 3) uint16 — the band runs vertically.
    """
    H, W = frame_u16.shape[:2]
    cx = _centerline(mask)
    half_w = strip_width / 2.0
    col_offsets = np.arange(strip_width) - half_w + 0.5
    sample_xs = np.clip(
        np.round(cx[:, None] + col_offsets[None, :]).astype(np.int32),
        0, W - 1,
    )
    row_idx = np.arange(H, dtype=np.int32)[:, None]
    return frame_u16[row_idx, sample_xs, :]


def pack_frame(frame, boxes, col_widths, gap, out_width, masks=None):
    """Pack LED strips into a black canvas.

    With masks: each strip is centerline-warped straight.
    Without masks: rectangular crop.
    """
    height = frame.shape[0]
    canvas = np.zeros((height, out_width, 3), dtype=np.uint16)
    x_cursor = 0
    for i, ((x_start, x_end), cw) in enumerate(zip(boxes, col_widths)):
        if masks is not None:
            strip = straighten_strip(frame, masks[i], cw)
        else:
            copy_w = min(x_end - x_start, cw)
            strip = frame[:, x_start:x_start + copy_w, :].copy()
            if copy_w < cw:
                pad = np.zeros((height, cw - copy_w, 3), dtype=np.uint16)
                strip = np.concatenate([strip, pad], axis=1)
        canvas[:, x_cursor:x_cursor + cw, :] = strip
        x_cursor += cw + gap
    return canvas


# ── Video processing ──────────────────────────────────────────────────────────

def process_video(in_path, out_path, gap=8, n_strips=7, keep_audio=True,
                  close_radius=8, start_frame=None, end_frame=None):

    stream = _probe_video(in_path)
    if not stream:
        raise RuntimeError("ffprobe failed — is ffmpeg installed?")
    hevc = _pick_hevc_encoder()
    if not hevc:
        raise RuntimeError("No HEVC encoder found — install ffmpeg with libx265")

    width   = stream['width']
    height  = stream['height']
    fps_n, fps_d = map(int, stream['r_frame_rate'].split('/'))
    n_frames_total = int(stream.get('nb_frames') or 0)
    if n_frames_total == 0 and stream.get('duration'):
        n_frames_total = int(float(stream['duration']) * fps_n / fps_d)

    fps = fps_n / fps_d
    sf  = max(0, start_frame or 0)
    ef  = min(end_frame, n_frames_total) if end_frame is not None else n_frames_total
    if ef <= sf:
        raise ValueError(f"end_frame ({ef}) must be greater than start_frame ({sf})")
    n_frames   = ef - sf
    t_start    = sf / fps
    t_duration = n_frames / fps
    if sf > 0 or ef < n_frames_total:
        print(f"  Range: frames {sf}–{ef}  ({t_start:.2f}s – {t_start + t_duration:.2f}s)")

    color_space     = stream.get('color_space')
    color_trc       = stream.get('color_transfer')
    color_primaries = stream.get('color_primaries')
    color_range     = stream.get('color_range')

    seek_pre = ['-ss', f'{t_start:.6f}'] if t_start > 0 else []
    dur_arg  = ['-t',  f'{t_duration:.6f}']

    # ── Stage 1: Background (always full clip) ────────────────────────────
    print("  Estimating background (temporal median, full clip)...")
    background_u8 = estimate_background(in_path, width, height, n_frames_total)

    # ── Stage 2: Detect on first frame of range ───────────────────────────
    frame_nbytes = width * height * 6
    print("  Detecting strips on first frame...")
    r0 = subprocess.Popen(
        ['ffmpeg'] + seek_pre + ['-i', in_path, '-vframes', '1',
         '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-v', 'error', 'pipe:1'],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
    )
    raw0 = r0.stdout.read(frame_nbytes)
    r0.stdout.close(); r0.wait()
    f0 = np.frombuffer(raw0, dtype=np.uint16).reshape(height, width, 3).copy()
    fg0 = _fg_gray(f0, background_u8)
    prev_boxes, prev_masks = _detect_strips_ski(fg0, n_strips, close_radius)
    if prev_boxes is None:
        raise RuntimeError(
            f"Could not detect {n_strips} strips in the first frame. "
            "Try --close-radius or --n-strips."
        )
    print(f"  Strips: {prev_boxes}")

    col_widths = [x1 - x0 for x0, x1 in prev_boxes]
    out_width  = sum(col_widths) + (n_strips - 1) * gap
    out_width += out_width % 2
    print(f"  Output: {out_width}×{height}  (gap={gap}px)")

    # ── Stage 3: Encode loop ──────────────────────────────────────────────
    encoder, pix_fmt_out, enc_extra = hevc
    silent_path = out_path + '.silent.mp4'

    reader = subprocess.Popen(
        ['ffmpeg'] + seek_pre + ['-i', in_path] + dur_arg +
        ['-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-v', 'error', 'pipe:1'],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
    )
    encode_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb48le',
        '-s', f'{out_width}x{height}', '-r', f'{fps_n}/{fps_d}',
        '-i', 'pipe:0', '-map', '0:v',
        '-c:v', encoder, '-pix_fmt', pix_fmt_out,
    ] + enc_extra + ['-v', 'error', silent_path]
    writer = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    _DONE    = object()
    decode_q = Queue(maxsize=4)
    encode_q = Queue(maxsize=4)

    def _bg_decode():
        while True:
            raw = reader.stdout.read(frame_nbytes)
            if len(raw) < frame_nbytes:
                decode_q.put(_DONE); return
            decode_q.put(np.frombuffer(raw, dtype=np.uint16).reshape(height, width, 3).copy())

    def _bg_encode():
        while True:
            item = encode_q.get()
            if item is _DONE: return
            writer.stdin.write(item.tobytes())

    decode_thread = Thread(target=_bg_decode, daemon=True)
    encode_thread = Thread(target=_bg_encode, daemon=True)
    decode_thread.start()
    encode_thread.start()

    frame_idx = 0
    basename  = os.path.basename(in_path)
    n_fallback = 0

    try:
        while True:
            item = decode_q.get()
            if item is _DONE:
                break
            frame = item

            fg = _fg_gray(frame, background_u8)
            boxes, masks = _detect_strips_ski(fg, n_strips, close_radius)
            if boxes is None:
                boxes, masks = prev_boxes, prev_masks
                n_fallback += 1
            else:
                prev_boxes, prev_masks = boxes, masks

            canvas = pack_frame(frame, boxes, col_widths, gap, out_width, masks=masks)
            encode_q.put(canvas)
            frame_idx += 1
            if frame_idx % 60 == 0:
                pct = f" ({int(frame_idx / n_frames * 100)}%)" if n_frames > 0 else ""
                print(f"  {basename}: {frame_idx}/{n_frames} frames{pct}")
    finally:
        encode_q.put(_DONE)
        encode_thread.join()
        reader.stdout.close()
        reader.wait()
        writer.stdin.close()
        writer.wait()
        decode_thread.join()

    if n_fallback:
        print(f"  ({n_fallback} frames used fallback from previous detection)")

    # ── Remux: HDR metadata + audio ───────────────────────────────────────
    remux_cmd = ['ffmpeg', '-y', '-i', silent_path]
    if keep_audio:
        remux_cmd += seek_pre + ['-t', f'{t_duration:.6f}', '-i', in_path,
                                 '-map', '0:v', '-map', '1:a?', '-c:a', 'aac', '-shortest']
    else:
        remux_cmd += ['-map', '0:v']
    remux_cmd += ['-c:v', 'copy']
    if color_space:     remux_cmd += ['-colorspace',      color_space]
    if color_trc:       remux_cmd += ['-color_trc',       color_trc]
    if color_primaries: remux_cmd += ['-color_primaries', color_primaries]
    if color_range:     remux_cmd += ['-color_range',     color_range]
    remux_cmd += ['-v', 'error', out_path]
    subprocess.run(remux_cmd, check=True)
    os.remove(silent_path)


def main():
    ap = argparse.ArgumentParser(
        description="Extract LED strips via scikit-image morphology and pack them side-by-side."
    )
    ap.add_argument("input", help="Input video file")
    ap.add_argument("-o", "--output", required=True, help="Output video file")
    ap.add_argument("--gap", type=int, default=8,
                    help="Black pixels between packed strips (default: 8)")
    ap.add_argument("--n-strips", type=int, default=7,
                    help="Number of LED strips to detect (default: 7)")
    ap.add_argument("--close-radius", type=int, default=8,
                    help="Vertical morphological closing radius in pixels (default: 8). "
                         "Increase to bridge larger gaps between LED dots; "
                         "decrease if adjacent strips merge.")
    ap.add_argument("--start-frame", type=int, default=None,
                    help="First frame to process (0-based, inclusive)")
    ap.add_argument("--end-frame", type=int, default=None,
                    help="Last frame to process (exclusive)")
    ap.add_argument("--no-audio", action="store_true", help="Skip audio muxing")
    args = ap.parse_args()

    print(f"Processing {args.input} -> {args.output}")
    process_video(
        args.input, args.output,
        gap=args.gap,
        n_strips=args.n_strips,
        keep_audio=not args.no_audio,
        close_radius=args.close_radius,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    print("Done.")


if __name__ == "__main__":
    main()

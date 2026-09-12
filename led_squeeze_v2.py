#!/usr/bin/env python3
"""
led_squeeze.py — extract each LED strip, straighten it vertically, and remove
                 dark gaps between LEDs.  Output is a compact clip where each
                 packed strip contains only its LED pixels.

Reuses the blob-based tracker from `led_pack.py` so detection matches
`led_track.py` and `led_pack.py --blob-detect`.

Usage:
    python3.12 led_squeeze.py input.mov -o out.mp4 \\
        --start-frame 2000 --end-frame 2200
"""

import argparse
import os
import subprocess
from queue import Queue
from threading import Thread

import numpy as np

from led_pack import (
    _probe_video,
    _pick_hevc_encoder,
    estimate_background,
    detect_strips_blobs,
)


# ── Curve boundary repair ─────────────────────────────────────────────────────
# `detect_strips_blobs` in led_pack.py smooths each per-row centreline with
# `np.convolve(mode='same')`, whose implicit zero-padding pulls the first and
# last ~15 rows toward 0.  That inflates `curve.max() - curve.min()` and, in
# the mask below, plants ghost LED positions near column 0 of every band.
# We fix each curve locally so this file works regardless of which led_pack
# version is on disk.

_CURVE_SMOOTH_K = 31   # must match the kernel size used in led_pack

def _repair_curves(curves):
    if curves is None:
        return curves
    n, H = curves.shape
    half = _CURVE_SMOOTH_K // 2
    fixed = curves.copy()
    fixed[:, :half]      = fixed[:, half:half + 1]
    fixed[:, H - half:H] = fixed[:, H - half - 1:H - half]
    return fixed


# ── Straighten + squeeze ──────────────────────────────────────────────────────

def _extract_band(frame_u16, curve, strip_width, led_half_w):
    """Cut out a fixed-x column window around the curve's middle-row x,
    then zero every pixel that's further than led_half_w from the curve
    at that pixel's row.  The result is the *curved* LED band placed
    inside a rectangular container — no warping, and no neighbouring
    strip's pixels bleed through even if `strip_width` overlaps them.

    Returns (H, strip_width, 3) uint16.
    """
    H, W = frame_u16.shape[:2]
    x_ref  = float(curve[H // 2])
    half_w = strip_width / 2.0
    col_offsets = np.arange(strip_width, dtype=np.float32) - half_w + 0.5
    sample_xs = np.clip(
        np.round(x_ref + col_offsets).astype(np.int32),
        0, W - 1,
    )  # (strip_width,) — same for every row (no warping)
    band = frame_u16[:, sample_xs, :].copy()

    # LED center at each row, in *band* coordinates
    curve_in_band = (curve.astype(np.float32) - x_ref) + half_w - 0.5   # (H,)
    cols = np.arange(strip_width, dtype=np.float32)[None, :]            # (1, W)
    keep = np.abs(cols - curve_in_band[:, None]) <= led_half_w          # (H, W)
    band[~keep] = 0
    return band


def _pack_with_blend(strips, blend_width, gap=0):
    """Pack variable-width strips.  If `blend_width > 0`, adjacent strips
    overlap by that many pixels and are alpha-crossfaded (partition of
    unity → no brightness change).  Otherwise strips abut with `gap`
    black pixels between them.
    """
    n = len(strips)
    H = strips[0].shape[0]
    widths = [s.shape[1] for s in strips]

    if blend_width > 0:
        # Effective overlap per boundary — capped so both sides can fit it.
        overlaps = [min(blend_width, widths[i] // 2, widths[i + 1] // 2)
                    for i in range(n - 1)]
        total = sum(widths) - sum(overlaps)
    else:
        overlaps = [-gap] * (n - 1)   # negative overlap = gap
        total = sum(widths) + gap * (n - 1)
    total += total % 2

    accum  = np.zeros((H, total, 3), dtype=np.float32)
    weight = np.zeros((H, total),    dtype=np.float32)

    x_cursor = 0
    for i, strip in enumerate(strips):
        sw = widths[i]
        alpha = np.ones(sw, dtype=np.float32)
        if i > 0 and overlaps[i - 1] > 0:
            b = overlaps[i - 1]
            alpha[:b] = np.linspace(0.0, 1.0, b + 2)[1:-1]
        if i < n - 1 and overlaps[i] > 0:
            b = overlaps[i]
            alpha[-b:] = np.linspace(1.0, 0.0, b + 2)[1:-1]

        end = min(x_cursor + sw, total)
        w = end - x_cursor
        accum[:, x_cursor:end, :] += strip[:, :w, :].astype(np.float32) * alpha[None, :w, None]
        weight[:, x_cursor:end]   += alpha[None, :w]

        advance = sw - overlaps[i] if i < n - 1 else sw
        x_cursor += advance

    return (accum / np.maximum(weight, 1e-6)[:, :, None]).clip(0, 65535).astype(np.uint16)


def _crop_to_content(band_u16):
    """Drop the fully-black columns on either side of the masked band, so the
    tight rectangle is only as wide as the curve range + LED margin."""
    nonzero_cols = band_u16.any(axis=(0, 2))
    if not nonzero_cols.any():
        return band_u16
    xs = np.where(nonzero_cols)[0]
    return band_u16[:, int(xs[0]):int(xs[-1]) + 1]


def _bright_row_range(band_u16, threshold_frac):
    """Return (y_top, y_bot) inclusive of the LED region, or (None, None) if empty."""
    row_max = band_u16.max(axis=(1, 2))
    peak = int(row_max.max())
    if peak == 0:
        return None, None
    bright = np.where(row_max > peak * threshold_frac)[0]
    if bright.size == 0:
        return None, None
    return int(bright[0]), int(bright[-1])


# ── Video processing ──────────────────────────────────────────────────────────

def process_video(in_path, out_path, n_strips=7, gap=8, out_height=None,
                  threshold_frac=0.15, keep_audio=True,
                  start_frame=None, end_frame=None, half_width=None,
                  blend_width=0):

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
        raise ValueError(f"end_frame ({ef}) must be > start_frame ({sf})")
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

    print("  Estimating background (temporal median, full clip)...")
    background_u8 = estimate_background(in_path, width, height, n_frames_total)

    frame_nbytes = width * height * 6

    # ── Frame 0: detect strips + auto-size the output ────────────────────
    print("  Detecting strips on first frame...")
    r0 = subprocess.Popen(
        ['ffmpeg'] + seek_pre + ['-i', in_path, '-vframes', '1',
         '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-v', 'error', 'pipe:1'],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
    )
    raw0 = r0.stdout.read(frame_nbytes)
    r0.stdout.close(); r0.wait()
    f0 = np.frombuffer(raw0, dtype=np.uint16).reshape(height, width, 3).copy()

    boxes, masks, blob_curves, blob_half_w = detect_strips_blobs(
        f0, background_u8, None, n_strips)
    if boxes is None:
        raise RuntimeError(f"Could not detect {n_strips} strips in frame 0.")
    blob_curves = _repair_curves(blob_curves)

    led_half_w = half_width if half_width else max(10, blob_half_w // 2)

    # `strip_width` needs to hold each band's curve range + LED margin on
    # both sides.  The curve-mask in `_extract_band` prevents neighbour
    # bleed even when strip_width is wider than the inter-strip gap, but
    # we still cap to avoid huge black borders when curves are small.
    max_curve_range = max(
        float(blob_curves[s].max() - blob_curves[s].min())
        for s in range(n_strips)
    )
    strip_width = int(max_curve_range * 1.30 + 2 * led_half_w)
    strip_width += strip_width % 2

    # Auto-detect the LED y-range across all strips on frame 0.  A fixed crop
    # window preserves each LED's native pixel size — no scaling / stretching.
    if out_height is None:
        y_top_all, y_bot_all = height, 0
        for s in range(n_strips):
            band = _extract_band(f0, blob_curves[s], strip_width, led_half_w)
            y0, y1 = _bright_row_range(band, threshold_frac)
            if y0 is not None:
                y_top_all = min(y_top_all, y0)
                y_bot_all = max(y_bot_all, y1)
        margin = 4
        y_top = max(0, y_top_all - margin)
        y_bot = min(height - 1, y_bot_all + margin)
        out_height = y_bot - y_top + 1
    else:
        # User-specified out_height: center it in the frame
        y_top = max(0, (height - out_height) // 2)
    out_height += out_height % 2

    # Measure the tight width of each strip on frame 0 to size the output canvas.
    tight_widths = []
    for s in range(n_strips):
        band = _extract_band(f0, blob_curves[s], strip_width, led_half_w)
        band = band[y_top:y_top + out_height]
        tight = _crop_to_content(band)
        tight_widths.append(tight.shape[1])

    if blend_width > 0:
        overlaps = [min(blend_width, tight_widths[i] // 2, tight_widths[i + 1] // 2)
                    for i in range(n_strips - 1)]
        out_width = sum(tight_widths) - sum(overlaps)
    else:
        out_width = sum(tight_widths) + gap * (n_strips - 1)
    out_width += out_width % 2
    print(f"  strip_width={strip_width}, led_half_w={led_half_w}, "
          f"out_height={out_height}, out_width={out_width}"
          + (f", blend={blend_width}" if blend_width > 0 else f", gap={gap}"))
    print(f"  tight per-strip widths: {tight_widths}")

    # ── Encode loop ───────────────────────────────────────────────────────
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
        '-s', f'{out_width}x{out_height}', '-r', f'{fps_n}/{fps_d}',
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

    frame_idx  = 0
    n_fallback = 0
    basename   = os.path.basename(in_path)

    try:
        while True:
            item = decode_q.get()
            if item is _DONE:
                break
            frame = item

            _, _, new_curves, _ = detect_strips_blobs(
                frame, background_u8, blob_curves, n_strips, half_w=blob_half_w)
            if new_curves is None:
                n_fallback += 1
            else:
                blob_curves = _repair_curves(new_curves)

            strips = []
            for s in range(n_strips):
                band = _extract_band(frame, blob_curves[s], strip_width, led_half_w)
                band = band[y_top:y_top + out_height]
                strips.append(_crop_to_content(band))

            canvas = _pack_with_blend(strips, blend_width, gap=gap)
            # Encoder expects fixed out_width. Pad or crop to match.
            if canvas.shape[1] != out_width:
                if canvas.shape[1] < out_width:
                    pad_w = out_width - canvas.shape[1]
                    # Split padding so content stays centered
                    left = pad_w // 2
                    right = pad_w - left
                    canvas = np.pad(canvas, ((0, 0), (left, right), (0, 0)))
                else:
                    excess = canvas.shape[1] - out_width
                    left = excess // 2
                    canvas = canvas[:, left:left + out_width, :]
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
        print(f"  ({n_fallback} frames reused the previous curves)")

    # ── Remux HDR metadata + audio ────────────────────────────────────────
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
        description="Extract LED strips, straighten and vertically squeeze to "
                    "remove dark gaps between LEDs."
    )
    ap.add_argument("input", help="Input video file")
    ap.add_argument("-o", "--output", required=True, help="Output video file")
    ap.add_argument("--n-strips", type=int, default=7,
                    help="Number of LED strips (default: 7)")
    ap.add_argument("--gap", type=int, default=8,
                    help="Black pixels between packed strips (default: 8)")
    ap.add_argument("--out-height", type=int, default=None,
                    help="Output video height in px (default: auto from frame 0)")
    ap.add_argument("--threshold-frac", type=float, default=0.15,
                    help="Row is kept if row_max > threshold_frac × peak "
                         "(default: 0.15).  Lower keeps more dim rows.")
    ap.add_argument("--half-width", type=int, default=None,
                    help="Half-width for band extraction in pixels. "
                         "Default: half of the auto-detected strip half-width — "
                         "so the extraction hugs the LEDs.")
    ap.add_argument("--blend-width", type=int, default=0,
                    help="Overlap adjacent strips by N pixels and linearly "
                         "alpha-blend the overlap (soft-edge crossfade). "
                         "Ignores --gap when > 0. Default: 0 (hard cut).")
    ap.add_argument("--start-frame", type=int, default=None,
                    help="First frame (0-based, inclusive)")
    ap.add_argument("--end-frame", type=int, default=None,
                    help="Last frame (exclusive)")
    ap.add_argument("--no-audio", action="store_true", help="Skip audio muxing")
    args = ap.parse_args()

    print(f"Squeezing {args.input} -> {args.output}")
    process_video(
        args.input, args.output,
        n_strips=args.n_strips,
        gap=args.gap,
        out_height=args.out_height,
        threshold_frac=args.threshold_frac,
        keep_audio=not args.no_audio,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        half_width=args.half_width,
        blend_width=args.blend_width,
    )
    print("Done.")


if __name__ == "__main__":
    main()

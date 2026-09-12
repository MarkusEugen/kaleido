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


# ── Straighten + squeeze ──────────────────────────────────────────────────────

def _straighten(frame_u16, curve, strip_width):
    """Warp source frame so `curve` maps to the center column of the output.

    Returns (H, strip_width, 3) uint16 — same H as source, band runs vertically.
    """
    H, W = frame_u16.shape[:2]
    half_w = strip_width / 2.0
    col_offsets = np.arange(strip_width) - half_w + 0.5
    sample_xs = np.clip(
        np.round(curve[:, None] + col_offsets[None, :]).astype(np.int32),
        0, W - 1,
    )
    row_idx = np.arange(H, dtype=np.int32)[:, None]
    return frame_u16[row_idx, sample_xs, :]


def _squeeze_rows(strip_u16, out_height, threshold_frac):
    """Keep rows above `threshold_frac × row_peak`; nearest-neighbor sample to out_height.

    Dark inter-LED rows fall below threshold and get dropped.  Bright rows are
    then evenly resampled so different strips (with different numbers of bright
    rows) all fit the same output height.
    """
    row_max = strip_u16.max(axis=(1, 2))
    peak = int(row_max.max())
    if peak == 0:
        return np.zeros((out_height, strip_u16.shape[1], 3), dtype=np.uint16)
    threshold = peak * threshold_frac
    bright_idx = np.where(row_max > threshold)[0]
    if bright_idx.size < 2:
        return np.zeros((out_height, strip_u16.shape[1], 3), dtype=np.uint16)
    kept = strip_u16[bright_idx]                                # (n_bright, W, 3)
    n_kept = kept.shape[0]
    src_idx = np.linspace(0, n_kept - 1, out_height).round().astype(np.int32)
    return kept[src_idx]


def _pack_with_blend(strips, blend_width):
    """Overlap adjacent strips by `blend_width` px and alpha-blend the overlap.

    Each strip is fully opaque in its middle and fades linearly to 0 over
    `blend_width` px at each inner edge (outer edges of the first/last strip
    stay opaque so the whole output has a clean rectangular border).
    Partition-of-unity normalisation → in every overlap region the weights
    sum to 1, so blending is a true crossfade with no brightness change.
    """
    n = len(strips)
    H, sw, _ = strips[0].shape
    b = max(0, min(blend_width, sw // 2 - 1))
    out_width = n * sw - (n - 1) * b
    out_width += out_width % 2

    accum  = np.zeros((H, out_width, 3), dtype=np.float32)
    weight = np.zeros((H, out_width),    dtype=np.float32)
    ramp = (np.linspace(0.0, 1.0, b + 2)[1:-1].astype(np.float32)
            if b > 0 else np.array([], dtype=np.float32))

    for i, strip in enumerate(strips):
        alpha = np.ones(sw, dtype=np.float32)
        if b > 0 and i > 0:
            alpha[:b] = ramp
        if b > 0 and i < n - 1:
            alpha[-b:] = ramp[::-1]

        x0 = i * (sw - b)
        accum[:, x0:x0 + sw, :] += strip.astype(np.float32) * alpha[None, :, None]
        weight[:, x0:x0 + sw]    += alpha[None, :]

    return (accum / np.maximum(weight, 1e-6)[:, :, None]).clip(0, 65535).astype(np.uint16)


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

    extract_half_w = half_width if half_width else max(10, blob_half_w // 2)
    strip_width = int(extract_half_w * 2)
    strip_width += strip_width % 2

    if out_height is None:
        counts = []
        for s in range(n_strips):
            straight = _straighten(f0, blob_curves[s], strip_width)
            row_max = straight.max(axis=(1, 2))
            peak = int(row_max.max())
            if peak > 0:
                counts.append(int((row_max > peak * threshold_frac).sum()))
        if not counts:
            raise RuntimeError("Could not determine output height from frame 0.")
        out_height = int(np.median(counts))
        out_height += out_height % 2

    if blend_width > 0:
        b = min(blend_width, strip_width // 2 - 1)
        out_width = n_strips * strip_width - (n_strips - 1) * b
    else:
        out_width = n_strips * strip_width + (n_strips - 1) * gap
    out_width += out_width % 2
    print(f"  strip_width={strip_width}, out_height={out_height}, out_width={out_width}"
          + (f", blend={blend_width}" if blend_width > 0 else f", gap={gap}"))

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
                blob_curves = new_curves

            strips = []
            for s in range(n_strips):
                straight = _straighten(frame, blob_curves[s], strip_width)
                strips.append(_squeeze_rows(straight, out_height, threshold_frac))
            if blend_width > 0:
                canvas = _pack_with_blend(strips, blend_width)
                # _pack_with_blend rounds to even; pad or crop to expected out_width
                if canvas.shape[1] != out_width:
                    pad_w = out_width - canvas.shape[1]
                    if pad_w > 0:
                        canvas = np.pad(canvas, ((0, 0), (0, pad_w), (0, 0)))
                    else:
                        canvas = canvas[:, :out_width, :]
            else:
                canvas = np.zeros((out_height, out_width, 3), dtype=np.uint16)
                x_cursor = 0
                for strip in strips:
                    canvas[:, x_cursor:x_cursor + strip_width, :] = strip
                    x_cursor += strip_width + gap
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

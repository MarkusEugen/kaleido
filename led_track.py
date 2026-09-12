#!/usr/bin/env python3
"""
led_track.py — overlay a marker on each LED strip at the middle row of every frame.

Diagnostic tool for verifying the blob-based strip tracker from `led_pack.py`.
The output is the source video with one coloured ring per strip, drawn at
(curve[H/2], H/2) — i.e. the middle-row x-position of each tracked strip.
HDR is preserved.

Usage:
    python3.12 led_track.py input.mov -o tracked.mp4 --start-frame 2000 --end-frame 2200
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


# Distinct high-contrast marker colours for up to 8 strips (uint16 HDR range)
STRIP_COLORS = [
    (65535,     0,     0),   # red
    (65535, 32768,     0),   # orange
    (65535, 65535,     0),   # yellow
    (    0, 65535,     0),   # green
    (    0, 65535, 65535),   # cyan
    (    0,     0, 65535),   # blue
    (65535,     0, 65535),   # magenta
    (65535, 65535, 65535),   # white
]


def _draw_ring(frame, cx, cy, color, radius=18, thickness=4):
    """Filled hollow ring on a uint16 (H, W, 3) frame."""
    H, W = frame.shape[:2]
    y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    ring = (d2 <= radius ** 2) & (d2 > (radius - thickness) ** 2)
    frame[y0:y1, x0:x1][ring] = color


def _draw_band_outline(frame, curve, half_w, color, thickness=3,
                       y_frac_top=0.10, y_frac_bot=0.92):
    """Draw a curved rectangular outline that follows `curve` ± half_w.

    Vertical extent is clipped to (y_frac_top·H, y_frac_bot·H) so the outline
    covers the LED region without hitting the top/bottom of the frame.
    """
    H, W = frame.shape[:2]
    y0 = max(0, int(H * y_frac_top))
    y1 = min(H, int(H * y_frac_bot))
    ys = np.arange(y0, y1)
    cx = curve[y0:y1].astype(np.int32)
    x_left  = np.clip(cx - half_w, 0, W - 1)
    x_right = np.clip(cx + half_w, 0, W - 1)

    # Left + right vertical edges, `thickness` px wide
    for dx in range(-(thickness // 2), thickness - thickness // 2):
        xl = np.clip(x_left  + dx, 0, W - 1)
        xr = np.clip(x_right + dx, 0, W - 1)
        frame[ys, xl] = color
        frame[ys, xr] = color

    # Top horizontal edge
    tl, tr = int(x_left[0]),  int(x_right[0])
    if tl < tr:
        frame[y0:y0 + thickness, tl:tr + 1] = color

    # Bottom horizontal edge
    bl, br = int(x_left[-1]), int(x_right[-1])
    if bl < br:
        frame[y1 - thickness:y1, bl:br + 1] = color


def process_video(in_path, out_path, n_strips=7, keep_audio=True,
                  start_frame=None, end_frame=None, marker_radius=18,
                  half_width=None):

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

    print("  Estimating background (temporal median, full clip)...")
    background_u8 = estimate_background(in_path, width, height, n_frames_total)

    frame_nbytes = width * height * 6
    y_marker     = height // 2

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
        '-s', f'{width}x{height}', '-r', f'{fps_n}/{fps_d}',
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

    blob_curves  = None
    blob_half_w  = None
    frame_idx    = 0
    n_fallback   = 0
    basename     = os.path.basename(in_path)

    try:
        while True:
            item = decode_q.get()
            if item is _DONE:
                break
            frame = item

            boxes, masks, new_curves, new_half_w = detect_strips_blobs(
                frame, background_u8, blob_curves, n_strips, half_w=blob_half_w)
            if boxes is None:
                n_fallback += 1
                # keep blob_curves at its last-known good value
            else:
                blob_curves = new_curves
                blob_half_w = new_half_w

            if blob_curves is not None and blob_half_w is not None:
                outline_hw = half_width if half_width else max(10, blob_half_w // 2)
                for s in range(min(n_strips, len(STRIP_COLORS))):
                    color = STRIP_COLORS[s]
                    _draw_band_outline(frame, blob_curves[s], outline_hw, color,
                                       thickness=3)
                    cx = int(blob_curves[s, y_marker])
                    _draw_ring(frame, cx, y_marker, color, radius=marker_radius)

            encode_q.put(frame)
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
        print(f"  ({n_fallback} frames used the last-known curves — blob detection failed)")

    # Remux: HDR metadata + optional audio
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
        description="Overlay a marker on each LED strip's middle-row position "
                    "throughout the selected frame range."
    )
    ap.add_argument("input", help="Input video file")
    ap.add_argument("-o", "--output", required=True, help="Output video file")
    ap.add_argument("--n-strips", type=int, default=7,
                    help="Number of LED strips to track (default: 7)")
    ap.add_argument("--start-frame", type=int, default=None,
                    help="First frame (0-based, inclusive)")
    ap.add_argument("--end-frame", type=int, default=None,
                    help="Last frame (exclusive)")
    ap.add_argument("--marker-radius", type=int, default=18,
                    help="Radius of the tracking marker in pixels (default: 18)")
    ap.add_argument("--half-width", type=int, default=None,
                    help="Half-width for the band outline in pixels. "
                         "If not set, uses half of the auto-detected strip half-width "
                         "so the outline hugs the LEDs.")
    ap.add_argument("--no-audio", action="store_true", help="Skip audio muxing")
    args = ap.parse_args()

    print(f"Tracking LEDs in {args.input} -> {args.output}")
    process_video(
        args.input, args.output,
        n_strips=args.n_strips,
        keep_audio=not args.no_audio,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        marker_radius=args.marker_radius,
        half_width=args.half_width,
    )
    print("Done.")


if __name__ == "__main__":
    main()

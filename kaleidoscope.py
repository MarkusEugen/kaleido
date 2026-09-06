#!/usr/bin/env python3
"""
kaleidoscope.py — apply a kaleidoscope effect to video clips.

Requirements:
    pip install opencv-python numpy
    ffmpeg must be installed and on PATH (recommended — required for HDR preservation)

Usage:
    python kaleidoscope.py input.mp4 -o output.mp4 --segments 8
    python kaleidoscope.py clip1.mp4 clip2.mov -o out_dir --segments 12 --spin 15
    python kaleidoscope.py input.mp4 -o output.mp4 --segments 6 --zoom 1.4 --cx 0.5 --cy 0.4
"""

import argparse
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np


def build_maps(width, height, segments, cx, cy, zoom, angle_offset_deg=0.0, ox=0.0, oy=0.0):
    """Precompute remap coordinate grids for a mirrored radial-segment kaleidoscope.

    segments < 1 blends the last mirror plane out toward a plain pass-through (segments=0).
    """
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    center_x = width * cx
    center_y = height * cy

    dx = (xs - center_x) / zoom
    dy = (ys - center_y) / zoom

    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx) + np.radians(angle_offset_deg)

    # Clamp to ≥1 for the fold math; use the sub-1 fraction to blend the mirror out.
    mirror_blend = float(np.clip(segments, 0.0, 1.0))
    eff_segs = max(segments, 1.0)

    seg_angle = 2 * np.pi / eff_segs
    theta_mod = np.mod(theta, seg_angle)
    # mirror the second half of each wedge so segments tile seamlessly
    mirror = theta_mod > (seg_angle / 2)
    theta_mod = np.where(mirror, seg_angle - theta_mod, theta_mod)

    src_ox = ox * width
    src_oy = oy * height
    map_x_k = center_x + src_ox + r * np.cos(theta_mod)
    map_y_k = center_y + src_oy + r * np.sin(theta_mod)

    if mirror_blend < 1.0:
        # Blend toward a plain pass-through (each output pixel → same source pixel + offset).
        map_x = (map_x_k * mirror_blend + (xs + src_ox) * (1 - mirror_blend)).astype(np.float32)
        map_y = (map_y_k * mirror_blend + (ys + src_oy) * (1 - mirror_blend)).astype(np.float32)
    else:
        map_x = map_x_k.astype(np.float32)
        map_y = map_y_k.astype(np.float32)

    return map_x, map_y


_PHI = (1 + math.sqrt(5)) / 2  # golden ratio — makes x/y pan frequencies incommensurable


def _detect_device():
    """Return best available torch GPU device, or None if torch is unavailable/CPU-only."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
    except ImportError:
        pass
    return None


def _build_maps_gpu(width, height, segments, cx, cy, zoom, angle_offset_deg, ox, oy, device):
    """GPU equivalent of build_maps.

    Returns (luma_grid, chroma_grid):
      luma_grid   — [1, H,   W,   2] grid for F.grid_sample on the Y plane
      chroma_grid — [1, H/2, W/2, 2] grid for F.grid_sample on the UV planes (4:2:0)
    """
    import torch

    ys = torch.arange(height, dtype=torch.float32, device=device)
    xs = torch.arange(width,  dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    center_x, center_y = width * cx, height * cy
    dx = (grid_x - center_x) / zoom
    dy = (grid_y - center_y) / zoom

    r     = torch.sqrt(dx * dx + dy * dy)
    theta = torch.atan2(dy, dx) + math.radians(angle_offset_deg)

    mirror_blend = min(max(float(segments), 0.0), 1.0)
    eff_segs  = max(float(segments), 1.0)
    seg_angle = 2 * math.pi / eff_segs

    theta_mod = theta.fmod(seg_angle)
    theta_mod = torch.where(theta_mod < 0, theta_mod + seg_angle, theta_mod)
    mirror    = theta_mod > (seg_angle / 2)
    theta_mod = torch.where(mirror, seg_angle - theta_mod, theta_mod)

    src_ox, src_oy = ox * width, oy * height
    map_x_k = center_x + src_ox + r * torch.cos(theta_mod)
    map_y_k = center_y + src_oy + r * torch.sin(theta_mod)

    if mirror_blend < 1.0:
        map_x = map_x_k * mirror_blend + (grid_x + src_ox) * (1 - mirror_blend)
        map_y = map_y_k * mirror_blend + (grid_y + src_oy) * (1 - mirror_blend)
    else:
        map_x, map_y = map_x_k, map_y_k

    # Luma grid — normalize full-resolution coords to [-1, 1]
    luma_grid = torch.stack([
        map_x / (width  - 1) * 2 - 1,
        map_y / (height - 1) * 2 - 1,
    ], dim=-1).unsqueeze(0)  # [1, H, W, 2]

    # Chroma grid — average 2×2 luma blocks → chroma pixel coords → normalize
    cw, ch = width // 2, height // 2
    cm_x = (map_x[0::2, 0::2] + map_x[0::2, 1::2] + map_x[1::2, 0::2] + map_x[1::2, 1::2]) / 4 / 2
    cm_y = (map_y[0::2, 0::2] + map_y[0::2, 1::2] + map_y[1::2, 0::2] + map_y[1::2, 1::2]) / 4 / 2
    chroma_grid = torch.stack([
        cm_x / (cw - 1) * 2 - 1,
        cm_y / (ch - 1) * 2 - 1,
    ], dim=-1).unsqueeze(0)  # [1, H/2, W/2, 2]

    return luma_grid, chroma_grid


def _remap_gpu_yuv(y_plane, uv_plane, luma_grid, chroma_grid, device):
    """Remap a p010le frame (Y + interleaved UV) on GPU. Returns (y_out, uv_out) as uint16."""
    import torch, torch.nn.functional as F

    h, w = y_plane.shape
    ch, cw = h // 2, w // 2

    # Y — [1, 1, H, W]
    y_t = torch.from_numpy(y_plane.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    y_out = F.grid_sample(y_t, luma_grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    y_np = y_out.squeeze().clamp(0, 65535).cpu().numpy().astype(np.uint16)

    # UV — NV12 interleaved [H/2, W], reshape to [1, 2, H/2, W/2]
    uv_t = torch.from_numpy(uv_plane.astype(np.float32)).reshape(ch, cw, 2).permute(2, 0, 1).unsqueeze(0).to(device)
    uv_out = F.grid_sample(uv_t, chroma_grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    # [1, 2, H/2, W/2] → [H/2, W/2, 2] → [H/2, W] interleaved
    uv_np = uv_out.squeeze(0).permute(1, 2, 0).clamp(0, 65535).cpu().numpy().reshape(ch, w).astype(np.uint16)

    return y_np, uv_np


def _remap_gpu(frame_np, grid, device):
    """Apply the precomputed grid to an rgb48le frame using GPU grid_sample."""
    import torch, torch.nn.functional as F

    luma_grid = grid  # for rgb path, grid is the luma grid
    max_val = float(np.iinfo(frame_np.dtype).max)
    t = torch.from_numpy(frame_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    out = F.grid_sample(t, luma_grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    return out.squeeze(0).permute(1, 2, 0).clamp(0, max_val).cpu().numpy().astype(frame_np.dtype)


def _probe_video(in_path):
    """Return first video stream info dict via ffprobe, or None if unavailable."""
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
    """Return (encoder, pix_fmt, extra_flags) for the best available HEVC encoder, or None.

    libx265 is preferred over hevc_videotoolbox for pipe-based encoding: VideoToolbox requires
    file-backed CVPixelBuffers and silently falls back to slow software when fed via stdin.
    On Apple Silicon with a file source, VideoToolbox is faster — but we always pipe here.
    """
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


def process_video(in_path, out_path, segments, zoom, cx, cy, spin_deg_per_sec, rotate=0.0,
                  pan_speed=0.0, pan_radius=0.15, ox=0.0, oy=0.0,
                  seg_end=None, seg_end_frame=None, seg_end_time=None, keep_audio=True):

    device = _detect_device()
    if device is not None:
        print(f"  GPU: {device.type.upper()}")

    stream = _probe_video(in_path)
    hevc = _pick_hevc_encoder() if stream else None

    if stream and hevc:
        _process_ffmpeg(in_path, out_path, stream, hevc, segments, zoom, cx, cy,
                        spin_deg_per_sec, rotate, pan_speed, pan_radius, ox, oy,
                        seg_end, seg_end_frame, seg_end_time, keep_audio, device)
    else:
        if stream and not hevc:
            print("  (no HEVC encoder found — falling back to 8-bit H.264, HDR will be lost)")
        elif not stream:
            print("  (ffprobe unavailable — falling back to OpenCV 8-bit path)")
        _process_opencv(in_path, out_path, segments, zoom, cx, cy,
                        spin_deg_per_sec, rotate, pan_speed, pan_radius, ox, oy,
                        seg_end, seg_end_frame, seg_end_time, keep_audio, device)


def _run_frame_loop(width, height, fps, n_frames, segments, zoom, cx, cy,
                    spin_deg_per_sec, rotate, pan_speed, pan_radius, ox, oy,
                    seg_end, seg_end_frame, seg_end_time, read_frame, write_frame, basename,
                    device=None, yuv=False):
    """Core per-frame loop shared by both I/O paths.

    When device is set and yuv=True, read_frame must return (y_plane, uv_plane) tuples
    and write_frame must accept them — the remap runs in YUV on the GPU.
    """
    needs_per_frame = spin_deg_per_sec or pan_speed or (seg_end is not None and seg_end != segments)

    static_luma_grid = static_chroma_grid = None
    static_map_x = static_map_y = None
    if not needs_per_frame:
        if device is not None:
            static_luma_grid, static_chroma_grid = _build_maps_gpu(
                width, height, segments, cx, cy, zoom, rotate, ox, oy, device)
        else:
            static_map_x, static_map_y = build_maps(
                width, height, segments, cx, cy, zoom, rotate, ox=ox, oy=oy)

    frame_idx = 0
    while True:
        frame = read_frame()
        if frame is None:
            break

        if needs_per_frame:
            t = frame_idx / fps
            angle_offset = rotate + t * spin_deg_per_sec
            if pan_speed:
                cx_t = cx + pan_radius * math.sin(2 * math.pi * pan_speed * t)
                cy_t = cy + pan_radius * math.sin(2 * math.pi * pan_speed * t * _PHI)
            else:
                cx_t, cy_t = cx, cy
            if seg_end is not None:
                if seg_end_frame is not None:
                    ramp_end = seg_end_frame
                elif seg_end_time is not None:
                    ramp_end = int(seg_end_time * fps)
                else:
                    ramp_end = max(n_frames - 1, 1)
                progress = min(frame_idx / ramp_end, 1.0)
                segs_t = segments + (seg_end - segments) * progress
            else:
                segs_t = segments

            if device is not None:
                luma_grid, chroma_grid = _build_maps_gpu(
                    width, height, segs_t, cx_t, cy_t, zoom, angle_offset, ox, oy, device)
            else:
                map_x, map_y = build_maps(
                    width, height, segs_t, cx_t, cy_t, zoom, angle_offset, ox=ox, oy=oy)
        else:
            luma_grid, chroma_grid = static_luma_grid, static_chroma_grid
            map_x, map_y = static_map_x, static_map_y

        if device is not None and yuv:
            y_plane, uv_plane = frame
            out_frame = _remap_gpu_yuv(y_plane, uv_plane, luma_grid, chroma_grid, device)
        elif device is not None:
            out_frame = _remap_gpu(frame, luma_grid, device)
        else:
            out_frame = cv2.remap(frame, map_x, map_y,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
        write_frame(out_frame)
        frame_idx += 1
        if frame_idx % 60 == 0:
            if n_frames > 0:
                print(f"  {basename}: {frame_idx}/{n_frames} frames")
            else:
                print(f"  {basename}: {frame_idx} frames")


def _process_ffmpeg(in_path, out_path, stream, hevc,
                    segments, zoom, cx, cy, spin_deg_per_sec, rotate,
                    pan_speed, pan_radius, ox, oy,
                    seg_end, seg_end_frame, seg_end_time, keep_audio, device=None):
    width = stream['width']
    height = stream['height']
    fps_n, fps_d = map(int, stream['r_frame_rate'].split('/'))
    fps = fps_n / fps_d
    n_frames = int(stream.get('nb_frames') or 0)
    if n_frames == 0 and stream.get('duration'):
        n_frames = int(float(stream['duration']) * fps)

    color_space     = stream.get('color_space')
    color_trc       = stream.get('color_transfer')
    color_primaries = stream.get('color_primaries')
    color_range     = stream.get('color_range')

    encoder, pix_fmt_out, enc_extra = hevc
    silent_path = out_path + '.silent.mp4'

    # With GPU: decode/encode p010le (native 10-bit YUV) — skips RGB conversion entirely.
    # Without GPU: decode rgb48le so OpenCV can remap on CPU.
    if device is not None:
        pipe_fmt   = 'p010le'
        y_bytes    = width * height * 2
        uv_bytes   = width * (height // 2) * 2
        frame_nbytes = y_bytes + uv_bytes
    else:
        pipe_fmt     = 'rgb48le'
        frame_nbytes = width * height * 6

    reader = subprocess.Popen(
        ['ffmpeg', '-i', in_path,
         '-f', 'rawvideo', '-pix_fmt', pipe_fmt, '-v', 'error', 'pipe:1'],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
    )

    encode_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', pipe_fmt,
        '-s', f'{width}x{height}', '-r', f'{fps_n}/{fps_d}',
        '-i', 'pipe:0',
        '-map', '0:v',
        '-c:v', encoder, '-pix_fmt', pix_fmt_out,
    ] + enc_extra + ['-v', 'error', silent_path]

    writer = subprocess.Popen(encode_cmd, stdin=subprocess.PIPE)

    if device is not None:
        # Threaded I/O: decode and encode run in background threads so the GPU
        # stays busy instead of waiting on pipe reads/writes between frames.
        from queue import Queue
        from threading import Thread

        _DONE = object()
        decode_q = Queue(maxsize=4)
        encode_q = Queue(maxsize=4)

        def _bg_decode():
            while True:
                raw = reader.stdout.read(frame_nbytes)
                if len(raw) < frame_nbytes:
                    decode_q.put(_DONE)
                    return
                y  = np.frombuffer(raw[:y_bytes],  dtype=np.uint16).reshape(height,      width).copy()
                uv = np.frombuffer(raw[y_bytes:],  dtype=np.uint16).reshape(height // 2, width).copy()
                decode_q.put((y, uv))

        def _bg_encode():
            while True:
                item = encode_q.get()
                if item is _DONE:
                    return
                y_out, uv_out = item
                writer.stdin.write(y_out.tobytes())
                writer.stdin.write(uv_out.tobytes())

        decode_thread = Thread(target=_bg_decode, daemon=True)
        encode_thread = Thread(target=_bg_encode, daemon=True)
        decode_thread.start()
        encode_thread.start()

        def read_frame():
            item = decode_q.get()
            return None if item is _DONE else item

        def write_frame(yuv_tuple):
            encode_q.put(yuv_tuple)

    else:
        decode_thread = encode_thread = None

        def read_frame():
            raw = reader.stdout.read(frame_nbytes)
            if len(raw) < frame_nbytes:
                return None
            return np.frombuffer(raw, dtype=np.uint16).reshape(height, width, 3).copy()

        def write_frame(f):
            writer.stdin.write(f.tobytes())

    try:
        _run_frame_loop(width, height, fps, n_frames, segments, zoom, cx, cy,
                        spin_deg_per_sec, rotate, pan_speed, pan_radius, ox, oy,
                        seg_end, seg_end_frame, seg_end_time,
                        read_frame, write_frame, os.path.basename(in_path),
                        device=device, yuv=(device is not None))
    finally:
        if encode_thread is not None:
            encode_q.put(_DONE)
            encode_thread.join()
        reader.stdout.close()
        reader.wait()
        writer.stdin.close()
        writer.wait()
        if decode_thread is not None:
            decode_thread.join()

    # Remux: stamp HDR metadata and mux audio in a single -c copy pass.
    remux_cmd = ['ffmpeg', '-y', '-i', silent_path]
    if keep_audio:
        remux_cmd += ['-i', in_path, '-map', '0:v', '-map', '1:a?', '-c:a', 'aac', '-shortest']
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


def _process_opencv(in_path, out_path, segments, zoom, cx, cy,
                    spin_deg_per_sec, rotate, pan_speed, pan_radius, ox, oy,
                    seg_end, seg_end_frame, seg_end_time, keep_audio, device=None):
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    silent_path = out_path + ".silent.mp4" if keep_audio else out_path
    # avc1 (H.264 via VideoToolbox) is required on macOS; mp4v silently produces empty files.
    for fourcc_str in ("avc1", "mp4v"):
        cv_writer = cv2.VideoWriter(
            silent_path, cv2.VideoWriter_fourcc(*fourcc_str), fps, (width, height)
        )
        if cv_writer.isOpened():
            break
    if not cv_writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {silent_path}")

    def read_frame():
        ret, frame = cap.read()
        return frame if ret else None

    def write_frame(f):
        cv_writer.write(f)

    try:
        _run_frame_loop(width, height, fps, n_frames, segments, zoom, cx, cy,
                        spin_deg_per_sec, rotate, pan_speed, pan_radius, ox, oy,
                        seg_end, seg_end_frame, seg_end_time,
                        read_frame, write_frame, os.path.basename(in_path), device=device)
    finally:
        cap.release()
        cv_writer.release()

    if keep_audio:
        mux_audio(in_path, silent_path, out_path)
        if os.path.exists(silent_path):
            os.remove(silent_path)


def mux_audio(original_path, silent_video_path, out_path):
    """Copy audio from the original clip onto the processed (silent) video, via ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", silent_video_path,
        "-i", original_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("  (no audio muxed — ffmpeg unavailable or source has no audio track)")
        os.replace(silent_video_path, out_path)


def main():
    ap = argparse.ArgumentParser(description="Apply a kaleidoscope effect to video clips.")
    ap.add_argument("inputs", nargs="+", help="Input video file(s)")
    ap.add_argument("-o", "--output", required=True,
                     help="Output file (single input) or output directory (multiple inputs)")
    ap.add_argument("--segments", type=int, default=8, help="Number of mirrored wedges (default: 8)")
    ap.add_argument("--zoom", type=float, default=1.0, help="Zoom factor on source (default: 1.0)")
    ap.add_argument("--cx", type=float, default=0.5, help="Center x as fraction of width (default: 0.5)")
    ap.add_argument("--cy", type=float, default=0.5, help="Center y as fraction of height (default: 0.5)")
    ap.add_argument("--spin", type=float, default=0.0,
                     help="Degrees/sec to rotate the sampled wedge for an animated swirl (default: 0, static)")
    ap.add_argument("--rotate", type=float, default=0.0,
                     help="Static rotation of the mirror axes in degrees (default: 0). "
                          "--spin adds on top of this.")
    ap.add_argument("--pan-speed", type=float, default=0.0,
                     help="Cycles/sec for the center to wander (default: 0, static). "
                          "Try 0.05–0.2 for a slow drift.")
    ap.add_argument("--pan-radius", type=float, default=0.15,
                     help="How far the center wanders from --cx/--cy, as a fraction of frame (default: 0.15)")
    ap.add_argument("--seg-end", type=float, default=None,
                     help="If set, segment count ramps linearly from --segments to this value.")
    seg_at = ap.add_mutually_exclusive_group()
    seg_at.add_argument("--seg-end-time", type=float, default=None,
                        help="Time in seconds at which --seg-end is reached (default: end of clip).")
    seg_at.add_argument("--seg-end-frame", type=int, default=None,
                        help="Frame number at which --seg-end is reached (default: end of clip).")
    ap.add_argument("--ox", type=float, default=0.0,
                     help="Source image offset x as a fraction of width (default: 0). "
                          "Shifts what content is sampled without moving the symmetry center.")
    ap.add_argument("--oy", type=float, default=0.0,
                     help="Source image offset y as a fraction of height (default: 0).")
    ap.add_argument("--no-audio", action="store_true", help="Skip audio muxing")
    args = ap.parse_args()

    multi = len(args.inputs) > 1
    if multi:
        os.makedirs(args.output, exist_ok=True)

    for in_path in args.inputs:
        if multi:
            base = os.path.splitext(os.path.basename(in_path))[0]
            out_path = os.path.join(args.output, f"{base}_kaleidoscope.mp4")
        else:
            out_path = args.output

        print(f"Processing {in_path} -> {out_path}")
        process_video(
            in_path, out_path,
            segments=args.segments,
            zoom=args.zoom,
            cx=args.cx,
            cy=args.cy,
            spin_deg_per_sec=args.spin,
            rotate=args.rotate,
            pan_speed=args.pan_speed,
            pan_radius=args.pan_radius,
            ox=args.ox,
            oy=args.oy,
            seg_end=args.seg_end,
            seg_end_frame=args.seg_end_frame,
            seg_end_time=args.seg_end_time,
            keep_audio=not args.no_audio,
        )

    print("Done.")


if __name__ == "__main__":
    main()

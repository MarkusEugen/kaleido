#!/usr/bin/env python3
"""
led_pack.py — extract LED strips from video and pack them side-by-side.

Detection pipeline:
  1. Temporal median background — sample ~30 evenly-spaced frames, compute per-pixel
     median.  Works without a clean background shot because the arm moves enough to
     expose background pixels over time.
  2. SAM2 segmentation — subtract background from each frame, find peak columns in the
     foreground image as point prompts, let SAM2 refine each prompt into a pixel-precise
     mask.  Each strip is cut out in its exact shape (not a rectangle) before packing.

Usage:
    python3.12 led_pack.py input.mov -o packed.mp4 --gap 8
    python3.12 led_pack.py input.mov -o packed.mp4 --gap 8 --sam2-interval 30
    python3.12 led_pack.py input.mov -o packed.mp4 --gap 8 --lock-boxes
    python3.12 led_pack.py input.mov -o packed.mp4 --no-sam2   # diff-based fallback
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


def _detect_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.device('mps')
        if torch.cuda.is_available():
            return torch.device('cuda')
    except ImportError:
        pass
    return None


# ── Background estimation ─────────────────────────────────────────────────────

def estimate_background(in_path, width, height, n_frames, n_samples=30):
    """Temporal median background from evenly-spaced frames → uint8 (H, W, 3)."""
    step = max(1, n_frames // n_samples)
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


# ── SAM2 detection ────────────────────────────────────────────────────────────

def load_sam2(device):
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    dev_str = device.type if device is not None else 'cpu'
    return SAM2ImagePredictor.from_pretrained('facebook/sam2.1-hiera-tiny', device=dev_str)


def _fg_for_sam2(frame_u16, background_u8):
    """Background-subtract frame → contrast-stretched uint8 for SAM2."""
    frame_u8 = (frame_u16 >> 8).astype(np.uint8)
    fg = np.clip(frame_u8.astype(np.int16) - background_u8.astype(np.int16), 0, 255).astype(np.uint8)
    peak = int(fg.max())
    if peak > 0:
        fg = (fg.astype(np.float32) * (255.0 / peak)).clip(0, 255).astype(np.uint8)
    return fg


def _col_peaks(col_profile, n_strips):
    """Return n_strips x-positions of the highest local maxima in a 1-D profile."""
    W = len(col_profile)
    k = np.ones(7, dtype=float) / 7
    s = np.convolve(col_profile.astype(float), k, mode='same')
    half_win = max(15, W // (n_strips * 6))
    peaks = [i for i in range(W)
             if s[i] == s[max(0, i - half_win):min(W, i + half_win + 1)].max()
             and s[i] > s.max() * 0.02]
    if not peaks:
        return []
    peaks = sorted(peaks, key=lambda i: s[i], reverse=True)[:n_strips]
    return sorted(peaks)


def _sam2_detect(predictor, fg_u8, prompt_xs, height):
    """Run SAM2 with one point prompt per strip.

    Returns (boxes, masks):
      boxes — sorted list of (x0, x1) column spans
      masks — list of (H, W) bool arrays, one per strip, in the same order
    """
    import torch
    predictor.set_image(fg_u8)
    boxes = []
    masks = []
    with torch.inference_mode():
        for x in prompt_xs:
            preds, scores, _ = predictor.predict(
                point_coords=np.array([[x, height // 2]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )
            mask = preds[int(scores.argmax())]   # (H, W) bool
            cols = np.where(mask.any(axis=0))[0]
            if cols.size:
                boxes.append((int(cols.min()), int(cols.max()) + 1))
                masks.append(mask)
    # Sort both lists by x0
    paired = sorted(zip(boxes, masks), key=lambda bm: bm[0][0])
    if paired:
        boxes, masks = zip(*paired)
        return list(boxes), list(masks)
    return [], []


def detect_strips_sam2(predictor, frame_u16, background_u8, prev_boxes, prev_masks, n_strips):
    """Detect strip boxes + pixel masks using background subtraction + SAM2.

    Returns (boxes, masks) if exactly n_strips detected, else (prev_boxes, prev_masks).
    """
    height = frame_u16.shape[0]
    fg = _fg_for_sam2(frame_u16, background_u8)
    col_profile = fg.max(axis=2).max(axis=0)
    prompt_xs = _col_peaks(col_profile, n_strips)
    if len(prompt_xs) < n_strips:
        return prev_boxes, prev_masks

    boxes, masks = _sam2_detect(predictor, fg, prompt_xs, height)
    if len(boxes) == n_strips:
        return boxes, masks
    return prev_boxes, prev_masks


# ── Keyframe interpolation with mask shifting ─────────────────────────────────

def _shift_mask_cols(mask, delta):
    """Shift a boolean mask horizontally by delta pixels (positive = right)."""
    if delta == 0:
        return mask
    W = mask.shape[1]
    result = np.zeros_like(mask)
    if delta > 0:
        result[:, delta:] = mask[:, :W - delta]
    else:
        result[:, :W + delta] = mask[:, -delta:]
    return result


def _interp_state(keyframes, frame_idx):
    """Interpolate box positions and shift nearest mask to match.

    Returns (boxes, masks) for frame_idx.
    """
    idxs = sorted(keyframes)
    lo = max((i for i in idxs if i <= frame_idx), default=idxs[0])
    hi = min((i for i in idxs if i >= frame_idx), default=idxs[-1])

    if lo == hi:
        kf = keyframes[lo]
        return kf['boxes'], kf['masks']

    t = (frame_idx - lo) / (hi - lo)
    lo_kf = keyframes[lo]
    hi_kf = keyframes[hi]

    interp_boxes = [
        (int(a0 + (a1 - a0) * t), int(b0 + (b1 - b0) * t))
        for (a0, b0), (a1, b1) in zip(lo_kf['boxes'], hi_kf['boxes'])
    ]

    # Shift the nearer keyframe's masks by the horizontal delta
    near_kf = lo_kf if t < 0.5 else hi_kf
    shifted_masks = []
    for i, (near_box, interp_box) in enumerate(zip(near_kf['boxes'], interp_boxes)):
        near_cx   = (near_box[0]   + near_box[1])   // 2
        interp_cx = (interp_box[0] + interp_box[1]) // 2
        shifted_masks.append(_shift_mask_cols(near_kf['masks'][i], interp_cx - near_cx))

    return interp_boxes, shifted_masks


def _build_keyframes(in_path, width, height, background_u8, predictor,
                     n_strips, n_frames, sam2_interval):
    """Pre-scan video at keyframe positions; return {frame_idx: {'boxes':…,'masks':…}}."""
    frame_nbytes = width * height * 6
    reader = subprocess.Popen(
        ['ffmpeg', '-i', in_path,
         '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-v', 'error', 'pipe:1'],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    keyframes  = {}
    prev_boxes = None
    prev_masks = None
    frame_idx  = 0
    try:
        while True:
            raw = reader.stdout.read(frame_nbytes)
            if len(raw) < frame_nbytes:
                break
            if frame_idx % sam2_interval == 0:
                frame = np.frombuffer(raw, dtype=np.uint16).reshape(height, width, 3)
                boxes, masks = detect_strips_sam2(
                    predictor, frame, background_u8, prev_boxes, prev_masks, n_strips)
                if boxes is not None and len(boxes) == n_strips:
                    keyframes[frame_idx] = {'boxes': boxes, 'masks': masks}
                    prev_boxes, prev_masks = boxes, masks
            frame_idx += 1
            if frame_idx % 300 == 0:
                pct = int(frame_idx / max(n_frames, 1) * 100)
                print(f"  SAM2 keyframe scan: {frame_idx}/{n_frames} ({pct}%)")
    finally:
        reader.stdout.close()
        reader.kill()
        reader.wait()

    if not keyframes:
        raise RuntimeError(
            f"SAM2 could not detect {n_strips} strips in any keyframe. "
            "Try --no-sam2 or adjust --n-strips."
        )
    return keyframes


# ── Diff-based fallback (no SAM2) ─────────────────────────────────────────────

def _runs_1d(bool_arr):
    padded = np.concatenate(([False], bool_arr, [False])).astype(np.int8)
    d = np.diff(padded)
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def _boxes_from_threshold(col_profile, threshold, n_strips, dilation=5):
    active = col_profile > threshold
    runs = _runs_1d(active)
    if len(runs) == n_strips:
        return sorted(runs)
    kernel = np.ones(dilation, dtype=np.uint8)
    dilated = np.convolve(active.astype(np.uint8), kernel, mode='same') > 0
    runs = _runs_1d(dilated)
    return sorted(runs) if len(runs) == n_strips else None


def _boxes_top_n(col_profile, n_strips):
    W = len(col_profile)
    k = np.ones(7, dtype=float) / 7
    s = np.convolve(col_profile, k, mode='same')
    half_win = max(15, W // (n_strips * 6))
    peaks = [i for i in range(W)
             if s[i] == s[max(0, i - half_win):min(W, i + half_win + 1)].max()
             and s[i] > s.max() * 0.02]
    if not peaks:
        raise RuntimeError("No peaks in column activity profile.")
    peaks = sorted(peaks, key=lambda i: s[i], reverse=True)[:n_strips]
    peaks.sort()
    intervals = []
    for p in peaks:
        half = s[p] * 0.5
        lo = p
        while lo > 0 and s[lo - 1] >= half:
            lo -= 1
        hi = p + 1
        while hi < W and s[hi] >= half:
            hi += 1
        intervals.append((lo, hi))
    intervals.sort()
    merged = []
    for a, b in intervals:
        if merged and a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def _find_initial_boxes_diff(in_path, width, height, threshold, n_strips):
    frame_nbytes = width * height * 6
    reader = subprocess.Popen(
        ['ffmpeg', '-i', in_path,
         '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-v', 'error', 'pipe:1'],
        stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    col_sum = np.zeros(width, dtype=np.float64)
    prev_frame = None
    n_pairs = 0
    try:
        for _ in range(31):
            raw = reader.stdout.read(frame_nbytes)
            if len(raw) < frame_nbytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint16).reshape(height, width, 3)
            if prev_frame is not None:
                diff = np.abs(frame.astype(np.int32) - prev_frame.astype(np.int32))
                col_sum += diff.max(axis=2).max(axis=0).astype(np.float64)
                n_pairs += 1
            prev_frame = frame.copy()
    finally:
        reader.stdout.close()
        reader.kill()
        reader.wait()

    if n_pairs == 0:
        raise RuntimeError("Could not read frames")
    col_mean = col_sum / n_pairs
    for thresh in [threshold, col_mean.max() * 0.30, col_mean.max() * 0.15, col_mean.max() * 0.05]:
        boxes = _boxes_from_threshold(col_mean, thresh, n_strips)
        if boxes is not None:
            if thresh != threshold:
                print(f"  (used adaptive threshold {thresh:.0f})")
            return boxes
    print("  (threshold approach failed — using peak-based detection)")
    return _boxes_top_n(col_mean, n_strips)


# ── Packing ───────────────────────────────────────────────────────────────────

def pack_frame(frame, boxes, col_widths, gap, out_width, masks=None):
    """Pack LED strips into a black canvas using pixel-precise SAM2 masks.

    If masks is provided (list of (H,W) bool arrays), each strip is cut out in its
    exact shape — non-strip pixels are zeroed.  Falls back to rectangular crop when
    masks is None (diff-based mode).
    """
    height = frame.shape[0]
    canvas = np.zeros((height, out_width, 3), dtype=np.uint16)
    x_cursor = 0
    for i, ((x_start, x_end), cw) in enumerate(zip(boxes, col_widths)):
        copy_w = min(x_end - x_start, cw)
        strip = frame[:, x_start:x_start + copy_w, :].copy()
        if masks is not None:
            strip_mask = masks[i][:, x_start:x_start + copy_w]
            strip[~strip_mask] = 0
        canvas[:, x_cursor:x_cursor + copy_w, :] = strip
        x_cursor += cw + gap
    return canvas


# ── Video processing ──────────────────────────────────────────────────────────

def process_video(in_path, out_path, gap=8, threshold=1000, lock_boxes=False,
                  n_strips=7, keep_audio=True, use_sam2=True, sam2_interval=1):

    device = _detect_device()
    if device is not None:
        print(f"  GPU: {device.type.upper()}")

    stream = _probe_video(in_path)
    if not stream:
        raise RuntimeError("ffprobe failed — is ffmpeg installed?")
    hevc = _pick_hevc_encoder()
    if not hevc:
        raise RuntimeError("No HEVC encoder found — install ffmpeg with libx265")

    width   = stream['width']
    height  = stream['height']
    fps_n, fps_d = map(int, stream['r_frame_rate'].split('/'))
    n_frames = int(stream.get('nb_frames') or 0)
    if n_frames == 0 and stream.get('duration'):
        n_frames = int(float(stream['duration']) * fps_n / fps_d)

    color_space     = stream.get('color_space')
    color_trc       = stream.get('color_transfer')
    color_primaries = stream.get('color_primaries')
    color_range     = stream.get('color_range')

    # ── Stage 1: Background estimation ───────────────────────────────────────
    print("  Estimating background (temporal median)...")
    background_u8 = estimate_background(in_path, width, height, n_frames)

    # ── Stage 2: Load SAM2 ────────────────────────────────────────────────
    sam2_predictor = None
    if use_sam2:
        try:
            print("  Loading SAM2 (facebook/sam2.1-hiera-tiny)...")
            sam2_predictor = load_sam2(device)
        except Exception as e:
            print(f"  (SAM2 unavailable: {e} — falling back to diff-based detection)")

    # ── Stage 3: Initial / keyframe detection ─────────────────────────────
    # keyframes mode: pre-scan, then encode with interp  (sam2_interval > 1 or lock_boxes)
    # inline mode:    SAM2 runs per-frame inside the encode loop (sam2_interval == 1)
    frame_nbytes = width * height * 6

    keyframes     = None   # populated in keyframes mode
    inline_sam2   = False  # per-frame SAM2 inside encode loop
    initial_boxes = None
    initial_masks = None

    if sam2_predictor is not None:
        if lock_boxes:
            print("  Running SAM2 on first frame (lock-boxes)...")
            r0 = subprocess.Popen(
                ['ffmpeg', '-i', in_path, '-vframes', '1',
                 '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-v', 'error', 'pipe:1'],
                stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
            )
            raw0 = r0.stdout.read(frame_nbytes)
            r0.stdout.close(); r0.wait()
            f0 = np.frombuffer(raw0, dtype=np.uint16).reshape(height, width, 3).copy()
            initial_boxes, initial_masks = detect_strips_sam2(
                sam2_predictor, f0, background_u8, None, None, n_strips)
            if initial_boxes is None or len(initial_boxes) != n_strips:
                raise RuntimeError(f"SAM2 could not detect {n_strips} strips in frame 0.")
            keyframes = {0: {'boxes': initial_boxes, 'masks': initial_masks}}
            print(f"  Strips: {initial_boxes}")

        elif sam2_interval == 1:
            # Inline SAM2 — no pre-scan; detect inside the encode loop.
            # Seed prev_boxes/masks from frame 0 so the first frame has a fallback.
            print("  Inline SAM2 mode (per-frame detection)...")
            r0 = subprocess.Popen(
                ['ffmpeg', '-i', in_path, '-vframes', '1',
                 '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-v', 'error', 'pipe:1'],
                stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
            )
            raw0 = r0.stdout.read(frame_nbytes)
            r0.stdout.close(); r0.wait()
            f0 = np.frombuffer(raw0, dtype=np.uint16).reshape(height, width, 3).copy()
            initial_boxes, initial_masks = detect_strips_sam2(
                sam2_predictor, f0, background_u8, None, None, n_strips)
            if initial_boxes is None or len(initial_boxes) != n_strips:
                raise RuntimeError(f"SAM2 could not detect {n_strips} strips in frame 0.")
            inline_sam2 = True
            print(f"  Strips (frame 0): {initial_boxes}")

        else:
            print(f"  SAM2 keyframe scan (every {sam2_interval} frames)...")
            keyframes = _build_keyframes(
                in_path, width, height, background_u8, sam2_predictor,
                n_strips, n_frames, sam2_interval,
            )
            first_kf = keyframes[min(keyframes)]
            initial_boxes = first_kf['boxes']
            initial_masks = first_kf['masks']
            print(f"  Strips (frame 0 keyframe): {initial_boxes}")

    else:
        print("  Detecting strips via temporal diff (no SAM2)...")
        initial_boxes = _find_initial_boxes_diff(in_path, width, height, threshold, n_strips)
        print(f"  Strips: {initial_boxes}")

    # Output dimensions derived from initial detection
    col_widths = [x_end - x_start for x_start, x_end in initial_boxes]
    out_width  = sum(col_widths) + (n_strips - 1) * gap
    out_width += out_width % 2   # libx265 requires even width for 4:2:0
    print(f"  Output: {out_width}×{height}  (gap={gap}px, mask cutouts={'yes' if sam2_predictor else 'no'})")

    # ── Stage 4: Encode loop ──────────────────────────────────────────────
    encoder, pix_fmt_out, enc_extra = hevc
    silent_path = out_path + '.silent.mp4'

    reader = subprocess.Popen(
        ['ffmpeg', '-i', in_path,
         '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-v', 'error', 'pipe:1'],
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

    prev_boxes = initial_boxes
    prev_masks = initial_masks
    frame_idx  = 0
    basename   = os.path.basename(in_path)

    try:
        while True:
            item = decode_q.get()
            if item is _DONE:
                break
            frame = item

            if inline_sam2:
                # Per-frame SAM2 — runs synchronously in this loop
                boxes, masks = detect_strips_sam2(
                    sam2_predictor, frame, background_u8, prev_boxes, prev_masks, n_strips)
                prev_boxes, prev_masks = boxes, masks

            elif keyframes is not None:
                # Pre-computed keyframes with interpolation
                boxes, masks = _interp_state(keyframes, frame_idx)

            else:
                # Diff fallback — fixed boxes, no masks
                boxes, masks = initial_boxes, None

            canvas = pack_frame(frame, boxes, col_widths, gap, out_width, masks=masks)
            encode_q.put(canvas)
            frame_idx += 1
            if frame_idx % 60 == 0:
                if n_frames > 0:
                    print(f"  {basename}: {frame_idx}/{n_frames} frames")
                else:
                    print(f"  {basename}: {frame_idx} frames")
    finally:
        encode_q.put(_DONE)
        encode_thread.join()
        reader.stdout.close()
        reader.wait()
        writer.stdin.close()
        writer.wait()
        decode_thread.join()

    # Remux: stamp HDR metadata + mux audio.
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


def main():
    ap = argparse.ArgumentParser(
        description="Extract LED strips via SAM2 + background subtraction and pack them side-by-side."
    )
    ap.add_argument("input", help="Input video file")
    ap.add_argument("-o", "--output", required=True, help="Output video file")
    ap.add_argument("--gap", type=int, default=8,
                    help="Black pixels between packed strips (default: 8)")
    ap.add_argument("--n-strips", type=int, default=7,
                    help="Number of LED strips to detect (default: 7)")
    ap.add_argument("--lock-boxes", action="store_true",
                    help="Run SAM2 once on frame 0 and reuse those masks for the whole video")
    ap.add_argument("--sam2-interval", type=int, default=1,
                    help="Run SAM2 every N frames and interpolate/shift masks between keyframes "
                         "(default: 1 = per-frame inline). Use 30 for ~1 call/sec at 30 fps.")
    ap.add_argument("--no-sam2", action="store_true",
                    help="Disable SAM2; fall back to temporal-diff column detection (rectangular crops)")
    ap.add_argument("--threshold", type=int, default=1000,
                    help="Diff threshold for --no-sam2 mode, 0-65535 scale (default: 1000)")
    ap.add_argument("--no-audio", action="store_true", help="Skip audio muxing")
    args = ap.parse_args()

    print(f"Processing {args.input} -> {args.output}")
    process_video(
        args.input, args.output,
        gap=args.gap,
        threshold=args.threshold,
        lock_boxes=args.lock_boxes,
        n_strips=args.n_strips,
        keep_audio=not args.no_audio,
        use_sam2=not args.no_sam2,
        sam2_interval=args.sam2_interval,
    )
    print("Done.")


if __name__ == "__main__":
    main()

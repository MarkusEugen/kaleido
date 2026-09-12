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

def estimate_background(in_path, width, height, n_frames_total, n_samples=30):
    """Temporal median background from evenly-spaced frames across the FULL clip → uint8 (H, W, 3).

    Always uses the full video so the arm covers enough different positions for
    the median to converge to the true static background.
    """
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
    """Return n_strips x-positions of local maxima, enforcing a minimum spacing.

    Naive top-N-by-intensity can pick two peaks on a single wide/bright strip
    and miss a dim one nearby.  This finds all local maxima, then greedy-picks
    n_strips of them in intensity order while requiring each new pick to be at
    least `min_spacing` away from every previously-picked peak.  min_spacing is
    relaxed if we can't fit n_strips with the initial constraint.
    """
    W = len(col_profile)
    if W < n_strips:
        return []
    k = np.ones(7, dtype=float) / 7
    s = np.convolve(col_profile.astype(float), k, mode='same')
    threshold = s.max() * 0.02

    # Find all local maxima above threshold with a small suppression window.
    half_win = 3
    candidates = [i for i in range(W)
                  if s[i] == s[max(0, i - half_win):min(W, i + half_win + 1)].max()
                  and s[i] > threshold]
    if len(candidates) < n_strips:
        return []

    candidates.sort(key=lambda i: s[i], reverse=True)   # brightest first
    expected_spacing = W / (n_strips + 1)

    for frac in (0.9, 0.7, 0.5, 0.3, 0.15):
        min_sp = max(3, int(expected_spacing * frac))
        picked = []
        for x in candidates:
            if all(abs(x - p) >= min_sp for p in picked):
                picked.append(x)
                if len(picked) == n_strips:
                    return sorted(picked)

    return sorted(candidates[:n_strips])


def _strip_curves(fg_gray, n_strips, n_bands=7):
    """Bowed centerline per strip, sampled at multiple heights.

    Splits the frame into `n_bands` horizontal slices, finds `n_strips` column
    peaks in each slice, then interpolates a smooth curve from top to bottom
    through those peaks — one curve per strip.

    Returns a (n_strips, H) float array with the per-row x-position of each
    strip's centerline, or None if any band has fewer than n_strips peaks.

    Assumes strips maintain left-to-right ordering along their length, which
    is true for LED strips wrapped around a forearm.
    """
    H, W = fg_gray.shape
    band_h = max(1, H // n_bands)

    band_ys = []
    band_peak_xs = []
    for b in range(n_bands):
        y0 = b * band_h
        y1 = (b + 1) * band_h if b < n_bands - 1 else H
        col_profile = fg_gray[y0:y1, :].max(axis=0)
        peaks = _col_peaks(col_profile, n_strips)
        if len(peaks) < n_strips:
            return None
        band_ys.append((y0 + y1) // 2)
        band_peak_xs.append(peaks)

    band_ys = np.array(band_ys, dtype=np.float32)
    band_peak_xs = np.array(band_peak_xs, dtype=np.float32)   # (n_bands, n_strips)

    curves = np.zeros((n_strips, H), dtype=np.float32)
    ys_all = np.arange(H, dtype=np.float32)
    for s in range(n_strips):
        curves[s] = np.interp(ys_all, band_ys, band_peak_xs[:, s])
    return curves


def _sam2_detect(predictor, fg_u8, curves, height):
    """Run SAM2 with multi-point prompts along each strip's bowed centerline,
    then clip each mask to a curved band that follows the same centerline.

    curves — (n_strips, H) float array of per-row x-positions (from `_strip_curves`)
    Returns (boxes, masks) sorted left→right.
    """
    import torch
    W = fg_u8.shape[1]
    n_strips, H = curves.shape

    # Half-width = 45 % of the minimum inter-strip gap at the middle row
    mid_xs = np.sort(curves[:, H // 2])
    if n_strips > 1:
        min_gap = float(np.min(np.diff(mid_xs)))
        half_w = max(10, int(min_gap * 0.45))
    else:
        half_w = W // 8

    # Three foreground point prompts along each strip's curve (top / middle / bottom).
    prompt_ys = np.array([H // 5, H // 2, 4 * H // 5], dtype=np.int32)
    col_arr = np.arange(W, dtype=np.float32)

    predictor.set_image(fg_u8)
    results = []
    with torch.inference_mode():
        for s in range(n_strips):
            curve = curves[s]                                    # (H,) float
            xs = curve[prompt_ys].astype(np.int32)
            pts = np.stack([xs, prompt_ys], axis=1)              # (3, 2)
            preds, scores, _ = predictor.predict(
                point_coords=pts,
                point_labels=np.array([1, 1, 1]),
                multimask_output=True,
            )
            mask = preds[int(scores.argmax())]                   # (H, W) bool

            # Curved-band clip: per-row window ±half_w around the strip's centerline.
            in_band = np.abs(col_arr[None, :] - curve[:, None]) <= half_w  # (H, W)
            clipped = mask & in_band

            cols = np.where(clipped.any(axis=0))[0]
            if cols.size:
                results.append(((int(cols.min()), int(cols.max()) + 1), clipped))

    results.sort(key=lambda r: r[0][0])
    if results:
        boxes, masks = zip(*results)
        return list(boxes), list(masks)
    return [], []


def detect_strips_sam2(predictor, frame_u16, background_u8, prev_boxes, prev_masks, n_strips):
    """Detect strip boxes + pixel masks using bowed centerlines + SAM2.

    Returns (boxes, masks) if exactly n_strips detected, else (prev_boxes, prev_masks).
    """
    fg = _fg_for_sam2(frame_u16, background_u8)
    fg_gray = fg.max(axis=2)
    curves = _strip_curves(fg_gray, n_strips)
    if curves is None:
        return prev_boxes, prev_masks

    boxes, masks = _sam2_detect(predictor, fg, curves, frame_u16.shape[0])
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
                     n_strips, n_frames, sam2_interval, t_start=0.0, t_duration=None):
    """Pre-scan video at keyframe positions; return {frame_idx: {'boxes':…,'masks':…}}."""
    frame_nbytes = width * height * 6
    seek_pre = ['-ss', f'{t_start:.6f}'] if t_start > 0 else []
    dur_arg  = ['-t',  f'{t_duration:.6f}'] if t_duration is not None else []
    reader = subprocess.Popen(
        ['ffmpeg'] + seek_pre + ['-i', in_path] + dur_arg +
        ['-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-v', 'error', 'pipe:1'],
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


# ── LED blob detection (per-frame, no SAM2) ───────────────────────────────────

def _led_blobs(fg_gray, min_distance=8, threshold=45, sigma=1.5):
    """Local maxima in the foreground → (N, 2) int32 array of (y, x) blob coords."""
    from scipy.ndimage import maximum_filter, gaussian_filter
    smoothed = gaussian_filter(fg_gray, sigma=sigma)
    max_filt = maximum_filter(smoothed, size=min_distance * 2 + 1)
    is_peak = (smoothed == max_filt) & (smoothed > threshold)
    ys, xs = np.where(is_peak)
    if ys.size == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([ys, xs], axis=1).astype(np.int32)


def _bootstrap_peak_xs(fg_gray, blobs, n_strips, min_support=5):
    """Robust initial strip x-positions.

    Uses the COLUMN SUM of the foreground in a tight middle vertical band
    (dominated by long bright vertical runs = real strips, not single bright
    reflections), then requires each peak candidate to have at least
    `min_support` LED blobs within 40 px — real strips have many stacked
    blobs, false peaks don't.  Picks n_strips of the supported candidates
    with a minimum-spacing constraint.
    """
    H, W = fg_gray.shape
    y0, y1 = int(H * 0.35), int(H * 0.70)
    prof = fg_gray[y0:y1, :].sum(axis=0).astype(np.float32)
    prof = np.convolve(prof, np.ones(7, dtype=np.float32) / 7, mode='same')
    if prof.max() <= 0:
        return None
    threshold = prof.max() * 0.05

    hw = 3
    candidates = [i for i in range(W)
                  if prof[i] == prof[max(0, i - hw):min(W, i + hw + 1)].max()
                  and prof[i] > threshold]

    if blobs.shape[0] > 0:
        blob_xs = blobs[:, 1]
        supported = [(c, float(prof[c])) for c in candidates
                     if int(np.sum(np.abs(blob_xs - c) < 40)) >= min_support]
    else:
        supported = [(c, float(prof[c])) for c in candidates]

    if len(supported) < n_strips:
        return None

    supported.sort(key=lambda t: t[1], reverse=True)
    expected_spacing = W / (n_strips + 1)
    for frac in (0.9, 0.7, 0.5, 0.3, 0.15):
        min_sp = max(3, int(expected_spacing * frac))
        picked = []
        for c, _ in supported:
            if all(abs(c - p) >= min_sp for p in picked):
                picked.append(c)
                if len(picked) == n_strips:
                    return sorted(picked)
    return None


def _curves_from_blobs(blobs, fg_gray, n_strips, seed_curves=None, ema_alpha=0.35):
    """Fit a smooth per-row x-position curve to each strip's LED blob chain.

    `seed_curves` — (n_strips, H) previous-frame curves.  Blobs are assigned to
    the strip whose curve is closest at the blob's row.  If None, seeds are
    bootstrapped via `_bootstrap_peak_xs`, which uses column-sum + blob-support
    to reject reflections and background noise.

    Blobs farther than 40 % of the median inter-strip gap from every seed curve
    are rejected as noise.

    If seed_curves came from a previous frame, the result is EMA-smoothed
    against it: curve = α·new + (1−α)·seed.  This eliminates per-frame jitter
    while still letting the curve track the arm's motion within a few frames.

    Returns (n_strips, H) float32 or None if bootstrapping fails.
    """
    H, W = fg_gray.shape
    if blobs.shape[0] < n_strips * 3:
        return None

    from_prev = seed_curves is not None

    if seed_curves is None:
        peak_xs = _bootstrap_peak_xs(fg_gray, blobs, n_strips)
        if peak_xs is None:
            return None
        seed_curves = np.zeros((n_strips, H), dtype=np.float32)
        for s, x in enumerate(peak_xs):
            seed_curves[s, :] = float(x)

    # Max distance for blob-to-strip assignment: 40 % of the median inter-strip gap.
    mid_xs = np.sort(seed_curves[:, H // 2])
    if len(mid_xs) > 1:
        max_dist = float(np.median(np.diff(mid_xs))) * 0.40
    else:
        max_dist = W * 0.10

    # Assign every blob to the strip whose seed curve is closest at that row.
    seed_at_y = seed_curves[:, blobs[:, 0]]              # (n_strips, N)
    dists     = np.abs(seed_at_y - blobs[:, 1][None, :]) # (n_strips, N)
    owner     = dists.argmin(axis=0)                     # (N,)
    keep      = dists.min(axis=0) < max_dist             # reject noise blobs

    curves = np.zeros((n_strips, H), dtype=np.float32)
    ys_all = np.arange(H, dtype=np.float32)
    kernel = np.ones(31, dtype=np.float32) / 31.0
    for s in range(n_strips):
        pts = blobs[keep & (owner == s)]
        if pts.shape[0] < 3:
            curves[s] = seed_curves[s]
            continue
        order = np.argsort(pts[:, 0])
        ys, xs = pts[order, 0].astype(np.float32), pts[order, 1].astype(np.float32)
        curves[s] = np.interp(ys_all, ys, xs)
        curves[s] = np.convolve(curves[s], kernel, mode='same')

    if from_prev:
        curves = ema_alpha * curves + (1.0 - ema_alpha) * seed_curves
    return curves


def _mask_from_curve(curve, W, half_w):
    """(H, W) bool mask: True where col is within ±half_w of the curve at that row."""
    col_arr = np.arange(W, dtype=np.float32)
    return np.abs(col_arr[None, :] - curve[:, None]) <= half_w


def detect_strips_blobs(frame_u16, background_u8, prev_curves, n_strips, half_w=None):
    """Per-frame strip detection via LED blob tracking.

    Returns (boxes, masks, curves, half_w) on success, or
    (None, None, prev_curves, half_w) if blob detection failed for this frame.
    """
    height, width = frame_u16.shape[:2]
    fg = _fg_for_sam2(frame_u16, background_u8)
    fg_gray = fg.max(axis=2)

    blobs = _led_blobs(fg_gray)
    curves = _curves_from_blobs(blobs, fg_gray, n_strips, seed_curves=prev_curves)
    if curves is None:
        return None, None, prev_curves, half_w

    if half_w is None:
        mid_xs = np.sort(curves[:, height // 2])
        min_gap = float(np.min(np.diff(mid_xs)))
        half_w = max(15, int(min_gap * 0.45))

    boxes = []
    masks = []
    for s in range(n_strips):
        mask = _mask_from_curve(curves[s], width, half_w)
        cols = np.where(mask.any(axis=0))[0]
        if cols.size == 0:
            return None, None, prev_curves, half_w
        boxes.append((int(cols.min()), int(cols.max()) + 1))
        masks.append(mask)

    order = sorted(range(n_strips), key=lambda i: boxes[i][0])
    boxes = [boxes[i] for i in order]
    masks = [masks[i] for i in order]
    curves = curves[order]
    return boxes, masks, curves, half_w


# ── Packing ───────────────────────────────────────────────────────────────────

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

    With masks: each strip is centerline-warped straight (no curved edges).
    Without masks (diff-based mode): rectangular crop.
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

def process_video(in_path, out_path, gap=8, threshold=1000, lock_boxes=False,
                  n_strips=7, keep_audio=True, use_sam2=True, sam2_interval=1,
                  start_frame=None, end_frame=None, blob_detect=False):

    device = _detect_device()
    if device is not None and not blob_detect:
        print(f"  GPU: {device.type.upper()}")
    if blob_detect:
        use_sam2 = False   # blob mode replaces SAM2 entirely

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
    n_frames   = ef - sf          # frames to process
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

    # ── Stage 1: Background estimation ───────────────────────────────────────
    # Always scans the FULL clip so the arm covers enough positions for the
    # temporal median to converge to the true static background.
    print("  Estimating background (temporal median, full clip)...")
    background_u8 = estimate_background(in_path, width, height, n_frames_total)

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

    blob_curves = None
    blob_half_w = None
    if blob_detect:
        print("  Blob detection mode — per-frame LED tracking (no SAM2)...")
        r0 = subprocess.Popen(
            ['ffmpeg'] + seek_pre + ['-i', in_path, '-vframes', '1',
             '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-v', 'error', 'pipe:1'],
            stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
        )
        raw0 = r0.stdout.read(frame_nbytes)
        r0.stdout.close(); r0.wait()
        f0 = np.frombuffer(raw0, dtype=np.uint16).reshape(height, width, 3).copy()
        initial_boxes, initial_masks, blob_curves, blob_half_w = detect_strips_blobs(
            f0, background_u8, None, n_strips)
        if initial_boxes is None:
            raise RuntimeError(
                f"Blob detection could not find {n_strips} strips in frame 0. "
                "Check that the LED strips are lit."
            )
        print(f"  Strips (frame 0): {initial_boxes}  half_w={blob_half_w}")

    elif sam2_predictor is not None:
        if lock_boxes:
            print("  Running SAM2 on first frame (lock-boxes)...")
            r0 = subprocess.Popen(
                ['ffmpeg'] + seek_pre + ['-i', in_path, '-vframes', '1',
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
                ['ffmpeg'] + seek_pre + ['-i', in_path, '-vframes', '1',
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
                t_start=t_start, t_duration=t_duration,
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

            if blob_detect:
                # Per-frame LED blob tracking — no SAM2, curves re-fit each frame.
                boxes, masks, blob_curves, blob_half_w = detect_strips_blobs(
                    frame, background_u8, blob_curves, n_strips, half_w=blob_half_w)
                if boxes is None:
                    boxes, masks = prev_boxes, prev_masks
                else:
                    prev_boxes, prev_masks = boxes, masks

            elif inline_sam2:
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
                    print(f"  {basename}: {frame_idx}/{n_frames} frames ({int(frame_idx/n_frames*100)}%)")
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
    ap.add_argument("--start-frame", type=int, default=None,
                    help="First frame to process (0-based, inclusive; default: 0)")
    ap.add_argument("--end-frame", type=int, default=None,
                    help="Last frame to process (exclusive; default: end of clip)")
    ap.add_argument("--blob-detect", action="store_true",
                    help="Track individual LED positions per frame and fit a bowed curve "
                         "through each strip's dots — no SAM2, adapts every frame.")
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
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        blob_detect=args.blob_detect,
    )
    print("Done.")


if __name__ == "__main__":
    main()

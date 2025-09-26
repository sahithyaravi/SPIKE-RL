# pip install opencv-python-headless numpy scikit-learn
import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple



def iter_video_frames(video_path: str, max_frames: int = None, stride: int = 1) -> Tuple[List[np.ndarray], float]:
    """
    Read frames with OpenCV (BGR). Returns (frames, fps).
    Set `stride` > 1 to downsample. Set `max_frames` to cap runtime.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames, fps



def nms_topk_1d(scores: np.ndarray, k: int, radius: int = 5) -> List[int]:
    """
    Greedy 1D non-maximum suppression.
    - scores: shape [T]
    - k: how many peaks to keep
    - radius: suppress neighbors within ±radius of a chosen peak
    Returns selected indices sorted in ascending order.
    """
    scores = scores.copy()
    selected = []
    for _ in range(min(k, len(scores))):
        i = int(np.argmax(scores))
        if scores[i] <= -np.inf:
            break
        selected.append(i)
        # suppress neighbors
        left, right = max(0, i - radius), min(len(scores), i + radius + 1)
        scores[left:right] = -np.inf
    return sorted(selected)

def _rgb_hist(frame: np.ndarray, bins: int = 8) -> np.ndarray:
    """Compute normalized RGB histogram (concatenated per channel)."""
    hists = []
    for c in range(3):  # BGR in OpenCV; we'll keep it as-is consistently
        hist = cv2.calcHist([frame], [c], None, [bins], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-8)
        hists.append(hist)
    return np.concatenate(hists, axis=0)

# ------------------------- 1) Histogram SBD -------------------------
def select_histogram_sbd_proportional(
    video_path: str,
    B: int,
    bins: int = 8,
    metric: str = "l1",         # 'l1' or 'chi2'
    stride: int = 1,
    n_peaks: int = 16,
    nms_radius: int = 6,
    window_radius: int = 3,
    temperature: float = 1.0,   # <1 spreads; >1 concentrates
    smooth: int = 3,            # 1 disables smoothing
    max_frames: int = None,
    placement: str = "center"   # or "random"
):
    """
    Single-function Histogram SBD with proportional allocation:
      1) per-step RGB histogram change (L1 or Chi^2),
      2) NMS to get salient peaks,
      3) allocate exactly B frames ∝ peak magnitudes (largest remainder),
      4) place frames around each peak within ±window_radius.

    Returns: list of selected *downsampled* frame indices (0-based).
    """
    # --- read frames (downsample with stride) ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(f)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()

    T = len(frames)
    if T == 0:
        return []
    if T == 1:
        return [0] * min(B, 1)

    # --- per-frame RGB histograms + change scores ---
    def rgb_hist(frame):
        hcat = []
        for c in range(3):  # OpenCV BGR order; consistent use is fine
            h = cv2.calcHist([frame], [c], None, [bins], [0, 256]).ravel()
            h = h / (h.sum() + 1e-8)
            hcat.append(h)
        return np.concatenate(hcat, axis=0)

    hprev = rgb_hist(frames[0])
    scores = [0.0]
    for t in range(1, T):
        h = rgb_hist(frames[t])
        if metric == "chi2":
            denom = (h + hprev + 1e-8)
            s = 0.5 * np.sum(((h - hprev) ** 2) / denom)
        else:  # 'l1'
            s = np.sum(np.abs(h - hprev))
        scores.append(float(s))
        hprev = h
    scores = np.asarray(scores, dtype=float)

    # --- optional smoothing ---
    if smooth and smooth > 1:
        pad = smooth // 2
        kernel = np.ones(smooth, dtype=float) / smooth
        scores = np.convolve(np.pad(scores, (pad, pad), mode='edge'), kernel, mode='valid')

    # --- NMS to keep up to n_peaks peaks ---
    s = scores.copy()
    peaks = []
    for _ in range(min(n_peaks, len(s))):
        j = int(np.argmax(s))
        if not np.isfinite(s[j]) or s[j] <= -np.inf:
            break
        peaks.append(j)
        L, R = max(0, j - nms_radius), min(len(s), j + nms_radius + 1)
        s[L:R] = -np.inf
    peaks = sorted(peaks)

    if len(peaks) == 0:
        # fallback: uniform across timeline
        return np.linspace(0, T - 1, B, dtype=int).tolist()

    # --- temperature + largest-remainder allocation to sum exactly B ---
    w = scores[peaks].astype(float)
    temperature = max(1e-6, float(temperature))
    w = np.power(np.maximum(0.0, w), 1.0 / temperature)

    if w.sum() == 0:
        alloc = np.full(len(peaks), B // len(peaks), dtype=int)
        alloc[:B % len(peaks)] += 1
    else:
        p = w / w.sum()
        raw = p * B
        alloc = np.floor(raw).astype(int)
        r = B - alloc.sum()
        if r > 0:
            frac_order = np.argsort(-(raw - alloc))
            for idx in frac_order[:r]:
                alloc[idx] += 1

    # --- place frames around peaks within ±window_radius ---
    selected = []
    for peak, c in zip(peaks, alloc):
        if c <= 0:
            continue
        if placement == "random":
            low, high = max(0, peak - window_radius), min(T - 1, peak + window_radius)
            selected.extend(np.random.randint(low, high + 1, size=c).tolist())
        else:  # "center": symmetric spread
            offsets, k = [], 0
            while len(offsets) < c:
                if k == 0:
                    offsets.append(0)
                else:
                    if len(offsets) < c: offsets.append(+k)
                    if len(offsets) < c: offsets.append(-k)
                k += 1
                if k > window_radius and len(offsets) < c:
                    while len(offsets) < c:
                        offsets.append(window_radius)
            for off in offsets:
                j = int(np.clip(peak + off, 0, T - 1))
                selected.append(j)

    # --- ensure exactly B frames (pad/truncate) ---
    if len(selected) < B:
        extra = np.linspace(0, T - 1, B, dtype=int).tolist()
        selected = (selected + extra)[:B]
    elif len(selected) > B:
        selected = selected[:B]

    return sorted(selected)


# ------------------------- 2) Edges Change Ratio (ECR) -------------------------

def select_ecr_proportional(
    video_path: str,
    B: int,
    canny_low: int = 100,
    canny_high: int = 200,
    stride: int = 1,
    n_peaks: int = 16,
    nms_radius: int = 6,
    window_radius: int = 3,
    temperature: float = 1.0,   # <1 spreads; >1 concentrates on strong peaks
    smooth: int = 3,            # 1 disables smoothing
    max_frames: int = None,
    placement: str = "center"   # or "random"
):
    """
    ECR with proportional allocation (guarantees exactly B frames):
      1) Compute per-step ECR_t = max(|E_t\E_{t-1}|/|E_t|, |E_{t-1}\E_t|/|E_{t-1}|).
      2) NMS to keep up to n_peaks salient change points.
      3) Allocate B frames ∝ peak magnitudes (largest remainder).
      4) Place frames within ±window_radius around each peak.

    Returns: list of selected *downsampled* frame indices (0-based).
    """
    # --- read frames with stride ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(f)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()

    T = len(frames)
    if T == 0:
        return []
    if T == 1:
        return [0] * min(B, 1)

    # --- edge maps & ECR scores ---
    def edges(im):
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        e = cv2.Canny(g, canny_low, canny_high)
        return (e > 0).astype(np.uint8)

    e_prev = edges(frames[0])
    scores = [0.0]
    for t in range(1, T):
        e = edges(frames[t])
        add = np.logical_and(e == 1, e_prev == 0).sum()   # edges that appeared
        rem = np.logical_and(e_prev == 1, e == 0).sum()   # edges that disappeared
        s1 = add / max(int(e.sum()), 1)
        s2 = rem / max(int(e_prev.sum()), 1)
        scores.append(float(max(s1, s2)))
        e_prev = e
    scores = np.asarray(scores, dtype=float)

    # --- optional smoothing ---
    if smooth and smooth > 1:
        pad = smooth // 2
        kernel = np.ones(smooth, dtype=float) / smooth
        scores = np.convolve(np.pad(scores, (pad, pad), mode='edge'), kernel, mode='valid')

    # --- NMS peaks ---
    s = scores.copy()
    peaks = []
    for _ in range(min(n_peaks, len(s))):
        j = int(np.argmax(s))
        if not np.isfinite(s[j]) or s[j] <= -np.inf:
            break
        peaks.append(j)
        L, R = max(0, j - nms_radius), min(len(s), j + nms_radius + 1)
        s[L:R] = -np.inf
    peaks = sorted(peaks)

    if len(peaks) == 0:
        # fallback: uniform over timeline
        return np.linspace(0, T - 1, B, dtype=int).tolist()

    # --- temperature & largest-remainder allocation (sum exactly B) ---
    w = scores[peaks].astype(float)
    temperature = max(1e-6, float(temperature))
    w = np.power(np.maximum(0.0, w), 1.0 / temperature)

    if w.sum() == 0:
        alloc = np.full(len(peaks), B // len(peaks), dtype=int)
        alloc[:B % len(peaks)] += 1
    else:
        p = w / w.sum()
        raw = p * B
        alloc = np.floor(raw).astype(int)
        r = B - alloc.sum()
        if r > 0:
            frac_order = np.argsort(-(raw - alloc))
            for idx in frac_order[:r]:
                alloc[idx] += 1

    # --- place frames around peaks within ±window_radius ---
    selected = []
    for peak, c in zip(peaks, alloc):
        if c <= 0:
            continue
        if placement == "random":
            lo, hi = max(0, peak - window_radius), min(T - 1, peak + window_radius)
            selected.extend(np.random.randint(lo, hi + 1, size=c).tolist())
        else:  # "center": symmetric spread
            offsets, k = [], 0
            while len(offsets) < c:
                if k == 0:
                    offsets.append(0)
                else:
                    if len(offsets) < c: offsets.append(+k)
                    if len(offsets) < c: offsets.append(-k)
                k += 1
                if k > window_radius and len(offsets) < c:
                    while len(offsets) < c:
                        offsets.append(window_radius)
            for off in offsets:
                j = int(np.clip(peak + off, 0, T - 1))
                selected.append(j)

    # --- ensure exactly B frames (pad/truncate) ---
    if len(selected) < B:
        extra = np.linspace(0, T - 1, B, dtype=int).tolist()
        selected = (selected + extra)[:B]
    elif len(selected) > B:
        selected = selected[:B]

    return sorted(selected)

# ------------------------- 3) Optical-Flow Motion -------------------------
def _flow_magnitudes(gray_frames, use_tvl1=False):
    mags = [0.0]
    if use_tvl1:
        tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
        for t in range(1, len(gray_frames)):
            flow = tvl1.calc(gray_frames[t-1], gray_frames[t], None)
            mags.append(float(np.linalg.norm(flow, axis=2).mean()))
    else:
        for t in range(1, len(gray_frames)):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[t-1], gray_frames[t], None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.1, flags=0
            )
            mags.append(float(np.linalg.norm(flow, axis=2).mean()))
    return np.asarray(mags)

def _nms_peaks(scores, k, radius=6):
    s = scores.copy()
    peaks = []
    for _ in range(min(k, len(s))):
        i = int(np.argmax(s))
        if not np.isfinite(s[i]) or s[i] <= -np.inf:
            break
        peaks.append(i)
        L, R = max(0, i - radius), min(len(s), i + radius + 1)
        s[L:R] = -np.inf
    return sorted(peaks)

def _largest_remainder_alloc(weights, B):
    """Deterministic: floor then give leftovers to largest fractional parts."""
    w = np.maximum(0.0, np.asarray(weights, dtype=float))
    if w.sum() == 0:
        # uniform fallback
        base = np.full(len(w), B // len(w), dtype=int)
        base[:B % len(w)] += 1
        return base
    p = w / w.sum()
    raw = p * B
    alloc = np.floor(raw).astype(int)
    r = B - alloc.sum()
    if r > 0:
        frac_order = np.argsort(-(raw - alloc))  # descending fractional parts
        for i in frac_order[:r]:
            alloc[i] += 1
    return alloc  # sums to B

def _place_frames_around_peaks(T, peaks, counts, window_radius=3, mode="center"):
    """
    For each peak i with counts[i]=c, place c indices around peak within ±window_radius.
    mode="center": symmetric spread; mode="random": uniform random in window.
    """
    selected = []
    for peak, c in zip(peaks, counts):
        if c <= 0: 
            continue
        if mode == "random":
            low, high = max(0, peak - window_radius), min(T - 1, peak + window_radius)
            # sample with replacement to honor counts (can dedup later if desired)
            idxs = np.random.randint(low, high + 1, size=c).tolist()
            selected.extend(idxs)
        else:
            # symmetric placement
            offsets = []
            k = 0
            while len(offsets) < c:
                if k == 0:
                    offsets.append(0)
                else:
                    if len(offsets) < c: offsets.append(+k)
                    if len(offsets) < c: offsets.append(-k)
                k += 1
                if k > window_radius and len(offsets) < c:
                    # pad at window edges if we ran out of space
                    while len(offsets) < c:
                        offsets.append(window_radius)
            for off in offsets:
                j = int(np.clip(peak + off, 0, T - 1))
                selected.append(j)
    # If duplicates matter to you, you can keep them (multiple frames from same region)
    # or enforce uniqueness and then refill—here we keep as-is to honor allocation.
    return sorted(selected)

def select_optical_flow_proportional(
    video_path: str,
    B: int,
    stride: int = 1,
    n_peaks: int = 16,
    nms_radius: int = 6,
    window_radius: int = 3,
    temperature: float = 1.0,
    use_tvl1: bool = False,
    smooth: int = 2,
    placement: str = "center",  # or "random"
    max_frames: int = None
):
    """
    1) Compute per-step flow magnitude scores.
    2) NMS to get up to `n_peaks` motion peaks.
    3) Allocate exactly B frames proportional to peak magnitudes (largest remainder).
    4) Place frames around each peak within ±window_radius.
    Returns: list of selected *downsampled* frame indices (0-based).
    """
    # Read frames with stride
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")
    frames = []
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()

    T = len(frames)
    if T == 0:
        return []
    if T == 1:
        return [0] * min(B, 1)

    scores = _flow_magnitudes(frames, use_tvl1=use_tvl1)  # shape [T]
    # optional smoothing
    if smooth and smooth > 1:
        pad = smooth // 2
        kernel = np.ones(smooth) / smooth
        scores = np.convolve(np.pad(scores, (pad, pad), mode='edge'), kernel, mode='valid')

    # Find motion peaks (indices in [0..T-1])
    peaks = _nms_peaks(scores, k=min(n_peaks, T), radius=nms_radius)
    if len(peaks) == 0:
        # fallback: uniform across timeline
        return np.linspace(0, T - 1, B, dtype=int).tolist()

    # Weight by (temperatured) scores at peaks
    w = scores[peaks].astype(float)
    if temperature <= 0:
        temperature = 1.0
    w = np.power(np.maximum(0.0, w), 1.0 / temperature)

    counts = _largest_remainder_alloc(w, B)  # sums to B
    selected = _place_frames_around_peaks(T, peaks, counts, window_radius=window_radius, mode=placement)

    # (Optional) ensure exactly B by padding/truncating:
    if len(selected) < B:
        # top up uniformly (or by global scores)
        extra = np.linspace(0, T - 1, B, dtype=int).tolist()
        selected = sorted((selected + extra)[:B])
    elif len(selected) > B:
        selected = selected[:B]

    return selected
# ------------------------- 4) Katna-style clustering -------------------------

def select_katna_kmeans(video_path: str, B: int, bins: int = 16,
                        stride: int = 1, max_frames: int = None, random_state: int = 0) -> List[int]:
    """
    Build per-frame color histograms -> KMeans with k=B -> pick frame nearest each centroid.
    """
    frames, _ = iter_video_frames(video_path, max_frames=max_frames, stride=stride)
    if len(frames) == 0:
        return []
    # feature per frame
    feats = np.stack([_rgb_hist(f, bins=bins) for f in frames], axis=0)  # [T, D]
    k = min(B, len(frames))
    if k == 0:
        return []
    if k == len(frames):
        return list(range(len(frames)))

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(feats)
    centers = km.cluster_centers_

    # for each cluster, pick the closest frame (ties -> earliest)
    selected = []
    for c in range(k):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        sub = feats[idxs]
        dists = np.linalg.norm(sub - centers[c][None, :], axis=1)
        best_local = idxs[int(np.argmin(dists))]
        selected.append(int(best_local))

    return sorted(selected)

def extract_frames(video_path, B=64, mode="motion"):
    if mode == "hist":
        print("Using Histogram SBD for keyframe selection", B)
        idx = select_histogram_sbd_proportional(video_path, B=B, bins=8, stride=2)
    elif mode == "ecr":
        print("Using Edges Change Ratio for keyframe selection", B)
        idx  = select_ecr_proportional(video_path, B, stride=2)
    elif mode == "motion":
        print("Using optical flow for keyframe selection", B)
        idx = select_optical_flow_proportional(video_path=video_path, B=B, use_tvl1=False, stride=2, smooth=5)
    else:
        print("Using Katna k-means for keyframe selection", B)
        idx   = select_katna_kmeans(video_path, B, bins=16, stride=2)
    return idx


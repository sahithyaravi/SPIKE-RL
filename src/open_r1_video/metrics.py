from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np


def precision_recall_hit_at_k(surprise_scores, frame_indices, amusing_ids, k=5, threshold: float = 0.05):
    # Gate by threshold first: ignore near-zero scores
    scores_arr = np.asarray(surprise_scores, dtype=float)
    valid = np.flatnonzero(scores_arr >= threshold)
    if valid.size == 0:
        return 0.0, 0.0, 0

    # Rank only among gated indices
    top_k_indices = valid[np.argsort(scores_arr[valid])[::-1][:k]]

    top_k_frame_ids = [frame_indices[i] for i in top_k_indices]
    amusing_set = set(amusing_ids)
    hits = [fid for fid in top_k_frame_ids if fid in amusing_set]

    precision = len(hits) / k
    recall = len(hits) / len(amusing_set) if amusing_set else 0.0
    hit = 1 if hits else 0
    return precision, recall, hit

def accuracy_at_delta_t(surprise_scores, frame_indices, amusing_ids, fps=30, delta_t=0.25):
    """
    Oops-style temporal localization accuracy.
    Uses the single most surprising frame (argmax of surprise_scores) as the prediction,
    and returns 1.0 if that predicted frame lies within ±delta_t seconds of ANY ground-truth
    amusing frame; otherwise 0.0.

    Args:
        surprise_scores: list/np.array of surprise scores, one per frame.
        frame_indices: list of frame indices corresponding to surprise_scores.
        amusing_ids: list or set of ground-truth amusing frame indices.
        fps: frames-per-second of the video.
        delta_t: time tolerance in seconds (e.g., 0.25 or 1.0).

    Returns:
        float: 1.0 if correct within tolerance, else 0.0
    """
    if len(surprise_scores) == 0 or len(frame_indices) == 0:
        return 0.0
    if len(surprise_scores) != len(frame_indices):
        raise ValueError("surprise_scores and frame_indices must have the same length.")
    if amusing_ids is None or len(amusing_ids) == 0:
        return 0.0

    pred_idx = frame_indices[int(np.argmax(surprise_scores))]
    tolerance_frames = int(round(delta_t * fps))
    amusing_set = set(int(x) for x in amusing_ids)

    hit = any(abs(pred_idx - gt) <= tolerance_frames for gt in amusing_set)
    return float(hit)

    
def iou_temporal(surprise_scores, frame_indices, amusing_ids, fps=30, half_window=0.5):
    """
    FunQA-style IoU for timestamp localization using windows around onsets.
    We convert the predicted onset (top-1 by surprise) into a segment by taking
    a symmetric window ±half_window seconds around it. We do the same for each
    ground-truth onset. The returned score is the MAX IoU over all GT onsets.

    NOTE: If your dataset provides ground-truth *spans*, replace the GT window
    construction below with the true [start,end] (in frames) and compute IoU directly.

    Args:
        surprise_scores: list/np.array of surprise scores, one per frame.
        frame_indices: list of frame indices corresponding to surprise_scores.
        amusing_ids: list or set of ground-truth amusing frame indices.
        fps: frames-per-second of the video.
        half_window: seconds for half the window length (segment length = 2 * half_window).

    Returns:
        float: best IoU in [0,1] between predicted window and any GT window.
    """
    if len(surprise_scores) == 0 or len(frame_indices) == 0:
        return 0.0
    if len(surprise_scores) != len(frame_indices):
        raise ValueError("surprise_scores and frame_indices must have the same length.")
    if amusing_ids is None or len(amusing_ids) == 0:
        return 0.0

    # Build predicted window from top-1 frame
    pred_center = frame_indices[int(np.argmax(surprise_scores))]
    hw = int(round(half_window * fps))
    pred_start = pred_center - hw
    pred_end   = pred_center + hw

    if pred_end <= pred_start:
        return 0.0  # degenerate window

    best_iou = 0.0
    for gt_center in set(int(x) for x in amusing_ids):
        gt_start = gt_center - hw
        gt_end   = gt_center + hw
        if gt_end <= gt_start:
            continue

        inter = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
        union = (pred_end - pred_start) + (gt_end - gt_start) - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou

    return float(best_iou)




# def _iou_time(a0, a1, b0, b1):
#     """Temporal IoU between [a0,a1) and [b0,b1) in seconds."""
#     if a1 <= a0 or b1 <= b0:
#         return 0.0
#     inter = max(0.0, min(a1, b1) - max(a0, b0))
#     union = (a1 - a0) + (b1 - b0) - inter
#     return float(inter / union) if union > 0 else 0.0

# def _gt_window_from_ids(amusing_ids, fps, pad_sec=0.0):
#     """
#     Collapse GT ids into one window in seconds.
#     Uses +1 frame on the right to make it half-open like [start, end).
#     """
#     gt_start_frame = int(min(amusing_ids))
#     gt_end_frame   = int(max(amusing_ids)) + 1
#     pad_frames = int(round(pad_sec * fps))
#     gt_start_frame -= pad_frames
#     gt_end_frame   += pad_frames
#     gt_start_sec = gt_start_frame / float(fps)
#     gt_end_sec   = gt_end_frame   / float(fps)
#     return gt_start_sec, gt_end_sec

# def _best_sum_window_seconds(times_sec, scores, length_sec):
#     """
#     Two-pointer scan to find the length_sec window with max total surprise.
#     Returns (win_start_sec, win_end_sec, best_sum, left_idx, right_idx_inclusive).
#     """
#     n = len(times_sec)
#     if n == 0 or length_sec <= 0:
#         return (0.0, 0.0, 0.0, -1, -1)

#     # sort by time
#     order = np.argsort(times_sec)
#     t = np.asarray(times_sec, dtype=float)[order]
#     s = np.asarray(scores,      dtype=float)[order]

#     best_sum = -np.inf
#     best_l = best_r = 0
#     cur_sum = 0.0
#     l = 0
#     for r in range(n):
#         cur_sum += s[r]
#         # maintain window length: t[r] - t[l] <= length_sec
#         while t[r] - t[l] > length_sec:
#             cur_sum -= s[l]
#             l += 1
#         if cur_sum > best_sum:
#             best_sum = cur_sum
#             best_l, best_r = l, r

#     # set predicted window start at the left edge of the best run
#     pred_start = t[best_l]
#     pred_end   = pred_start + length_sec
#     return (pred_start, pred_end, float(best_sum), int(best_l), int(best_r))

# def compute_iou_coverage(
#     surprise_scores,
#     frame_indices,
#     amusing_ids,
#     fps=30,
#     length_sec=None,       # e.g., 0.25 or 0.5
#     match_gt_length=True, # if True, ignore length_sec and use GT span length
#     gt_pad_sec=0.0
# ):
#     """
#     Finds the fixed-duration window (in seconds) with maximum total surprise,
#     then computes IoU vs. GT window collapsed from amusing_ids.

#     Assumes frame_indices are global frame numbers at `fps` (not necessarily consecutive).
#     """
#     if not surprise_scores or not frame_indices or not amusing_ids:
#         return 0.0
#     if len(surprise_scores) != len(frame_indices):
#         raise ValueError("surprise_scores and frame_indices must have the same length.")
#     if fps <= 0:
#         raise ValueError("fps must be positive.")

#     # Build GT window in seconds
#     gt_start, gt_end = _gt_window_from_ids(amusing_ids, fps, pad_sec=gt_pad_sec)
#     gt_len = max(0.0, gt_end - gt_start)
#     if match_gt_length:
#         length = gt_len
#     else:
#         if length_sec is None or length_sec <= 0:
#             raise ValueError("Provide a positive length_sec or set match_gt_length=True.")
#         length = float(length_sec)

#     # Convert sampled frames to timestamps in seconds
#     times_sec = np.asarray(frame_indices, dtype=float) / float(fps)

#     # Best-scoring fixed-duration window
#     pred_start, pred_end, _, _, _ = _best_sum_window_seconds(times_sec, surprise_scores, length)

#     # IoU in time
#     return _iou_time(pred_start, pred_end, gt_start, gt_end)

def get_surprise_windows(surprise_scores, frame_indices, fps=30):
    """
    Find contiguous windows where surprise scores > 0.
    Returns list of (start_time, end_time) tuples.
    """
    windows = []
    window_start = None
    
    for i, score in enumerate(surprise_scores):
        if score > 0 and window_start is None:
            # Start new window
            window_start = frame_indices[i]
        elif score == 0 and window_start is not None:
            # End current window
            window_end = frame_indices[i-1]
            start_time = window_start / fps
            end_time = window_end / fps
            windows.append((start_time, end_time))
            window_start = None
    
    # Handle window that goes to the end
    if window_start is not None:
        window_end = frame_indices[-1]
        start_time = window_start / fps
        end_time = window_end / fps
        windows.append((start_time, end_time))
    
    return windows

def window_iou(pred_windows, gt_frames, fps=30):
    """
    Check if GT frames fall inside predicted windows.
    Returns coverage (fraction of GT frames covered).
    """
    if not pred_windows or not gt_frames:
        return 0.0
    
    gt_times = [frame / fps for frame in gt_frames]
    covered = 0
    
    for gt_time in gt_times:
        for start_time, end_time in pred_windows:
            if start_time <= gt_time <= end_time:
                covered += 1
                break
    
    return covered / len(gt_frames)

def compute_iou_coverage(surprise_scores, frame_indices, amusing_ids, fps=30):
    """
    Compute IoU coverage metric based on surprise score windows.
    """
    pred_windows = get_surprise_windows(surprise_scores, frame_indices, fps)
    coverage = window_iou(pred_windows, amusing_ids, fps)
    return coverage
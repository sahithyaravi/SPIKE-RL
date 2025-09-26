

from typing import List, Sequence, Callable, Optional, Dict, Any, Tuple, Union
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)


def _text_embed(texts: List[str]) -> np.ndarray:
    vecs = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vecs

def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    if x.size == 0:
        return x
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def _cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    A = _l2_normalize(A, axis=1)
    B = _l2_normalize(B, axis=1)
    return A @ B.T

def plausibility_per_window(
    hypotheses_per_window,
    memory_evolution,
):
    T = len(hypotheses_per_window)

    # Encode memory
    M = _text_embed(list(memory_evolution))  # [T, D]
    out = []
    for t in range(T):
        H_t = hypotheses_per_window[t]
        H = _text_embed(list(H_t))
        if H.size == 0 or M.shape[0] <= t or M.shape[1] == 0:
            out.append(np.nan)
            continue
        sims = _cosine_matrix(H, M[t:t+1, :])
        out.append(float(sims.max()) if sims.size else np.nan)
    return out


def diversity_per_window(
    hypotheses_per_window,
) -> List[float]:
    out: List[float] = []
    for H_t in hypotheses_per_window:
        H = _text_embed(list(H_t))
        K = H.shape[0]
        S = _cosine_matrix(H, H) 
        iu = np.triu_indices(K, k=1)
        cos_ij = S[iu]
        div = np.mean(1.0 - cos_ij) if cos_ij.size else np.nan
        out.append(float(div))
    return out


def belief_shift_quality(
    hypotheses_per_window,
    captions,
) :

    T = len(hypotheses_per_window)

    # Encode captions
    C = _text_embed(list(captions))  # [T', D]
    out: List[float] = []
    for t in range(T):
        # Need c_t and c_{t+1}
        if C.shape[0] <= t or C.shape[0] <= t + 1 or C.shape[1] == 0:
            out.append(np.nan)
            continue

        c_t = C[t:t+1, :]       # [1, D]
        c_tp1 = C[t+1:t+2, :]   # [1, D]

        base = float(_cosine_matrix(c_t, c_tp1)[0, 0])
        H_t = hypotheses_per_window[t]
        H = _text_embed(list(H_t))
        sims = _cosine_matrix(H, c_tp1)[:, 0]  # [K_t]
        bestH = float(sims.max()) if sims.size else np.nan
        
        # take modulus to avoid negative values
        out.append(abs(bestH - base) if bestH == bestH else np.nan)
    return out

def compute_hyp_metrics_from_record(record, window_size=4):

    hypotheses_per_window = record.get("explanations") or record.get("Explanations") or []
    captions_per_window = record.get("captions") or record.get("Captions") or []
    memory_evolution = record.get("memory_evolution") or record.get("Memory_Evolution") or []

    # if captions_per_window is empty, use memory_evolutions last sentence as caption
    if len(captions_per_window) == 0 and len(memory_evolution) > 0:
        captions_per_window = [me.split(". ")[-1] for me in memory_evolution]

    pl = plausibility_per_window(hypotheses_per_window[window_size:], memory_evolution[window_size:])
    dv = diversity_per_window(hypotheses_per_window[window_size:])
    bs = belief_shift_quality(hypotheses_per_window[window_size:], captions_per_window[window_size:])

    # find average
    pl = np.nanmean(pl) if len(pl) > 0 else np.nan
    dv = np.nanmean(dv) if len(dv) > 0 else np.nan
    bs = np.nanmean(bs) if len(bs) > 0 else np.nan
    return {
        "plausibility": pl,
        "diversity": dv,
        "bsq": bs,
    }

if __name__ == "__main__":
    # Self-test with synthetic 3 windows, 3 hypotheses each, using random embeddings.
    record = {"video_path":"/data\/oops\/oops_val_v7_merged\/2119_E_merged.mp4","amusing_ids":[142,189,237],"surprise_scores":[0.0,0.0,0.0,0.0,0.388671875,0.53515625,0.466796875,0.376953125],"Explanations":[["","",""],["","",""],["","",""],["","",""],["the person continues paddling in the kayak through the","the figure in the canoe will continue paddling towards","the person in the boat will paddle forward through the"],["the person will continue to row the boat while enjoying","the rower will continue to row smoothly along the","the person continues to row, navigating through the river"],["the person struggles to get back into the boat or","the person might try to swim back towards the boat","the person will likely struggle to stay afloat in"],["the person will struggle to get back into the boat","the person will try to get back into the boat","the person will try to get back into the boat"]],"precision_at_k":0.6,"recall_at_k":1.0,"hit_at_1":1,"auc":0.7,"frame_indices":[0,47,94,142,189,237,284,332],"memory_evolution":["","","","",""," Then, a person is rowing a boat on a river, surrounded by trees and a cloudy sky."," Then, a person is rowing a boat on a river, surrounded by trees and a cloudy sky. Then, the person falls out of the boat."," Then, a person is rowing a boat on a river, surrounded by trees and a cloudy sky. Then, the person falls out of the boat. Then, the person is swimming in the water."]}

    metrics = compute_hyp_metrics_from_record(record, window_size=4)
    print(metrics)
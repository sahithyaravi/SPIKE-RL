import torch

from torch.nn.functional import softmax
from typing import List, Dict
from PIL import Image

from tqdm import tqdm

import os
import math
from peft import PeftModel
import gc
import numpy as np

from .weighted_captioning_grad import caption_by_weight

import torch
from contextlib import nullcontext

import torch
import secrets

import random

import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast
from typing import List, Optional

from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def score_window_hypotheses(
    video_frames: list,
    window_size: int,
    all_hypotheses: list,
    model,
    processor,
    memories: list = None,
    advantages=None,
    ref_model=None,
    beta=0.02,
):


    # ── 1. Build batch ────────────────────────────────────────────────────
    batch_prompts, batch_videos, batch_prefix_prompts, batch_prefix_videos, mapping = [], [], [], [], []
 
    for r_idx, hyp_per_window in enumerate(all_hypotheses):
        memory_rollout = memories[r_idx] if memories is not None else None
        for w_idx, hyps in enumerate(hyp_per_window):
            if not hyps or w_idx < window_size:
                continue
            memory_text = memory_rollout[w_idx] if memory_rollout is not None else ""
            context_frames = video_frames[w_idx - window_size : w_idx]
            prefix_conv = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Here is what has happened so far: {memory_text}"},
                    {"type": "video"},
                ],
            }]
            prefix_prompt = processor.apply_chat_template(prefix_conv, add_generation_prompt=True)
            for h_idx, hyp in enumerate(hyps):
                batch_prompts.append(prefix_prompt + "This will happen next: " + hyp)
                batch_videos.append(context_frames)
                batch_prefix_prompts.append(prefix_prompt)
                batch_prefix_videos.append(context_frames)
                mapping.append((r_idx, w_idx, h_idx))
 
    if not batch_prompts:
        return [[[] for _ in hyp_per_window] for hyp_per_window in all_hypotheses], None
 
    # ── 2. Tokenize ───────────────────────────────────────────────────────
    inputs = processor(
        text=batch_prompts, videos=batch_videos,
        return_tensors="pt", padding=True, truncation=True,
    ).to(model.device)
 
    # ── 3. Build labels (mask prefix + padding) ───────────────────────────
    labels = inputs.input_ids.clone()
    for i, (pfx_prompt, pfx_frames) in enumerate(zip(batch_prefix_prompts, batch_prefix_videos)):
        pfx_len = processor(text=pfx_prompt, videos=pfx_frames, return_tensors="pt", truncation=True).input_ids.size(1)
        pad_offset = (inputs.attention_mask[i] == 0).sum().item()  # ← how many pad tokens are prepended
        labels[i, :pad_offset + pfx_len] = -100                    # ← mask padding + prefix together
    labels[inputs.attention_mask == 0] = -100  # redundant now but harmless to keep

    # ── 4. Policy forward pass — per-sample summed log-probs ─────────────
    policy_lp_full = F.log_softmax(model(**inputs).logits.float()[:, :-1], dim=-1)  # (B, T-1, V)
    shift_labs     = labels[:, 1:].clone()                                            # (B, T-1)
    cont_mask      = shift_labs != -100
    shift_labs[~cont_mask] = 0  # avoid -100 index into vocab
    policy_logps   = (policy_lp_full.gather(2, shift_labs.unsqueeze(-1)).squeeze(-1) * cont_mask).sum(-1)  # (B,)
 


    # ── 5. KL penalty (token-by-token to avoid two full (B,T,V) tensors) ─
    kl_divergences = None
    has_peft = hasattr(model, "disable_adapter")
    print(f"Ref model: {ref_model is not None}, PEFT: {has_peft}")
 
    if ref_model is not None or has_peft:
        with torch.no_grad():
            if ref_model is not None:
                ref_lp_full = F.log_softmax(ref_model(**inputs).logits.float()[:, :-1], dim=-1)
            else:
                with model.disable_adapter():
                    ref_lp_full = F.log_softmax(model(**inputs).logits.float()[:, :-1], dim=-1)
 
        # kl_per_token = torch.stack([
        #     (policy_lp_full[:, t].exp() * (policy_lp_full[:, t] - ref_lp_full[:, t])).sum(-1)
        #     for t in range(policy_lp_full.size(1))
        # ], dim=1)  # (B, T-1)
        # del ref_lp_full
        # print("KL per token", kl_per_token)

        # # AFTER
        kl_per_token = (policy_lp_full.exp() * (policy_lp_full - ref_lp_full)).sum(-1)  # (B, T-1)
 
        kl_per_sample  = (kl_per_token * cont_mask).sum(-1) / cont_mask.sum(-1).clamp(min=1)
        kl_divergences = kl_per_sample.detach()
        policy_logps   = policy_logps - beta * kl_per_sample
        print(kl_divergences)
        del kl_per_token, kl_per_sample
 
    del policy_lp_full, inputs, labels, cont_mask, shift_labs
    torch.cuda.empty_cache()
    gc.collect()
 
    # ── 6. Unpack into result structure ───────────────────────────────────
    result = [[[] for _ in hyp_per_window] for hyp_per_window in all_hypotheses]
    for idx, (r_idx, w_idx, h_idx) in enumerate(mapping):
        result[r_idx][w_idx].append(policy_logps[idx])
 
    return result, kl_divergences

def new_generator(device):
    g = torch.Generator(device=device)
    g.manual_seed(secrets.randbits(64))   # unique each call
    return g


def summarize_memory(memory_text: str) -> str:
    """
    Summarizes the memory text to fit within the model's context window.
    Args:
        memory_text: The text to summarize.
    Returns:
        Summarized text.
    """
    with torch.no_grad():
        if len(memory_text.split()) <= 200:
            return memory_text
        else:
            # return max 200 words from the memory text
            # return ' '.join(memory_text.split()[:200])
            return summarizer(memory_text, max_length=50, min_length=20, do_sample=False)[0]["summary_text"]
        
def generate_hypothesis1(conv, visual_context, num_generations=3, model=None, processor=None, rolling=False):
    rolling = True
    prompt = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=prompt, 
        videos=visual_context, 
        return_tensors="pt",             
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    
    outputs = model.generate(
        **inputs.to(model.device),
        do_sample=True,
        temperature=1.0,
        top_k=50,
        num_return_sequences=num_generations,
        max_new_tokens=25,
    )
    hypotheses = processor.batch_decode(outputs, skip_special_tokens=True)
    # print(f"Generated {len(hypotheses)} hypotheses.")
    hypotheses = [hyp.strip().lower().split("assistant\n")[1].replace(":", "") for hyp in hypotheses]
    torch.cuda.empty_cache()
    gc.collect()
    return hypotheses


def generate_hypothesis(conv, visual_context, num_generations=3, model=None, processor=None, rolling=False):
    rolling = True
    prompt = processor.apply_chat_template(
        conv, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=prompt, 
        videos=visual_context, 
        return_tensors="pt",             
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    )
    all_hypotheses = []
    for i in range(num_generations):
        if rolling:
                
            # Solution 1: Set a different random seed for each generation
            torch.manual_seed(torch.randint(0, 2**32 - 1, (1,)).item())
            
            # Solution 2: Add some randomness to generation parameters
            temp = 1.0 + random.uniform(-0.1, 0.1)  # Vary temperature slightly
            top_p = 0.9 + random.uniform(-0.05, 0.05)  # Vary top_p slightly
            
            outputs = model.generate(
                **inputs.to(model.device),
                do_sample=True,
                temperature=max(0.1, temp),  # Ensure temperature doesn't go too low
                top_p=max(0.1, min(0.99, top_p)),  # Keep top_p in valid range
                num_return_sequences=1,
                max_new_tokens=25,
                return_dict_in_generate=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                # Solution 3: Force different random states
                use_cache=False,  # Disable KV cache which might affect randomness
            )
        else:
            torch.manual_seed(torch.randint(0, 2**32 - 1, (1,)).item())
            outputs = model.generate(
                **inputs.to(model.device),
                do_sample=True,
                temperature=1.1,
                top_p=0.5,
                num_return_sequences=1,
                max_new_tokens=20,
                return_dict_in_generate=True,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        hyp = processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        hyp = hyp.split("assistant\n")[1].replace(":", "")
        all_hypotheses.append(hyp)
        
        #Solution 4: Clear any cached states
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
    return all_hypotheses


def caption_frame(context_frames: List[Image.Image], observed_frame: Image.Image, memory_text: str, model=None, processor=None) -> str:
    # Observe the actual frame and caption it. This is used to build working memory.
    cap_conv = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                 "text": f"Here is what has happened so far: {memory_text}"},
                {"type": "text",
                 "text": "Describe what is happening in the last and most recent frame."},
                {"type": "video"},
                {"type": "image"},
            ],
        }
    ]
    cap_prompt = processor.apply_chat_template(cap_conv, add_generation_prompt=True)
    cap_inputs = processor(
        text=cap_prompt,
        videos=context_frames,
        images=[observed_frame],
        return_tensors="pt"
    )
    with torch.no_grad():
        cap_ids = model.generate(**cap_inputs.to(model.device), max_new_tokens=30)
        caption = processor.batch_decode(cap_ids, skip_special_tokens=True)[0].lower().split("assistant\n")[1].replace(":", "")
    return caption

def batch_score_hypotheses(hypotheses, memory_text, context_frames, model, processor):
    """Batch score all hypotheses in a single forward pass"""
    
    # Build all prompts at once
    prefix_conv = [{
        "role": "user",
        "content": [
            {"type": "text", "text": f"Here is what has happened so far: {memory_text}"},
            {"type": "video"},
        ],
    }]
    prefix_prompt = processor.apply_chat_template(prefix_conv, add_generation_prompt=True)
    
    # Prepare batch inputs
    batch_texts = []
    batch_videos = []
    
    for hyp in hypotheses:
        cont = "This will happen next: " + hyp
        full_txt = prefix_prompt + cont
        batch_texts.append(full_txt)
        batch_videos.append(context_frames)
    
    # Single tokenization call for all hypotheses
    batch_inputs = processor(
        text=batch_texts, 
        videos=batch_videos, 
        return_tensors="pt", 
        padding=True,
        padding_side="left",
        truncation=True
    ).to(model.device)
    
    # Get prefix length once
    prefix_inputs = processor(text=prefix_prompt, videos=context_frames, return_tensors="pt")
    prefix_len = prefix_inputs.input_ids.size(1)
    
    # Prepare labels for batch
    labels = batch_inputs.input_ids.clone()
    labels[:, :prefix_len] = -100
    
    # Single forward pass for all hypotheses
    with torch.no_grad():
        logits = model(**batch_inputs).logits  # no labels
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:].clone()
        cont_mask = shift_labels != -100
        shift_labels[~cont_mask] = 0
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, T-1)
        sum_logps = (token_logps * cont_mask).sum(-1)   # (B,) summed log-probs
        prior_scores = sum_logps / cont_mask.sum(-1).clamp(min=1)  # (B,) length-normalized

    
    del batch_inputs, labels, outputs, losses
    torch.cuda.empty_cache()
    gc.collect()
    return prior_scores, sum_logps


def batch_compute_likelihoods(hypotheses, memory_text, observed_frame, model, processor):
    """Batch compute P(observation|hypothesis) for all hypotheses"""
    
    # Pre-tokenize yes/no tokens once
    yes_id = processor.tokenizer.encode(" yes", add_special_tokens=False)[0]
    no_id = processor.tokenizer.encode(" no", add_special_tokens=False)[0]
    
    # Build all prompts
    batch_prompts = []
    batch_videos = []
    
    for hyp in hypotheses:
        conv_obs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Here is what has happened so far: {memory_text}"},
                {"type": "text", "text": f"Statement: {hyp}\nIs this statement true in the shown frame? Answer 'yes' or 'no'."},
                {"type": "video"},
            ],
        }]
        prompt = processor.apply_chat_template(conv_obs, add_generation_prompt=True)
        batch_prompts.append(prompt)
        batch_videos.append(observed_frame)
    
    # Batch tokenize base prompts
    base_inputs = processor(
        text=batch_prompts, 
        videos=batch_videos, 
        return_tensors="pt", 
        padding=True,
        padding_side="left",
    ).to(model.device)
    
    # Get logits for yes/no tokens in single forward pass
    with torch.no_grad():
        outputs = model(**base_inputs)
        logits = outputs.logits[:, -1, :]  # Last token logits for each sequence
        
        # Extract yes/no probabilities
        yes_logits = logits[:, yes_id]
        no_logits = logits[:, no_id]
        
        # Normalize to get proper probabilities
        yes_no_logits = torch.stack([yes_logits, no_logits], dim=1)
        log_probs = torch.log_softmax(yes_no_logits, dim=1)
        log_like = log_probs[:, 0]  # P(yes | hypothesis, observation)
    
    del base_inputs, outputs, logits, yes_no_logits, yes_logits, no_logits, log_probs
    torch.cuda.empty_cache()
    gc.collect()
    return log_like

@torch.no_grad()
def qwen_bayesian_surprise_text_future(memory_text: str, context_frames: List[Image.Image], observed_frame: Image.Image, num_hypotheses: int, model=None, processor=None, requires_grad=False, rolling=False) -> Dict:
    """
    Computes Bayesian surprise using the Qwen model on visual data.
    Args:
        memory_text: Textual memory to provide context.
        context_frames: List of PIL images representing the context frames.
        observed_frame: The current frame to analyze.
        num_hypotheses: Number of hypotheses to generate.
    Returns:
        Dictionary containing surprise scores, hypotheses, priors, posteriors, and memory evolution.
    """

    # Sample hypotheses based on W and H
    conv = [
        {
            "role": "user", 
            "content": [{"type": "text", "text" : f"Here is what happened so far from the beginning of the video: {memory_text}"},
                        {"type": "text", "text": "Based on this information, and recent frames from the video, answer in 8-10 words what will most likely happen in the next frame."},
                        {"type": "video"}],
        }
    ]
    h0 = generate_hypothesis(conv, context_frames, num_generations=num_hypotheses, model=model, processor=processor, rolling=True)
    hypotheses = h0 

    # Compute P_Prior
    prior_scores = []
    prefix_conv = [
        {
            "role": "user",
            "content": [
                {"type": "text",
                "text": f"Here is what has happened so far: {memory_text}"},
                {"type": "video"},
                {"type": "text",
                "text": "Here is what will happen next:"},   # neutral anchor line
            ],
        }
    ]
    prefix_prompt = processor.apply_chat_template(
        prefix_conv, add_generation_prompt=False
    )
    prefix_inputs = processor(
        text=prefix_prompt,
        videos=context_frames,          # the k past frames
        return_tensors="pt"
    )
    prefix_len = prefix_inputs.input_ids.size(1)        # tokens in the input prompt
    # empty cuda cache
    torch.cuda.empty_cache()
    for hyp in hypotheses:
        full_txt   = prefix_prompt + hyp 
        full_input = processor(
            text=full_txt,
            videos=context_frames,      # same frames each time
            return_tensors="pt"
        )

        # mask: ignore input prompt or prefix tokens in the loss
        labels = full_input.input_ids.clone()
        labels[:, :prefix_len] = -100

        with torch.no_grad():
            loss = model(**full_input.to(model.device), labels=labels).loss
        prior_scores.append(-loss.item())                
        del full_input, labels, loss
        torch.cuda.empty_cache()
        gc.collect()
    log_prior_raw = torch.tensor(prior_scores, device=model.device)     # negative log likelihoods
    log_prior = log_prior_raw - torch.logsumexp(log_prior_raw, dim=0)   # log P(h) - https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
    P_prior   = log_prior.exp()      

    # Compute posterior
    # convert_tokens_to_ids when tokens may contain a leading space.
    def _single_id(tok: str) -> int:
        ids = processor.tokenizer.encode(tok, add_special_tokens=False)
        assert len(ids) == 1, f"‘{tok}’ splits into {ids}"
        return ids[0]

    yes_id = _single_id(" yes")
    no_id  = _single_id(" no")
    log_like = []
    all_frames = context_frames + [observed_frame]

    for hyp in hypotheses:
        conv_obs = [
            {
                "role": "user",
                "content": [
                    {"type": "text",
                    "text": f"Here is what has happened so far: {memory_text}"},
                    {"type": "video"},
                    {"type": "text",
                    "text": f"Statement: {hyp}\nIs this statement true in the CURRENT frame? "
                            "Answer 'yes' or 'no'."},
                ],
            }
        ]

        prompt_obs = processor.apply_chat_template(conv_obs, add_generation_prompt=True)
        # print(repr(prompt_obs[-120:]))      
        inp_obs = processor(
            text=prompt_obs,
            videos=all_frames,                
            return_tensors="pt"
        )

        with torch.no_grad():
            # logits shape: (batch_size, sequence_length, vocab_size)
            logits = model(**inp_obs.to(model.device)).logits[0, -1] 


        sub_logits = logits[[yes_id, no_id]]             
        logprob_yes = sub_logits.log_softmax(dim=-1)[0]    # log P(yes|hyp, obs)
        log_like.append(logprob_yes)
        del inp_obs, logits
        torch.cuda.empty_cache()
        gc.collect()


    log_like   = torch.stack(log_like)                                  # (H,)
    log_unnorm = log_prior + log_like                                   # log P*Z - Bayes rule
    log_post   = log_unnorm - torch.logsumexp(log_unnorm, dim=0)        # log P(h|obs) P_post = torch.softmax(log_unnorm, dim=0)
    P_post     = log_post.exp()


    kl  = torch.sum(P_post * (log_post - log_prior)).item()

    # kl_torch = kl_div(
    #     input = log_post,                 # log-probs
    #     target = log_prior.exp(),         # plain probs
    #     reduction = "sum",                # ∑_h  p_post * (log p_post − log p_prior)
    #     log_target = False                # because target is in probability space
    #  ).item()
    
    log_mix = torch.logaddexp(log_prior, log_post) - math.log(2.0)      # log M
    js  = 0.5 * (
            torch.sum(P_prior * (log_prior - log_mix)) +
            torch.sum(P_post  * (log_post  - log_mix))
        )
    js_norm = (js / math.log(2.0)).item()

    # --- per-belief (signed) JS contribution ------------------------------------
    d_js = 0.5 * (P_prior * (log_prior - log_mix) +
                P_post  * (log_post  - log_mix))           # tensor, same length as hypotheses


    
    # add prior, posterior as string to each hypothesis in hypotheses
    # hypotheses = [
    #     f"{hyp} (prior: {P_prior[i]:.3f}, posterior: {P_post[i]:.3f}), JS: {abs(P_post[i]-P_prior[i]):.3f}"
    #     for i, hyp in enumerate(hypotheses)
    # ]
    return {
        "hypotheses":      hypotheses,
        "prior_probs":     P_prior.tolist(),
        "posterior_probs": P_post.tolist(),
        "KL_divergence":   kl,
        "JS_divergence": js_norm,
        "hyp_logps": P_post.tolist()
    }

# def qwen_bayesian_surprise_text_future(memory_text: str, context_frames: List[Image.Image], observed_frame: Image.Image, num_hypotheses: int, model=None, processor=None, requires_grad=False, rolling=False) -> Dict:
#     """
#     Computes Bayesian surprise using the Qwen model on visual data.
#     Args:
#         memory_text: Textual memory to provide context.
#         context_frames: List of PIL images representing the context frames.
#         observed_frame: The current frame to analyze.
#         num_hypotheses: Number of hypotheses to generate.
#     Returns:
#         Dictionary containing surprise scores, hypotheses, priors, posteriors, and memory evolution.
#     """

#     # Sample hypotheses based on W and H
#     conv = [
#         {
#             "role": "user", 
#             "content": [{"type": "text", "text" : f"Here is what happened so far from the beginning of the video: {memory_text}"},
#                         {"type": "text", "text": f"Based on this information, and recent frames from the video, answer in 8-10 words what will most likely happen in the next frame."},
#                         {"type": "video"}],
#         }
#     ]
#     h0 = generate_hypothesis(conv, context_frames, num_generations=num_hypotheses, model=model, processor=processor, rolling=rolling)
#     # Sample hypotheses based on W, H and O
#     # conv = [
#     #     {
#     #         "role": "user",
#     #         "content": [{"type": "text", "text" : f"Here is what happened so far from the beginning of the video: {memory_text}"},
#     #                     {"type": "text", "text": "You are now provided with recent frames from the video. Answer in one sentence what is happening now."},
#     #                     {"type": "video"}],
#     #     }
#     # ]
#     # h1 = generate_hypothesis(conv, observed_frame, num_generations=num_hypotheses, model=model, processor=processor)
#     hypotheses = h0
#     prior_scores, sum_logps = batch_score_hypotheses(hypotheses, memory_text, context_frames, model, processor)
#     log_prior_raw = torch.tensor(prior_scores, device=model.device)     # negative log likelihoods
#     log_prior = log_prior_raw - torch.logsumexp(log_prior_raw, dim=0)   # log P(h) 
#     P_prior   = log_prior.exp()    

#     log_like = batch_compute_likelihoods(hypotheses, memory_text, context_frames + [observed_frame], model, processor)
#     log_unnorm = log_prior + log_like                                   # log P*Z - Bayes rule
#     log_post   = log_unnorm - torch.logsumexp(log_unnorm, dim=0)        # log P(h|obs) P_post = torch
#     P_post     = log_post.exp()
#     hypotheses = [hyp.strip().lower() for hyp in hypotheses]


#     kl  = torch.sum(P_post * (log_post - log_prior))
    
#     log_mix = torch.logaddexp(log_prior, log_post) - math.log(2.0)    
#     js  = 0.5 * (
#             torch.sum(P_prior * (log_prior - log_mix)) +
#             torch.sum(P_post  * (log_post  - log_mix))
#         )
#     js_norm = (js / math.log(2.0))

#     d_js = 0.5 * (P_prior * (log_prior - log_mix) +
#                 P_post  * (log_post  - log_mix))           

#     # hypotheses = [
#     #     f"{hyp} (prior: {P_prior[i]:.3f}, posterior: {P_post[i]:.3f}), JS: {d_js[i]:.3f}"
#     #     for i, hyp in enumerate(hypotheses)
#     # ]
#     torch.cuda.empty_cache()
#     gc.collect()
#     del prior_scores, log_prior_raw, log_prior, log_like, log_unnorm, log_post

#     return {
#         "hypotheses":      hypotheses,
#         "prior_probs":     P_prior,
#         "posterior_probs": P_post,
#         "KL_divergence":   kl,
#         "JS_divergence": js_norm,
#         "hyp_logps": sum_logps
#     }



def run_bayesian_surprise_over_video(video_frames, window_size, num_hypotheses, method="prior_frame_bayesian_approach", model=None, processor=None,
requires_grad=False, rolling=False, use_history: bool = True):
    """
    Runs Bayesian surprise over a sequence of video frames.
    Args:
        video_frames: List of video frames (PIL images).
        window_size: Number of frames to consider for context.
        num_hypothesis: Number of hypotheses to generate."""
    surprise_scores = []
    running_memory = ""
    priors = []
    posteriors = []
    hypotheses = []
    all_window_logps = []
    memory_summaries_per_window = []
    captions_per_window = []
    

    for i in range(1, len(video_frames), 1):
        if i < window_size:
            prior_window = video_frames[0:i]
            observed_frame = video_frames[i]
        else:
            prior_window = video_frames[i-window_size:i]
            observed_frame = video_frames[i]
        if use_history:
            running_memory = summarize_memory(running_memory)
        else:
            running_memory = ""
        if method == "prior_frame_bayesian_approach":
            result = qwen_bayesian_surprise_text_future(
                memory_text=running_memory,
                context_frames=prior_window,
                observed_frame=observed_frame,
                num_hypotheses=num_hypotheses,
                model=model,
                processor=processor,
                requires_grad=requires_grad,
                rolling=rolling
            )

            with torch.no_grad():
                caption = caption_frame(
                    context_frames=prior_window,
                    observed_frame=observed_frame,
                    memory_text=running_memory,
                    model=model,
                    processor=processor
                )
                result["caption"] = caption
            
        captions_per_window.append(result["caption"])
        memory_summaries_per_window.append(running_memory)
        running_memory += (f" Then, {caption}")
        surprise_scores.append(result["KL_divergence"])
        priors.append(result["prior_probs"])
        posteriors.append(result["posterior_probs"])
        hypotheses.append(result["hypotheses"])
        all_window_logps.append(result["hyp_logps"])
        
    # for _ in range(1):
    surprise_scores.insert(0, 0.0)
    priors.insert(0, [0.0] * num_hypotheses)
    posteriors.insert(0, [0.0] * num_hypotheses)
    hypotheses.insert(0, [""] * num_hypotheses)
    memory_summaries_per_window.insert(0, "")
    captions_per_window.insert(0, "")
    # Convert list inside hypotheses into single string with \n between them dd

    return {
        "surprise_scores": surprise_scores,
        "memory_evolution": memory_summaries_per_window,
        "priors": priors,
        "posteriors": posteriors,
        "explanations": hypotheses,
        "frames": video_frames,
        "captions": captions_per_window,
        # "all_hyp_logps": all_window_logps 
    }


def qwen_surprise_tracker(
    frames: List[np.ndarray],
    window_size: int = 1,
    top_k: int = 5,
    method: str = "prior_frame_bayesian_approach",
    caption_video: bool = False,
    vr=None,
    model=None,
    processor=None,
    requires_grad=False,
    rolling=False,
    use_history: bool = True,
) -> Dict[str, List]:
    """
    Process the video to extract frames and save them to the output directory.
    """

    surprise_output = run_bayesian_surprise_over_video(
        video_frames=frames,
        window_size=window_size,
        num_hypotheses=top_k,
        method=method,
        model=model,
        processor=processor,
        requires_grad=requires_grad,
        rolling=rolling,
        use_history=use_history,

    )

    if caption_video:
        caption_weighted, sampled_frames_weighted = caption_by_weight(
            model=model,
            processor=processor,
            frames=frames,
            scores=surprise_output["surprise_scores"],
            vr=vr
           
        )
        caption_unweighted, sampled_frames_unweighted = caption_by_weight(
            model=model,
            processor=processor,
            frames=frames,
            scores=[1.0] * len(frames),
            vr = vr
     
        )
        surprise_output["caption_weighted"] = caption_weighted
        surprise_output["caption_unweighted"] = caption_unweighted
        surprise_output["sampled_frames_weighted"] = sampled_frames_weighted
        surprise_output["sampled_frames_unweighted"] = sampled_frames_unweighted
    return surprise_output 

    





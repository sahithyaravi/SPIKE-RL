import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

def patched_get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
    """
    Robust fix that handles shape mismatches properly
    """
    # Get the spatial merge size - try different attributes that might exist in your version
    if hasattr(self.config, 'spatial_merge_size'):
        spatial_merge_size = self.config.spatial_merge_size
    elif hasattr(self.config, 'patch_size'):
        spatial_merge_size = self.config.patch_size
    else:
        spatial_merge_size = 2  # Default fallback
    
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id
    mrope_position_deltas = []
    
    if image_grid_thw is not None or video_grid_thw is not None:
        device = input_ids.device
        
        # Ensure all tensors are on the same device
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(device)
        
        for i, input_ids_i in enumerate(input_ids):
            image_nums, video_nums = 0, 0
            
            # CRITICAL FIX: Handle shape mismatch properly
            if attention_mask is not None:
                attention_mask_i = attention_mask[i].to(device)
                
                # Check if shapes match and handle mismatch
                if attention_mask_i.shape[0] != input_ids_i.shape[0]:
                    min_len = min(attention_mask_i.shape[0], input_ids_i.shape[0])
                    # Truncate both to the same length
                    attention_mask_i = attention_mask_i[:min_len]
                    input_ids_i = input_ids_i[:min_len]
                
                # Now apply the mask safely
                input_ids_i = input_ids_i[attention_mask_i == 1]
            
            # Handle empty tensor case
            if input_ids_i.numel() == 0:
                mrope_position_deltas.append(0)
                continue
                
            vision_start_indices = torch.argwhere(input_ids_i == vision_start_token_id).squeeze(1)
            
            # Handle case where no vision tokens found
            if vision_start_indices.numel() == 0:
                mrope_position_deltas.append(0)
                continue
                
            # Check bounds to prevent index errors
            valid_indices = vision_start_indices[vision_start_indices < len(input_ids_i) - 1]
            if valid_indices.numel() == 0:
                mrope_position_deltas.append(0)
                continue
                
            vision_tokens = input_ids_i[valid_indices + 1]
            
            image_indices = torch.argwhere(vision_tokens == image_token_id).squeeze(1)
            video_indices = torch.argwhere(vision_tokens == video_token_id).squeeze(1)
            
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_grid_thw, video_grid_thw
            
            for vision_idx in valid_indices:
                vision_idx = vision_idx.item()
                llm_pos_ids_list.append(torch.arange(st, vision_idx, device=device, dtype=torch.long))
                
                if image_indices.numel() > 0 and image_nums < len(image_indices) and image_nums in image_indices:
                    if remain_images is not None and image_nums < remain_images.shape[0]:
                        t, h, w = remain_images[image_nums].tolist()
                        h //= spatial_merge_size
                        w //= spatial_merge_size
                        pos_id = torch.arange(h * w, device=device, dtype=torch.long) + vision_idx
                        llm_pos_ids_list.append(pos_id)
                        st = vision_idx + h * w + 1
                    image_nums += 1
                elif video_indices.numel() > 0 and video_nums < len(video_indices) and video_nums in video_indices:
                    if remain_videos is not None and video_nums < remain_videos.shape[0]:
                        t, h, w = remain_videos[video_nums].tolist()
                        h //= spatial_merge_size
                        w //= spatial_merge_size
                        pos_id = torch.arange(t * h * w, device=device, dtype=torch.long) + vision_idx
                        llm_pos_ids_list.append(pos_id)
                        st = vision_idx + t * h * w + 1
                    video_nums += 1
                else:
                    st = vision_idx + 1
                    
            llm_pos_ids_list.append(torch.arange(st, len(input_ids_i), device=device, dtype=torch.long))
            
            if llm_pos_ids_list:
                llm_positions = torch.cat(llm_pos_ids_list)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids_i))
            else:
                mrope_position_deltas.append(0)
    else:
        mrope_position_deltas = [0] * input_ids.shape[0]
    
    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return input_ids, mrope_position_deltas


def _get_rope_index_patched(
    self,
    input_ids: torch.LongTensor = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    second_per_grid_ts: torch.Tensor | None = None,  # <-- add this
    attention_mask: torch.Tensor | None = None,
):
    # Copied structure from HF, with TWO changes:
    # (A) clamp per-sample attention_mask to match ids length before boolean indexing
    # (B) reuse that exact clamped mask for the final assignment and delta calc

    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id     = self.config.image_token_id
    video_token_id     = self.config.video_token_id
    vision_start_id    = self.config.vision_start_token_id
    mrope_deltas = []

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_ids)

        position_ids = torch.ones(
            3, total_ids.shape[0], total_ids.shape[1],
            dtype=torch.long, device=total_ids.device
        )
        img_idx = vid_idx = 0

        for i, ids in enumerate(total_ids):
            mask_i = attention_mask[i].to(ids.device)

            # --- (A) ensure same length to avoid IndexError ---
            if mask_i.shape[0] != ids.shape[0]:
                min_len = min(mask_i.shape[0], ids.shape[0])
                mask_i = mask_i[:min_len]
                ids    = ids[:min_len]

            nonpad = (mask_i == 1)
            ids_np = ids[nonpad]       # ids without padding

            # Count media tokens in the non-padded portion
            vision_starts = torch.argwhere(ids_np == vision_start_id).squeeze(1)
            vision_tokens = ids_np[vision_starts + 1] if vision_starts.numel() > 0 else ids_np[:0]
            n_images = (vision_tokens == image_token_id).sum().item()
            n_videos = (vision_tokens == video_token_id).sum().item()

            # Walk the sequence and build per-segment THW + text positions
            tokens_list = ids_np.tolist()
            llm_pos_chunks = []
            st = 0
            rem_img, rem_vid = n_images, n_videos

            for _ in range(n_images + n_videos):
                ed_img = tokens_list.index(image_token_id, st) if (image_token_id in tokens_list and rem_img > 0) else len(tokens_list) + 1
                ed_vid = tokens_list.index(video_token_id, st) if (video_token_id in tokens_list and rem_vid > 0) else len(tokens_list) + 1

                if ed_img < ed_vid:
                    t, h, w = image_grid_thw[img_idx].tolist()
                    img_idx += 1; rem_img -= 1; ed = ed_img
                else:
                    t, h, w = video_grid_thw[vid_idx].tolist()
                    vid_idx += 1; rem_vid -= 1; ed = ed_vid

                # prepend text positions in this chunk
                text_len = ed - st
                st_idx = (llm_pos_chunks[-1].max() + 1).item() if llm_pos_chunks else 0
                llm_pos_chunks.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # append THW grid positions for this media block
                h //= spatial_merge_size
                w //= spatial_merge_size
                t_index = torch.arange(t).view(-1, 1).expand(-1, h * w).flatten()
                h_index = torch.arange(h).view(1, -1, 1).expand(t, -1, w).flatten()
                w_index = torch.arange(w).view(1, 1, -1).expand(t, h, -1).flatten()
                llm_pos_chunks.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)

                st = ed + t * h * w

            if st < len(tokens_list):
                text_len = len(tokens_list) - st
                st_idx = (llm_pos_chunks[-1].max() + 1).item() if llm_pos_chunks else 0
                llm_pos_chunks.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_chunks, dim=1).reshape(3, -1).to(position_ids.device)

            # --- (B) reuse the same clamped mask for assignment ---
            position_ids[..., i, nonpad] = llm_positions
            mrope_deltas.append(llm_positions.max() + 1 - int(nonpad.sum()))

        mrope_deltas = torch.tensor(mrope_deltas, device=position_ids.device).unsqueeze(1)
        return position_ids, mrope_deltas

    # fallback (text-only / no grids): keep HF logic, but guard mismatched masks
    if attention_mask is not None and input_ids is not None and attention_mask.shape[-1] != input_ids.shape[1]:
        attention_mask = attention_mask[..., : input_ids.shape[1]]
    if attention_mask is not None:
        pos = attention_mask.long().cumsum(-1) - 1
        pos.masked_fill_(attention_mask == 0, 1)
        pos = pos.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_pos = pos.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope = max_pos + 1 - attention_mask.shape[-1]
    else:
        pos = torch.arange(input_ids.shape[1], device=input_ids.device).view(1,1,-1).expand(3, input_ids.shape[0], -1)
        mrope = torch.zeros([input_ids.shape[0], 1], device=input_ids.device, dtype=input_ids.dtype)
    return pos, mrope
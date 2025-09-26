# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import gc
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TorchAoConfig

)


from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

# from sentence_transformers import SentenceTransformer, util
from qwen_vl_utils import process_vision_info
from open_r1_video.belief_tracker import qwen_surprise_tracker, score_window_hypotheses
from open_r1_video.weighted_captioning import adaptive_frame_sampling_pdf
from open_r1_video.video_processing import extract_k_frames_decord_cpu
from .dataset import get_data
from .patch_rope_index import patched_get_rope_index, _get_rope_index_patched
from typing import List
from torch.nn import functional as F

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionAttention
from torch.cuda.amp import autocast


from peft import LoraConfig, get_peft_model



VisionAttention.is_causal=False
Qwen2VLForConditionalGeneration.get_rope_index = _get_rope_index_patched
Qwen2_5_VLForConditionalGeneration.get_rope_index = _get_rope_index_patched
print("Applied Qwen2-VL device consistency fix")
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]



class Qwen2VLGRPOTrainerBelief(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model_init_kwargs["torch_dtype"] = torch.bfloat16
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL"  in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # Find all attention projection modules
        target_modules = []
        for name, module in model.named_modules():
            if name.endswith(('q_proj', 'k_proj', 'v_proj', 'o_proj')):
                # Extract just the suffix for PEFT
                suffix = name.split('.')[-1]
                if suffix not in target_modules:
                    target_modules.append(suffix)

        print(f"Found target modules: {target_modules}")

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_modules,  # Use discovered modules
            bias="none",
            task_type="CAUSAL_LM",
        )
        print(peft_config)

        # if peft_config is not None:
        model = get_peft_model(model, peft_config)
    

        # Reference model
        # if is_deepspeed_zero3_enabled():
        #     if "Qwen2-VL" in model_id:
        #         self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        #     elif "Aria" in model_id:
        #         self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        #     elif "Qwen2.5-VL" in model_id:
        #         self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        #     else:
        #         self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        # elif peft_config is None:
        #     # If PEFT configuration is not provided, create a reference model based on the initial model.
        #     self.ref_model = create_reference_model(model)
        # else:
        #     # If PEFT is used, the reference model is not needed since the adapter can be disabled
        #     # to revert to the initial model.
        #     self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen2-VL" in model_id or "Qwen2.5" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        # self.sim_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # if self.ref_model is not None:
        #     # if self.is_deepspeed_enabled:
        #     #     self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
        #     # else:
        #     self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        self.ref_model = None

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    
    # def _simple_cosine_similarity(self, text1: str, text2: str) -> float:
    #     # 1. Encode each sentence into an embedding (vector)
    #     emb1 = self.sim_encoder.encode(text1, convert_to_tensor=True)
    #     emb2 = self.sim_encoder.encode(text2, convert_to_tensor=True)

    #     # 2. Compute cosine similarity between the two embeddings
    #     similarity_value = util.cos_sim(emb1, emb2)

    #     return similarity_value

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if hasattr(self, 'state') and self.state is not None:
            current_step = self.state.global_step
            # print(f"Current step: {current_step}")
        else:
            # print("State not available")
            current_step = 0
    
        total_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        m = 3 # 3 rollouts
        beta = 0.02

        for example in inputs:
            try:
                torch.cuda.empty_cache()
                gc.collect()

                # Get Video Frames
                video_path, ground_truth = get_data(example) 
                try:
                    save_test_image = False if current_step != 0  else True
                    frames, frame_indices, total_frames, fps, vr = extract_k_frames_decord_cpu(
                        video_path=video_path, k=None, save_image=save_test_image
                    )
                except Exception as e:
                    print(f"Error extracting frames from video {video_path}: {e}")
                    continue
                rollout_data = []

                # Step 1: generate m roll-outs with no grad
                for idx in range(m):
                    torch.cuda.empty_cache()
                    gc.collect()
                    with torch.no_grad(), autocast():
                        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                            result = qwen_surprise_tracker(
                                frames=frames,
                                window_size=4,
                                top_k=3,
                                method="prior_frame_bayesian_approach",
                                caption_video=False,   
                                vr=vr,
                                model=unwrapped_model,
                                processor=self.processing_class,
                                requires_grad=False,
                                rolling=True
                            )
                            top_frames, frame_indices = adaptive_frame_sampling_pdf(
                                scores=result["surprise_scores"], vr=vr, max_frames=len(frames)
                            )

                            prompt_variants = ["Explain what is happening in the video in 50-100 words.",
                                            "Describe the events occurring in the video in 50-100 words.",
                                            "Provide a detailed explanation of the video's content in 50-100 words.",
                                                "Summarize the main events in the video in 50-100 words.",]
                            prompt = prompt_variants[idx % len(prompt_variants)]
                            cap_conv = [{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "video"},
                                ],
                            }]
                            cap_prompt = self.processing_class.apply_chat_template(cap_conv, add_generation_prompt=True)
                            with torch.no_grad():
                                gen_inputs = self.processing_class(text=cap_prompt, videos=top_frames, return_tensors="pt").to(model.device)
                                gen_out = unwrapped_model.generate(
                                    **gen_inputs, max_new_tokens=80,
                                    temperature=0.8, top_p=0.9, do_sample=True,
                                    return_dict_in_generate=True,
                                )
                                generated_ids = gen_out.sequences[0]
                                caption_text = self.processing_class.tokenizer.decode(generated_ids, skip_special_tokens=True)
                                del gen_inputs, gen_out, generated_ids
                                torch.cuda.empty_cache()
                            
                            # Compute Reward
                            with torch.no_grad():
                                rewards_tensor = self.reward_funcs[0](responses=[[caption_text]], ground_truths=[str(ground_truth)])
                            r_scalar = torch.as_tensor(rewards_tensor, dtype=torch.float32).view(-1)[0].to(model.device)
                            rollout_data.append({
                                'frames': frames,
                                'reward': r_scalar,
                                'caption': caption_text,
                                'hypothesis_per_window': result['explanations'],
                                "memory_evolution": result['memory_evolution'],
                            })
                            del result, top_frames, rewards_tensor
                            torch.cuda.empty_cache()
                            gc.collect()
            
                # Step 2: Score the hypotheses for each window using the model
                with autocast(): 
                    all_hypotheses = []
                    for info in rollout_data:
                        all_hypotheses.append(info["hypothesis_per_window"])
                    
                    # print("All hypotheses: ", all_hypotheses)

                    all_memories = []
                    for info in rollout_data:
                        all_memories.append(info["memory_evolution"])

                    # print("All hypotheses: ", all_hypotheses)
                    rewards = torch.stack([r_data["reward"] for r_data in rollout_data])
                    adv = ((rewards - rewards.mean()) / (rewards.std() + 1e-6)).detach()
                    logps_per_rollout = score_window_hypotheses(video_frames=frames, window_size=4, all_hypotheses=all_hypotheses, model=model, processor=self.processing_class,
                    memories=all_memories, advantages=adv)
                
                    # Step 3: aggregate log-probs per roll-out
                    # Compute advantages from the rewards and do a single backward pass
                    total_logp_per_rollout = [sum(sum(window_logps) for window_logps in lp) for lp in logps_per_rollout]
                    

                    loss = -(adv * torch.stack(total_logp_per_rollout)).mean()

                    total_loss = total_loss + loss
                            # Log metrics
                self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
                # Log reward statistics for better understanding of performance
                self._metrics["reward_std"].append(rewards.std().item())
                self._metrics["reward_min"].append(rewards.min().item())
                self._metrics["reward_max"].append(rewards.max().item())
            
                self._metrics["advantages"].append(self.accelerator.gather_for_metrics(adv).mean().item())
                # print(f"Rewards for video path {video_path} ", rewards)

                del rollout_data, all_hypotheses, logps_per_rollout, total_logp_per_rollout
                del rewards, adv, frames
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                # clear all gpu memory and restart
                torch.cuda.empty_cache()
                gc.collect()
                print(f"Error during video {video_path}: {e}")
                import time
                time.sleep(0.1)
                continue
        
        return total_loss/len(inputs) if len(inputs) > 0 else total_loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
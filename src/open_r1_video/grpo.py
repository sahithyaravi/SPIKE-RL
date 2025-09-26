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

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List

from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1_video.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOTrainerBelief
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from llm_match_reward import HuggingFaceLLMReward


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy",],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    jsonl_path: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "JSONL file paths",
            "nargs": "+",  # One or more arguments
        },
    )
    training_data_size: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of training samples to use. ActivityNet: Oops will be in the ratio of 800:200" },
    )

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    print(contents[:2]) # print online completion
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r"<answer>(.*?)</answer>", sol, re.DOTALL)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
                # student_answer = content_match.group(1).strip() if content_match else content.strip()
                if content_match:
                    student_answer = content_match.group(1).strip()
                    # HACK, if First letter is correct reward 1
                    # Compare the extracted answers
                    if student_answer[0] == ground_truth[0]:
                        reward = 1.0
                else:
                    reward = 0.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "llm_match": HuggingFaceLLMReward(),
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

from datasets import Dataset, DatasetDict
import json

def create_dataset_from_jsonl_simple(jsonl_paths, training_data_size=1000):
    if isinstance(jsonl_paths, str):
        jsonl_paths = [jsonl_paths]
    
    if not jsonl_paths:
        raise ValueError("No JSONL paths provided")
    
    datasets = []
    
    for path in jsonl_paths:
        try:
            print(f"Loading dataset from: {path}")
            dataset = Dataset.from_json(path)
            
            # Filter each dataset individually
            filtered_dataset = dataset.filter(lambda x: x['duration'] < 80) if "duration" in dataset.column_names else dataset
            sorted_dataset = filtered_dataset.sort("duration") if "duration" in filtered_dataset.column_names else filtered_dataset
            print(f"Filtered dataset from {path} has length: {len(sorted_dataset)}")
            if "activitynet" in path.lower():
                K = int(training_data_size * 0.7)
            else:
                K = int(training_data_size * 0.3)
            if len(sorted_dataset) > 0:
                # Sample up to K from this file
                num_samples = min(K, len(sorted_dataset))
                sampled_dataset = sorted_dataset.shuffle(seed=50).select(range(num_samples))
                print(f"Selected {num_samples} samples from {path}")
                
                datasets.append(sampled_dataset)
            else:
                print(f"Warning: No samples left after filtering from {path}")
                
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No valid datasets could be loaded")
    
    # Concatenate all sampled datasets
    if len(datasets) == 1:
        base_dataset = datasets[0]
    else:
        from datasets import concatenate_datasets
        base_dataset = concatenate_datasets(datasets)
    
    print(f"Final combined dataset length: {len(base_dataset)}")
    return DatasetDict({
        "train": base_dataset
    })


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in reward_funcs_registry]
    
    if script_args.jsonl_path:
        # # load dataset from jsonl
        dataset = create_dataset_from_jsonl_simple(script_args.jsonl_path, script_args.training_data_size)
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Caption this video." }, #
            ],
        }

    QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>. "

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        # {"type": "video", "video": example["video"]},
                        # {"type": "video", "bytes": open(example["video"],"rb").read()},
                        {"type": "text", "text": "Explain what is happening in this video. "},
                    ],
                },
            ],
    }

    if "image" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
    elif "video" in dataset[script_args.dataset_train_split].features:
        dataset = dataset.map(
            make_conversation_video,
        )
    else:
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")
    
    # import pdb; pdb.set_trace()

    trainer_cls =  Qwen2VLGRPOTrainerBelief

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        # peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Check if state was properly restored
    current_step  = trainer.state.global_step
    print(f"Trainer now at step: {current_step}")

    # after building the trainer
    resume = getattr(training_args, "resume_from_checkpoint", None)
    print("Resuming from:", resume)
    
    trainer.train(resume_from_checkpoint=resume)  # <- this is required
    resumed_step = trainer.state.global_step
    print(f"Resumed at step: {resumed_step}")

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

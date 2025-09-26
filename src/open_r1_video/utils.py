
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,

)
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TorchAoConfig
import math
import torch


def load_model_and_processor(model_path = "Qwen/Qwen2.5-VL-7B-Instruct"):
    quantization_config = TorchAoConfig("int4_weight_only", group_size=128)
    if "32B" in model_path or "72B" in model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, quantization_config=quantization_config,
        torch_dtype=torch.bfloat16, device_map="auto",  attn_implementation="flash_attention_2",
    )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path , torch_dtype=torch.bfloat16, device_map="auto",  attn_implementation="flash_attention_2",
        )
    processor = AutoProcessor.from_pretrained(model_path)
    print("Model loaded.", model)

    return model, processor

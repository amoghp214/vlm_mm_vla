import torch


from transformers import AutoProcessor
from transformers import AutoModelForSeq2SeqLM
from transformers import GenerationConfig
from transformers import Gemma3ForConditionalGeneration
from transformers import LlavaForConditionalGeneration


def _patch_ambiguous_text_config_for_generation(model):
    """Patch configs that expose multiple text sub-configs (e.g., decoder + text_config)."""
    cfg = model.config

    try:
        cfg.get_text_config()
        return model
    except Exception as e:
        if "Multiple valid text configs" not in str(e):
            return model

    preferred_text_config = getattr(cfg, "decoder", None) or getattr(cfg, "text_config", None)
    if preferred_text_config is None:
        return model

    def _get_text_config(*args, **kwargs):
        return preferred_text_config

    cfg.get_text_config = _get_text_config

    try:
        model.generation_config = GenerationConfig.from_model_config(cfg)
    except Exception:
        pass

    return model

def load_llava(device="cuda:0"):
    try:
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(0)
    except RuntimeError as e:
        print(f"Primary load failed: {e}\nFalling back to CPU load with reduced memory requirements.")
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(0)
    processor = AutoProcessor.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        padding_side="left",
        use_fast=False
    )
    model.eval()
    model.to(device)
    return model, processor


def load_gemma3(device="cuda:0"):
    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            "google/gemma-3-4b-it",
            torch_dtype=torch.bfloat16,
            # dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
    except RuntimeError as e:
        print(f"Primary load failed: {e}\nFalling back to CPU load with reduced memory requirements.")
        model = Gemma3ForConditionalGeneration.from_pretrained(
            "google/gemma-3-4b-it",
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
        )
    processor = AutoProcessor.from_pretrained(
        "google/gemma-3-4b-it",
        padding_side="left",
        use_fast=False
    )
    # model = _patch_ambiguous_text_config_for_generation(model)
    model.eval()
    model.to(device)
    return model, processor

def load_t5gemma(device="cuda:0"):
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/t5gemma-2-270m-270m",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
    except RuntimeError as e:
        print(f"Primary load failed: {e}\nFalling back to CPU load with reduced memory requirements.")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/t5gemma-2-270m-270m",
            torch_dtype=torch.bfloat16,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
        )
    processor = AutoProcessor.from_pretrained(
        "google/t5gemma-2-270m-270m",
        padding_side="left",
        use_fast=False
    )
    model = _patch_ambiguous_text_config_for_generation(model)
    model.eval()
    model.to(device)
    return model, processor


import time
import io

import requests
import torch
import json
from PIL import Image
from transformers import AutoProcessor
from transformers import Gemma3ForConditionalGeneration
from transformers import LlavaForConditionalGeneration

from vlm_loaders import load_llava, load_gemma3, load_t5gemma

system_prompt_path = "/home/hice1/apalasamudram6/scratch/vlm_mm_vla/vlm_mm/mm_prompts/task_splitting_system_prompt.txt"


def configure_fast_inference():
    """Enable safe speed-oriented runtime knobs for GPU inference."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Avoid recording a separate CUDAGraph for each dynamic input shape.
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    

# Load system prompt from JSON file
with open(system_prompt_path, "r") as f:
    system_prompt = json.load(f)

user_prompt = "The robot's prompt is: 'put the black bowl in the bottom drawer of the cabinet and close it'."
img_url = "https://drive.google.com/uc?export=download&id=1br3juDZO5_R04wIUgKZwUqY9jgbNQnId"

configure_fast_inference()
model, processor = load_gemma3()

start_time = time.time()

try:
    messages = [
        system_prompt,
        {
            "role": "user", "content": [
                {"type": "image", "url": img_url},
                {"type": "text", "text": user_prompt},
            ]
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        # padding="max_length",
        # truncation=True,
        # max_length=512,
    ).to(model.device)
except ValueError as e:
    prompt = system_prompt['content'][0]['text'] + "\n" + user_prompt + "\n The image is: <start_of_image>. The task plan is: "
    # prompt = "<start_of_image> in this image, there is"
    image = Image.open(requests.get(img_url, stream=True).raw)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    

inputs_processed_time = time.time()

with torch.inference_mode():
# with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        cache_implementation="static",
    )

generation_completed_time = time.time()

if getattr(model.config, "is_encoder_decoder", False):
    generated_ids = output[0]
else:
    prompt_length = inputs.input_ids.shape[1]
    generated_ids = output[0][prompt_length:]

decoded_text = processor.decode(generated_ids, skip_special_tokens=True).strip()
print(f"Generated token count: {generated_ids.shape[0]}")
print("Output:", decoded_text)
print(f"Inputs processed in {inputs_processed_time - start_time:.2f} seconds")
print(f"Generation completed in {generation_completed_time - inputs_processed_time:.2f} seconds")
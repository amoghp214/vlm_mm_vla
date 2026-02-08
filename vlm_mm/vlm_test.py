import time
import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

try:
    model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it",
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
except RuntimeError as e:
    print(f"Primary load failed: {e}\nFalling back to CPU load with reduced memory requirements.")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it",
        dtype=torch.float32,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
processor = AutoProcessor.from_pretrained(
    "google/gemma-3-4b-it",
    padding_side="left"
)

messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant."}
        ]
    },
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "What is shown in this image?"},
        ]
    },
]

start_time = time.time()

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)

inputs_processed_time = time.time()

output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static", remove_invalid_values=True)

generation_completed_time = time.time()

prompt_length = inputs.input_ids.shape[1]

print("Output:", processor.decode(output[0][prompt_length:], skip_special_tokens=True))
print(f"Inputs processed in {inputs_processed_time - start_time:.2f} seconds")
print(f"Generation completed in {generation_completed_time - inputs_processed_time:.2f} seconds")
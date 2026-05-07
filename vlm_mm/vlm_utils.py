import torch

def configure_fast_inference():
    """Enable safe speed-oriented runtime knobs for GPU inference."""
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Avoid recording a separate CUDAGraph for each dynamic input shape.
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    
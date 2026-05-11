# VLM Micromanager for Vision-Language Action Models

This repository implements a **VLM Micromanager (VLM MM)** system that steers and guides Vision-Language Action (VLA) models to improve performance on robotic manipulation tasks. The VLM MM uses advanced visual reasoning to dynamically adjust and refine task execution in real-time, enabling the VLA to handle complex, multi-step manipulation tasks more effectively.

## Overview

### What is VLM MM?

The VLM Micromanager is an intelligent oversight system that:
- **Monitors** VLA execution in real-time by analyzing robot observations and video context
- **Reasons** about task progress using Vision-Language Models (Gemma-3, LLaVA, T5-Gemma)
- **Guides** the VLA by generating refined prompts and task breakdowns when it detects suboptimal behavior
- **Improves performance** by decomposing complex tasks into achievable subtasks and providing contextual guidance

### Architecture

```
┌─────────────────────────────────────────────────────┐
│         Original Task Prompt                        │
│  "Put bowl in drawer and close it"                  │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │   VLA observes      │
        │   environment       │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────────────────┐
        │   VLM Micromanager              │
        │  • Analyzes current state       │
        │  • Detects task progress        │
        │  • Generates guidance           │
        └──────────┬──────────────────────┘
                   │
     ┌─────────────┼─────────────┐
     │             │             │
  Success      Refine        Decompose
     │        Prompt        Subtask
     │             │             │
     └─────────────┴─────────────┘
              │
     ┌────────▼────────┐
     │   VLA acts with  │
     │  refined prompt  │
     └─────────────────┘
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ VRAM (for VLM inference)
- Linux/MacOS

### Installation

1. **Create and activate conda environment:**

```bash
conda env create -f environment.yml
conda activate vla-e
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Install the package:**

```bash
pip install -e .
```

### Configuration

Edit `vlm_mm/configs/vlm_config_main.yaml` to set up your environment:

```yaml
# Cache directory for downloading models
cache_dir: /path/to/your/huggingface/cache

# VLM Micromanager configuration
vlmmm:
  model_name: gemma3  # Options: gemma3, llava, t5gemma
  vlm_mm_context_dir: vlm_mm/mm_context
  vlm_mm_prompts_dir: vlm_mm/mm_prompts
  original_prompt: "your task prompt"
  curr_image_path: "vlm_mm/mm_context/curr_image.png"
  context_video_path: "vlm_mm/mm_context/curr_context.mp4"
  device: cuda:0

# VLA configuration
vla:
  model: openvla
  task_suite_name: libero_10  # or libero_spatial, libero_object, etc.
  bddl_file: /path/to/task.bddl
  prompt: "your task prompt"
  record_path: /path/to/output/video.mp4
  out_file: /path/to/output/demo.hdf5
  device: cuda:0
```

## Usage

### Running VLA with VLM Micromanager Guidance

1. **Start the VLM Server:**

```bash
cd vlm_mm
python vlm_server.py --config configs/vlm_config_main.yaml --port 8000
```

The server will:
- Load the specified VLM (Gemma-3, LLaVA, or T5-Gemma)
- Initialize the VLM Micromanager
- Listen for task execution requests on `http://localhost:8000`

2. **In another terminal, run VLA inference with VLM MM guidance:**

```bash
python vlm_mm/vla_runner.py --config vlm_mm/configs/vlm_config_main.yaml
```

This will:
- Load the OpenVLA model for your task suite
- Execute the task while querying the VLM MM for guidance
- Record the demonstration with VLM MM-guided actions
- Save output to HDF5 file with trajectory data

### Testing

To verify the VLM MM server is working:

```bash
cd vlm_mm
python test_vlm_mm.py --config configs/vlm_config_main.yaml
```

## Data and Results

### Output Structure

The VLA runner generates:

- **HDF5 Demonstration File** (`out_file`): Contains:
  - Observations (RGB images, joint states)
  - Actions (robot end-effector commands)
  - Rewards
  - Task metadata

- **Video Recording** (`record_path`): MP4 video of the task execution

- **Logs**: Console output with task progress and VLM MM reasoning

## Performance Analysis

Use the included analysis tools to evaluate performance:

```bash
# Convert HDF5 to JSON format
python utils/hdf5_to_json.py --hdf5_file /path/to/demo.hdf5 --output_dir /path/to/output

# Run metrics analysis
python explainability/vla_metrics.py --demo_file /path/to/demo.hdf5
```

## Supported VLM Models

- **Gemma-3**: Recommended for balanced performance and speed
- **LLaVA**: Vision-specific language model with strong visual understanding
- **T5-Gemma**: Encoder-decoder architecture for detailed reasoning

## Task Suites

This system supports LIBERO task suites:

- `libero_10`: 10 diverse manipulation tasks
- `libero_spatial`: 50 tasks with spatial variations
- `libero_object`: 50 tasks with object variations
- `libero_100`: 100 diverse tasks

## Advanced Features

### Perturbation Analysis

Generate perturbed datasets and analyze VLA robustness:

```bash
python scripts/launcher.py --config configs/main.yaml
```

This creates perturbed BDDL files, records demonstrations for each perturbation, and computes performance metrics. See [README_LAUNCHER.md](README_LAUNCHER.md) for details.

### Custom Prompts and Context

Customize VLM MM behavior by:

1. **Modifying mm_context/**: Place reference images and videos showing task execution patterns
2. **Updating mm_prompts/**: Create custom prompt templates for different task types
3. **Adjusting model_name**: Switch between VLM models for different performance characteristics

## Troubleshooting

### VLM Server Connection Error

```
Error: Could not connect to http://localhost:8000
```

**Solution**: Ensure the VLM server is running in a separate terminal and using the same port.

### Out of GPU Memory

**Solution**: 
- Use a smaller VLM model (e.g., LLaVA instead of Gemma-3)
- Reduce batch size or image resolution
- Use CPU mode (slower): Set `device: cpu` in config

### BDDL File Not Found

**Solution**: Provide absolute path to BDDL file and ensure the file exists in the specified location.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{vlmmm2026,
  title={VLM Micromanager for Vision-Language Action Models},
  year={2026}
}
```

## License

See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

For questions or issues, please open an issue on the repository or contact the maintainers.

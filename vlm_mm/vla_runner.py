"""
Record OpenVLA demonstrations on LIBERO tasks.

This script runs OpenVLA inference on a LIBERO task and saves:
- HDF5 demonstration file with actions, observations, and rewards
- Supports recording multiple demos in a single HDF5 file
- Can add Gaussian noise to actions to simulate reality

Usage:
    python record.py --config ../configs/inference_config.yaml


The script automatically:
    - Loads the appropriate finetuned OpenVLA model for your task suite
    - Preprocesses images for LIBERO
    - Handles action normalization and gripper control
    - Records multiple demos with different random seeds
    - Adds small Gaussian noise to simulate real-world variability


Configuration
-------------
The script expects a YAML config file with the following top-level keys:

- cache_dir (str): HuggingFace cache directory (e.g. /path/to/huggingface).
- device (str): PyTorch device string (e.g. "cuda:0" or "cpu").
- seed (int): Global RNG seed.

- vlmmm (mapping): VLMMM launcher config used by this pipeline. Required keys:
    model_name: gemma3
    vlm_mm_context_dir: path/to/mm_context
    vlm_mm_prompts_dir: path/to/mm_prompts
    original_prompt: <string prompt given to the VLA/MM>
    curr_image_path: path/to/curr_image.png
    context_video_path: path/to/curr_video_context.mp4

- vlm_server (mapping): VLMM server config used by this script:
    server: http://localhost:8000

- vla (mapping): OpenVLA / LIBERO recording config. Required keys:
    model: openvla
    task_suite_name: libero_10 | libero_spatial | libero_object | ...
    bddl_file: /absolute/path/to/task.bddl
    prompt: textual task prompt (same as original_prompt)
    record_path: path/to/save/recordings/
    out_file: path/to/output/demo.hdf5
    action_scale: float (optional, default 1.0)
    noise_std: float (optional, default 0.0)

"""

import os
import sys
import time
import h5py
import argparse
import json
import yaml
import numpy as np
from PIL import Image
import requests
import time
import cv2
from typing import List
from pathlib import Path

import torch
import transformers
from packaging import version
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer



def _disable_wandb_for_recording():
    """Disable wandb in this script since recording does not use it."""
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_MODE", "disabled")


def _snapshot_download_model(model_id, cache_dir, local_files_only=False):
    """Download model snapshot without deprecated resume_download usage."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)

for _p in [_THIS_DIR, _REPO_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

_libero_env_path = os.environ.get("LIBERO_PATH")
if _libero_env_path and _libero_env_path not in sys.path:
    sys.path.insert(0, _libero_env_path)

try:
    from libero.libero.envs import OffScreenRenderEnv
except ModuleNotFoundError as _e:
    _tried = str(sys.path[:6])
    _msg = (
        f"Could not import libero. Tried sys.path: {_tried}\n"
        "Fix options:\n"
        "  1. Run from repo root:  cd <repo_root> && python scripts/playback.py --config ...\n"
        "  2. Install editable:    pip install -e <repo_root>\n"
        "  3. Set env var:         export LIBERO_PATH=<repo_root>\n"
        f"Original error: {_e}"
    )
    raise ModuleNotFoundError(_msg) from _e

from libero.libero.envs import OffScreenRenderEnv

# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.environment_utils import get_init_state_from_episode, resume_episode

SERVER = None  # will be set from config in main()


def preprocess_image(obs, resize_size=256, center_crop=True):
    """Preprocess image with LIBERO-specific 180-degree rotation."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # Critical: rotate 180 degrees for LIBERO
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize((resize_size, resize_size), Image.LANCZOS)
    
    if center_crop:
        crop_size = 224
        left = (resize_size - crop_size) // 2
        top = (resize_size - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))
    
    return img


def normalize_gripper_action(action, binarize=True):
    """Normalize gripper action from [0,1] to [-1,+1]."""
    action[-1] = 2.0 * action[-1] - 1.0
    if binarize:
        action[-1] = 1.0 if action[-1] > 0 else -1.0
    return action


def invert_gripper_action(action):
    """Invert gripper action sign for LIBERO."""
    action[-1] = -action[-1]
    return action


def load_openvla(task_suite_name, device, cache_dir):
    """Load OpenVLA model and processor with automatic model selection."""
    LIBERO_MODELS = {
        "libero_spatial": "openvla/openvla-7b-finetuned-libero-spatial",
        "libero_object": "openvla/openvla-7b-finetuned-libero-object",
        "libero_goal": "openvla/openvla-7b-finetuned-libero-goal",
        "libero_10": "openvla/openvla-7b-finetuned-libero-10",
    }
    
    model_path = LIBERO_MODELS.get(task_suite_name, "openvla/openvla-7b")
    local_files_only = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true"}
    print(f"Loading model: {model_path}")
    
    local_model_path = _snapshot_download_model(
        model_path,
        cache_dir,
        local_files_only=local_files_only,
    )

    processor = AutoProcessor.from_pretrained(
        local_model_path,
        # add_prefix_space=True, # trying to get back legacy behavior
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        local_model_path,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        # torch_dtype='auto',
        # attn_implementation="eager",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=local_files_only,
    ).to(device)
    
    # Load norm_stats if not already present
    if not hasattr(vla, 'norm_stats'):
        import glob
        if cache_dir:
            pattern = os.path.join(cache_dir, f"models--{model_path.replace('/', '--')}", 
                                 "snapshots", "*", "dataset_statistics.json")
            matches = glob.glob(pattern)
            if matches:
                with open(matches[0], "r") as f:
                    vla.norm_stats = json.load(f)
                print(f"Loaded norm_stats from cache")
    
    return processor, vla


def add_gaussian_noise(action, noise_std):
    """Add Gaussian noise to action to simulate reality."""
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, size=action.shape)
        # Only add noise to the first 6 dimensions (position/orientation deltas)
        # Don't add noise to gripper action (last dimension)
        action[:6] = action[:6] + noise[:6]
    return action

def save_demo_video(frames: List[np.ndarray], output_path: str, fps: int = 20) -> None:
    """Save a list of (H, W, 3) uint8 frames as an MP4 video."""
    if not frames:
        print(f"  ⚠ No frames to save, skipping video.")
        return
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()
    print(f"  ✓ Saved video: {output_path} ({len(frames)} frames)")

def save_curr_video_context(frames: List[np.ndarray], output_path: str, fps: int = 20) -> None:
    """
    Save current context frames to a video file using OpenCV.

    Args:
        frames: list of HxWx3 uint8 RGB frames (numpy arrays)
        output_path: path to write the MP4 file
        fps: frames per second
    """
    if not frames:
        print("  ⚠ No context frames to save, skipping save_curr_video_context.")
        return

    out_path = Path(output_path)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))

    for idx, frame in enumerate(frames):
        if frame.dtype != np.uint8:
            frame_to_write = (np.clip(frame, 0, 255)).astype(np.uint8)
        else:
            frame_to_write = frame
        # Convert RGB -> BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_to_write, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"  ✓ Saved current context video to {out_path} ({len(frames)} frames, {fps} fps)")

def get_vlmmm_response(call="run_step", debug=False):
    """Query VLMMM server for action prediction."""
    assert call in ["run_step", "analyse_performance"], "Invalid VLMMM call"
    try:
        payload = {"debug": debug}

        start = time.perf_counter()
        r = requests.post(f"{SERVER}/{call}", json=payload, timeout=300)
        elapsed = time.perf_counter() - start

        r.raise_for_status()
        resp = r.json()

        print(f"  ✓ VLMMM response time: {elapsed:.2f}s")
        return resp.get("outputs", {})

    except Exception as e:
        print(f"  ⚠ Error querying VLMMM server: {e}")
        return {}


def run_vla_episode(env, processor, vla, config, use_vlmmm=False):
    """Record a single demonstration."""
    seed = config.get("seed", 42)
    np.random.seed(seed)
    env.seed(seed)
    obs = env.reset()
    for _ in range(10):
        obs, _, _, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            

    # Set max steps based on task suite
    max_steps_dict = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        # "libero_10": 520,
        "libero_10": 300,
        "libero_90": 400,
    }
    max_steps = max_steps_dict.get(config["vla"]["task_suite_name"], 200)
    action_scale = config["vla"].get("action_scale", 1.0)
    noise_std = config["vla"].get("noise_std", 0.0)
    
    # Initialize recording
    actions, dones, rewards, states, obs_list, frames = [], [], [], [], [], []
    curr_context_frames = []
    step = 0
    done = False
    # no_movement_counter = 0
    query_vlmmm_countdown = 0
    first_call = True
    
    print(f"  Starting policy rollout (max {max_steps} steps, noise_std={noise_std})...")
    
    while not done and step < max_steps:
        # Preprocess image
        img = preprocess_image(obs, resize_size=256, center_crop=True)
        frames.append(np.array(img))
        curr_context_frames.append(np.array(img))
        img.save(config["vlmmm"]["curr_image_path"])  # Save current image for VLMMM context

        # Get action from model or from VLMMM
        prompt, action = None, None

        if (use_vlmmm and query_vlmmm_countdown <= 0):
            ### TODO: create video using current video frames and save to config["vlmmm"]["curr_video_path"] for VLMMM context
            if (not first_call):
                print(f"  Querying VLMMM server for step {step} analysis and action...")
                save_curr_video_context(curr_context_frames, config["vlmmm"]["context_video_path"], fps=20)
                vlmmm_step_outputs = get_vlmmm_response(call="analyse_performance", debug=False)
                curr_context_frames = [] # reset context frames after saving video for VLMMM
                print(f"  VLMMM analysis: {vlmmm_step_outputs.get('analysis', {})}")
            else:
                first_call = False

            print(f"  Querying VLMMM server for next step prediction...")
            vlmmm_step_outputs = get_vlmmm_response(call="run_step", debug=True)
            print(f"  VLMMM run_step response: {vlmmm_step_outputs}")
            query_vlmmm_countdown = vlmmm_step_outputs["num_steps_to_spend"]
            if (not vlmmm_step_outputs["is_robot_struggling"] and vlmmm_step_outputs["use_revised_prompt"]):
                prompt = vlmmm_step_outputs["revised_prompt"]
                print(f"  VLMMM revised prompt: {prompt}")
            elif (not vlmmm_step_outputs["is_robot_struggling"] and vlmmm_step_outputs["use_override_action"]):
                action = override_action
            
            if (vlmmm_step_outputs["create_smaller_tasks"] or vlmmm_step_outputs["is_current_subtask_done"]):
                first_call = True  # reset VLMMM state for next subtask


        # NOTE TODO: I do not think the revised prompts and override actions are being used. Recheck logic
        # NOTE TODO: Need to fix analysis prompt - it is not seeing stuff properly (maybe video is not getting updated?)
        # NOTE TODO: fix step prompt to make override actions and reasoning make sense.


        
        if prompt is None:
            prompt = f"In: What action should the robot take to {config['vla']['prompt']}?\nOut:"
        if action is None:
            inputs = processor(prompt, img).to(config["vla"].get("device", "cuda:0"), dtype=torch.bfloat16)
            action = vla.predict_action(**inputs, unnorm_key=config["vla"]["task_suite_name"], do_sample=False)

            # Process gripper and apply scaling
            action = normalize_gripper_action(action, binarize=True)
            action = invert_gripper_action(action)
            if action_scale != 1.0:
                action[:6] = action[:6] * action_scale
        
        # Add Gaussian noise to simulate reality
        action = add_gaussian_noise(action, noise_std)
        print(f"  Step {step + 1}/{max_steps}, Action (pre-noise): {list(action)}, Noise std: {noise_std}")
        
        # Execute action
        obs_new, reward, done, info = env.step(action.tolist())
        obs = obs_new
        
        # Save data
        flat_state = np.concatenate([
            np.ravel(obs[k]) for k in sorted(obs.keys()) if not k.endswith("image")
        ])
        actions.append(action)
        dones.append(done)
        rewards.append(reward)
        states.append(flat_state)
        obs_list.append({k: np.array(v) for k, v in obs.items() if not k.endswith("image")})

        print(f"  Step {step}/{max_steps}, Reward: {reward:.2f}, Done: {done}, Action: {action}")

        query_vlmmm_countdown -= 1
        step += 1
        if step % 10 == 0 or done:
            print(f"  Step {step}/{max_steps}, Reward: {reward:.2f}, Done: {done}")
        
        # if (len(actions) > 1 and np.allclose(actions[-1][:6], [0,0,0,0,0,0], atol=1e-2)):
        #     no_movement_counter += 1
        
        # if no_movement_counter >= 10:
        #     print(f"  No significant movement detected for 10 steps, ending demo early.")
        #     break
    
    demo_data = {
        "actions": np.array(actions),
        "dones": np.array(dones, dtype=bool),
        "rewards": np.array(rewards),
        "states": np.array(states, dtype=np.float32),
        "obs_list": obs_list,
        "frames": frames,
    }
    
    print(f"  ✓ Episode completed with {len(actions)} steps, final reward: {reward:.2f}")
    
    return demo_data


def record_demo(config):
    """Record demonstration(s) using OpenVLA."""
    env_args = {
        "bddl_file_name": config["vla"]["bddl_file"],
        "camera_heights": 256,
        "camera_widths": 256,
    }
    
    print("Initializing environment...")
    env = OffScreenRenderEnv(**env_args)
    
    print("Loading model...")
    processor, vla = load_openvla(
        config["vla"]["task_suite_name"],
        config.get("device", "cuda:0"),
        config["cache_dir"]
    )
    
    demo_data = run_vla_episode(env, processor, vla, config, use_vlmmm=True)
    
    env.close()

    # ---- Save HDF5 ----
    out_file = config["vla"]["out_file"]
    print(f"\nSaving episode HDF5 to {out_file}...")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with h5py.File(out_file, "w") as f:
        dset = f.create_group(f"data/")
        dset.create_dataset("actions", data=demo_data["actions"], compression="gzip")
        dset.create_dataset("dones", data=demo_data["dones"], compression="gzip")
        dset.create_dataset("rewards", data=demo_data["rewards"], compression="gzip")
        dset.create_dataset("states", data=demo_data["states"], compression="gzip")

        obs_grp = dset.create_group("obs")
        for k in demo_data["obs_list"][0].keys():
            obs_stack = np.stack(
                [step_obs[k] for step_obs in demo_data["obs_list"]], axis=0
            )
            obs_grp.create_dataset(k, data=obs_stack, compression="gzip")
    
    print(f"✓ Saved episode HDF5 file to {out_file}")

    # ---- Save videos ----
    print(f"\nSaving video...")
    video_path = config["vla"]["record_path"]
    save_demo_video(demo_data["frames"], video_path, fps=20)
    print("\n✓ Recording and video export complete!")

    return demo_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record OpenVLA demonstrations on LIBERO tasks")
    parser.add_argument("--config", type=str, required=True, help="Path to inference_config.yaml")
    args = parser.parse_args()

    _disable_wandb_for_recording()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    assert config["vla"]["prompt"] == config["vlmmm"]["original_prompt"], "VLA prompt must match VLMMM original_prompt for consistency."

    SERVER = config["vlm_server"]["server"]
    
    print("=" * 80)
    print("OpenVLA LIBERO Recording")
    print("=" * 80)
    print(f"Task suite: {config['vla']['task_suite_name']}")
    print(f"BDDL file: {config['vla']['bddl_file']}")
    print(f"Prompt: {config['vla']['prompt']}")
    print(f"Output: {config['vla']['out_file']}")
    print("=" * 80)
    
    record_demo(config)
    
    print("\n✓ Recording complete!")

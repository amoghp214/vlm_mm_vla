"""
Record OpenVLA demonstrations on LIBERO tasks.

This script runs OpenVLA inference on a LIBERO task and saves:
- HDF5 demonstration file with actions, observations, and rewards
- Supports recording multiple demos in a single HDF5 file
- Can add Gaussian noise to actions to simulate reality

Usage:
    python record.py --config ../configs/inference_config.yaml

Configuration:
    Edit your config YAML to specify:
    - Model and task suite
    - BDDL file and task prompt
    - Output path for demo HDF5 file
    - num_demos: Number of demonstrations to record (default: 1)
    - noise_std: Gaussian noise std deviation for actions (default: 0.0)
    - Optional action scaling

The script automatically:
    - Loads the appropriate finetuned OpenVLA model for your task suite
    - Preprocesses images for LIBERO
    - Handles action normalization and gripper control
    - Records multiple demos with different random seeds
    - Adds small Gaussian noise to simulate real-world variability

Note: To create videos from recorded demos, use playback.py with the same config file.
"""

import os
import time
import h5py
import argparse
import json
import yaml
import numpy as np
from PIL import Image

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

from libero.libero.envs import OffScreenRenderEnv

# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.environment_utils import get_init_state_from_episode, resume_episode


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

    # tokenizer_kwargs = {}
    # if version.parse(transformers.__version__) >= version.parse("4.57.0"):
    #     # Preserve legacy tokenization behavior for OpenVLA action decoding.
    #     tokenizer_kwargs.update({"use_fast": False, "legacy": True})
    # try:
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         local_model_path,
    #         trust_remote_code=True,
    #         local_files_only=local_files_only,
    #         **tokenizer_kwargs,
    #     )
    # except TypeError:
    #     tokenizer_kwargs.pop("legacy", None)
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         local_model_path,
    #         trust_remote_code=True,
    #         local_files_only=local_files_only,
    #         **tokenizer_kwargs,
    #     )
    # processor.tokenizer = tokenizer
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


def record_single_demo(env, processor, vla, config, demo_index, seed, episode_to_resume=None):
    """Record a single demonstration."""
    # Set random seed for this demo
    np.random.seed(seed)
    env.seed(seed)
    if episode_to_resume and os.path.isfile(episode_to_resume):
        init_state = get_init_state_from_episode(episode_to_resume, demo_index=demo_index)
        env.set_init_state(init_state)
        obs = env.reset()
    else:
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
    max_steps = max_steps_dict.get(config["task_suite_name"], 200)
    action_scale = config.get("action_scale", 1.0)
    noise_std = config.get("noise_std", 0.0)
    
    # Initialize recording
    actions, dones, rewards, states, obs_list = [], [], [], [], []
    step = 0
    done = False

    # for _ in range(20):
    #     action = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    #     obs, reward, done, info = env.step(action)
    #     flat_state = np.concatenate([
    #         np.ravel(obs[k]) for k in sorted(obs.keys()) if not k.endswith("image")
    #     ])
    #     actions.append(action)
    #     dones.append(done)
    #     rewards.append(reward)
    #     states.append(flat_state)
    #     obs_list.append({k: np.array(v) for k, v in obs.items() if not k.endswith("image")})
    for _ in range(20):
        action = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        time.sleep(0.1)  # Small delay to ensure environment is ready
        obs, reward, done, info = env.step(action)
        flat_state = np.concatenate([
            np.ravel(obs[k]) for k in sorted(obs.keys()) if not k.endswith("image")
        ])
        actions.append(action)
        dones.append(done)
        rewards.append(reward)
        states.append(flat_state)
        obs_list.append({k: np.array(v) for k, v in obs.items() if not k.endswith("image")})
    # for _ in range(20):
    #     action = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    #     time.sleep(0.1)  # Small delay to ensure environment is ready
    #     obs, reward, done, info = env.step(action)
    #     flat_state = np.concatenate([
    #         np.ravel(obs[k]) for k in sorted(obs.keys()) if not k.endswith("image")
    #     ])
    #     actions.append(action)
    #     dones.append(done)
    #     rewards.append(reward)
    #     states.append(flat_state)
    #     obs_list.append({k: np.array(v) for k, v in obs.items() if not k.endswith("image")})
    # for _ in range(20):
    #     action = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    #     obs, reward, done, info = env.step(action)
    #     flat_state = np.concatenate([
    #         np.ravel(obs[k]) for k in sorted(obs.keys()) if not k.endswith("image")
    #     ])
    #     actions.append(action)
    #     dones.append(done)
    #     rewards.append(reward)
    #     states.append(flat_state)
    #     obs_list.append({k: np.array(v) for k, v in obs.items() if not k.endswith("image")})
    # for _ in range(30):
    #     action = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     time.sleep(0.1)  # Small delay to ensure environment is ready
    #     obs, reward, done, info = env.step(action)
    #     flat_state = np.concatenate([
    #         np.ravel(obs[k]) for k in sorted(obs.keys()) if not k.endswith("image")
    #     ])
    #     actions.append(action)
    #     dones.append(done)
    #     rewards.append(reward)
    #     states.append(flat_state)
    #     obs_list.append({k: np.array(v) for k, v in obs.items() if not k.endswith("image")})
    # for _ in range(30):
    #     action = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     time.sleep(0.1)  # Small delay to ensure environment is ready
    #     img = preprocess_image(obs, resize_size=256, center_crop=True)
    #     img.save(f"/home/hice1/apalasamudram6/scratch/vlm_mm_vla/test/last_step.png")
    #     obs, reward, done, info = env.step(action)
    #     flat_state = np.concatenate([
    #         np.ravel(obs[k]) for k in sorted(obs.keys()) if not k.endswith("image")
    #     ])
    #     actions.append(action)
    #     dones.append(done)
    #     rewards.append(reward)
    #     states.append(flat_state)
    #     obs_list.append({k: np.array(v) for k, v in obs.items() if not k.endswith("image")})
    
    # print(f"  Starting policy rollout (max {max_steps} steps, noise_std={noise_std})...")

    if (episode_to_resume and os.path.isfile(episode_to_resume)):
        print(f"Resuming from episode: {episode_to_resume}")
        prev_actions, prev_dones, prev_rewards, prev_states, prev_obs_list, obs = resume_episode(env, episode_to_resume, demo_index=demo_index)
        actions.extend(prev_actions)
        dones.extend(prev_dones)
        rewards.extend(prev_rewards)
        states.extend(prev_states)
        obs_list.extend(prev_obs_list)
        step = len(prev_actions)
    no_movement_counter = 0
    while not done and step < max_steps:
        # Preprocess image
        img = preprocess_image(obs, resize_size=256, center_crop=True)
        img.save(f"/home/hice1/apalasamudram6/scratch/vlm_mm_vla/test/last_step.png")
        
        # Get action from model
        prompt = f"In: What action should the robot take to {config['prompt']}?\nOut:"
        inputs = processor(prompt, img).to(config.get("device", "cuda:0"), dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key=config["task_suite_name"], do_sample=False)
        # if step < 159: action = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0])

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
        # print(f"  Step {step + 1}/{max_steps}, Reward: {reward:.2f}, Done: {done}, Action: {action}")
        step += 1
        if step % 10 == 0 or done:
            print(f"  Step {step}/{max_steps}, Reward: {reward:.2f}, Done: {done}")
        
        # if (len(actions) > 1 and np.allclose(actions[-1][:6], actions[-2][:6], atol=5e-3)): # should this be with respect to the previous action or just based on the magnitude of the current action?
        if (len(actions) > 1 and np.allclose(actions[-1][:6], [0,0,0,0,0,0], atol=1e-2)):
            no_movement_counter += 1
        
        if no_movement_counter >= 10:
            print(f"  No significant movement detected for 10 steps, ending demo early.")
            break
    
    demo_data = {
        "actions": np.array(actions),
        "dones": np.array(dones, dtype=bool),
        "rewards": np.array(rewards),
        "states": np.array(states, dtype=np.float32),
        "obs_list": obs_list
    }
    
    print(f"  ✓ Demo {demo_index} completed with {len(actions)} steps, final reward: {reward:.2f}")
    
    return demo_data


def record_demo(config, episode_to_resume=None):
    """Record demonstration(s) using OpenVLA."""
    env_args = {
        "bddl_file_name": config["bddl_file"],
        "camera_heights": 256,
        "camera_widths": 256,
    }
    
    print("Initializing environment...")
    env = OffScreenRenderEnv(**env_args)
    
    print("Loading model...")
    processor, vla = load_openvla(
        config["task_suite_name"],
        config.get("device", "cuda:0"),
        config["cache_dir"]
    )
    
    # Get number of demos to record
    num_demos = config.get("num_demos", 1)
    print(f"\nRecording {num_demos} demonstration(s)...")
    
    # Record all demos
    all_demos = []
    for demo_idx in range(num_demos):
        print(f"\n{'='*60}")
        print(f"Recording demo {demo_idx + 1}/{num_demos}")
        print(f"{'='*60}")
        
        # Use different seed for each demo
        seed = demo_idx
        demo_data = record_single_demo(env, processor, vla, config, demo_idx, seed, episode_to_resume)
        all_demos.append(demo_data)
    
    env.close()
    
    # Save all demos to HDF5
    print(f"\nSaving {num_demos} demo(s) to {config['out_file']}...")
    os.makedirs(os.path.dirname(config['out_file']), exist_ok=True)
    
    with h5py.File(config['out_file'], "w") as f:
        for demo_idx, demo_data in enumerate(all_demos):
            dset = f.create_group(f"data/demo_{demo_idx}")
            dset.create_dataset("actions", data=demo_data["actions"], compression="gzip")
            dset.create_dataset("dones", data=demo_data["dones"], compression="gzip")
            dset.create_dataset("rewards", data=demo_data["rewards"], compression="gzip")
            dset.create_dataset("states", data=demo_data["states"], compression="gzip")
            
            obs_grp = dset.create_group("obs")
            for k in demo_data["obs_list"][0].keys():
                obs_stack = np.stack([step_obs[k] for step_obs in demo_data["obs_list"]], axis=0)
                obs_grp.create_dataset(k, data=obs_stack, compression="gzip")

    print(f"✓ Saved {num_demos} demo(s) to HDF5 file")


def main():
    parser = argparse.ArgumentParser(description="Record OpenVLA demonstrations on LIBERO tasks")
    parser.add_argument("--config", type=str, required=True, help="Path to inference_config.yaml")
    parser.add_argument("--episode_to_resume", type=str, default="", help="Path to HDF5 episode to resume")
    args = parser.parse_args()

    _disable_wandb_for_recording()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 80)
    print("OpenVLA LIBERO Recording")
    print("=" * 80)
    print(f"Task suite: {config['task_suite_name']}")
    print(f"BDDL file: {config['bddl_file']}")
    print(f"Prompt: {config['prompt']}")
    print(f"Output: {config['out_file']}")
    print("=" * 80)
    
    episode_to_resume = args.episode_to_resume or config.get("episode_to_resume", "")
    record_demo(config, episode_to_resume)
    
    print("\n✓ Recording complete!")


if __name__ == "__main__":
    main()

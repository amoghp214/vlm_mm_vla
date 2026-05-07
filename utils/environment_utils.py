import h5py
import numpy as np

from PIL import Image
from libero.libero.envs import OffScreenRenderEnv


def extract_final_state_from_episode(bddl_file, episode_hdf5_path):
    """
    Extract the final MuJoCo state by replaying the last demo in an HDF5 episode file.

    This mirrors playback behavior: set initial state from the first stored state and
    step through actions to the end, then return the final simulator state.
    """
    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(0)

    try:
        with h5py.File(episode_hdf5_path, "r") as f:
            if "data" not in f:
                raise ValueError("HDF5 file missing 'data' group")

            demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
            if not demo_keys:
                raise ValueError("No demos found in HDF5 file")
            print(demo_keys)
            exit()

            demo_key = f"data/{demo_keys[-1]}"
            actions = f[demo_key]["actions"][()]

            if "sim_states" in f[demo_key]:
                init_state = f[demo_key]["sim_states"][0]
            elif "states" in f[demo_key]:
                init_state = f[demo_key]["states"][0]
            else:
                raise ValueError("No 'sim_states' or 'states' dataset found for demo")

        env.reset()
        env.regenerate_obs_from_state(np.array(init_state, dtype=np.float64))

        for action in actions:
            obs, reward, done, info = env.step(action)
            if done:
                break

        final_state = env.get_sim_state()
        return final_state
    finally:
        env.close()

def get_init_state_from_episode(episode_hdf5_path, demo_index=-1):
    """
    Extract the initial state from the first demo in an HDF5 episode file.
    """
    with h5py.File(episode_hdf5_path, "r") as f:
        if "data" not in f:
            raise ValueError("HDF5 file missing 'data' group")

        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
        if not demo_keys:
            raise ValueError("No demos found in HDF5 file")
        demo_key = f"data/{demo_keys[demo_index]}"

        init_state = f[demo_key]["states"][0]
    return init_state

def resume_episode(env, demo_file, demo_index=-1):
    """
    Render a single demo (HDF5 + BDDL) into a video.
    """
    # Load HDF5 demo
    with h5py.File(demo_file, "r") as f:
        if "data" not in f:
            raise ValueError("HDF5 file missing 'data' group")

        demo_keys = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
        if not demo_keys:
            raise ValueError("No demos found in HDF5 file")
        demo_key = f"data/{demo_keys[demo_index]}"

        actions = f[demo_key]["actions"][()]
        init_state = f[demo_key]["states"][0]

    # Step through actions
    obs = None
    dones = []
    rewards = []
    states = []
    obs_list = []
    for i, action in enumerate(actions):
        # print(f"  Frame {i + 1}/{len(actions)}, Action: {list(action)}")
        if (i + 1) % 10 == 0:
            print(f"  Frame {i + 1}/{len(actions)}")

        obs, reward, done, info = env.step(action)

        flat_state = np.concatenate([
            np.ravel(obs[k]) for k in sorted(obs.keys()) if not k.endswith("image")
        ])

        dones.append(done)
        rewards.append(reward)
        states.append(flat_state)
        obs_list.append({k: np.array(v) for k, v in obs.items() if not k.endswith("image")})
        if done:
            print(f"  Demo finished at step {i}")
            break
        # if (i >= 70): break

    img = preprocess_image(obs, resize_size=256, center_crop=True)
    img.save(f"/home/hice1/apalasamudram6/scratch/vlm_mm_vla/test/demo_{demo_index}_last_frame.png")
    # return actions, dones, rewards, states, obs_list, obs
    return actions[:len(dones)], dones[:len(dones)], rewards[:len(dones)], states[:len(dones)], obs_list[:len(dones)], obs


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
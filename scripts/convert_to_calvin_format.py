from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def main():
    # obj = np.load("../mdt_policy/dataset/task_ABC_D/training/episode_0000000.npz")
    # print(list(obj.keys()))
    # ['actions', 'rel_actions', 'robot_obs', 'scene_obs', 'rgb_static', 'rgb_gripper', 'rgb_tactile', 'depth_static', 'depth_gripper', 'depth_tactile']
    # for k, v in obj.items():
    #     print(k, v.shape, v.dtype)
    """
actions (7,) float64
rel_actions (7,) float64
robot_obs (15,) float64
scene_obs (24,) float64
rgb_static (200, 200, 3) uint8
rgb_gripper (84, 84, 3) uint8
rgb_tactile (160, 120, 6) uint8
depth_static (200, 200) float32
depth_gripper (84, 84) float32
depth_tactile (160, 120, 2) float32
        """

    root_path = Path("libero/datasets")
    save_root_path = Path("libero/datasets/calvin_format")

    for split, target_split in [("libero_90", "validation"), ("libero_90", "training")]:
        split_dir = root_path / split

        print("=" * 80)
        print(f"Processing {split}")
        print("=" * 80)

        # ep_start_end_ids.npy
        # auto_lang_ann.npy

        idx = 0

        ep_start_end_ids = []
        instrs = []

        scene_paths = []
        for scene_path in split_dir.glob("*.hdf5"):
            instruction = scene_path.stem

            instruction = instruction.split("_")[:2]

            if instruction[0] == "KITCHEN" and instruction[1] == "SCENE4":
                if target_split == "validation":
                    scene_paths.append(scene_path)
            else:
                if target_split == "training":
                    scene_paths.append(scene_path)

        print(scene_paths)

        for scene_path in scene_paths:
            instruction = scene_path.stem

            instruction = instruction.split("_")
            instruction = filter(lambda x: not x.isupper(), instruction)
            instruction = filter(lambda x: x != "demo", instruction)
            instruction = " ".join(instruction)
            instruction = instruction[0].upper() + instruction[1:] + "."

            print(instruction)

            with h5py.File(str(scene_path), "r") as f:
                for k, v in tqdm(f["data"].items()):
                    actions: np.ndarray = v["actions"][:]
                    # dones: np.ndarray = v["dones"][:]
                    obs: dict[str, np.ndarray] = {kk: vv[:] for kk, vv in v["obs"].items()}
                    # robot_states: np.ndarray = v["robot_states"][:]
                    # states: np.ndarray = v["states"][:]

                    padded_actions = np.zeros((actions.shape[0] + 1, actions.shape[1]), dtype=actions.dtype)
                    padded_actions[1:] = actions
                    padded_actions[0] = actions[0]

                    robot_obs = np.concatenate([
                        obs["ee_pos"],
                        obs["ee_ori"],
                        obs["gripper_states"][:, 0:1],
                        obs["joint_states"],
                        padded_actions[:-1, -1:],
                    ], axis=-1)

                    assert robot_obs.shape[1] == 15

                    num_frames = actions.shape[0]

                    ep_start_end_ids.append((idx, idx + num_frames - 1))

                    instrs.append(instruction)

                    for i in range(num_frames):
                        sample = {
                            "action": actions[i],
                            "rel_actions": actions[i],
                            "rgb_static": obs["agentview_rgb"][i],
                            "rgb_gripper": obs["eye_in_hand_rgb"][i],
                            "robot_obs": robot_obs[i],
                        }

                        save_path = save_root_path / target_split / f"episode_{idx:07d}.npz"
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(save_path, **sample)

                        idx += 1

        auto_lang_ann = {
            "info": {
                "indx": ep_start_end_ids,
            },
            "language": {
                "emb": [
                    np.zeros((1, 2), dtype=np.float32)
                    for _ in range(len(instrs))
                ],
                "ann": instrs,
            },
        }

        auto_lang_ann_path = save_root_path / target_split / "lang_annotations" / "auto_lang_ann.npy"
        auto_lang_ann_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(auto_lang_ann_path, np.array(auto_lang_ann, dtype=object))

        np.save(save_root_path / target_split / "ep_start_end_ids.npy", np.array(ep_start_end_ids))


if __name__ == "__main__":
    main()

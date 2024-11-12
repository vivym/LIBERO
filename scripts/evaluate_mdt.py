from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import torch
from torchvision import transforms as T
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from mdt.get_mdt_model import get_mdt
from mdt.models.edm_dp_agent import EDMDPAgent


def capitalize_and_period(instr: str) -> str:
    """
    Capitalize the first letter of a string and add a period to the end if it's not there.
    """
    if len(instr) > 0:
        # if the first letter is not capital, make it so
        if not instr[0].isupper():
            # if the first letter is not capital, make it so
            instr = instr[0].upper() + instr[1:]
        # add period to the end if it's not there
        if instr[-1] != '.':
            # add period to the end if it's not there
            instr = instr + '.'
    return instr


def save_to_video(frames, filename):
    container = av.open(str(filename), mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width = 128
    stream.height = 128
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23"}

    for frame in frames:
        frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


def save_to_videos(
    save_dir: Path,
    rgb_static_history: list[list[np.ndarray]],
    rgb_gripper_history: list[list[np.ndarray]],
    dones: np.ndarray,
):
    for env_idx in tqdm(range(len(rgb_static_history[0])), desc="Saving videos"):
        done = dones[env_idx].item()
        suffix = "success" if done else "fail"

        rgb_static_history_i = [rgb_static[env_idx] for rgb_static in rgb_static_history]
        rgb_gripper_history_i = [rgb_gripper[env_idx] for rgb_gripper in rgb_gripper_history]

        save_to_video(rgb_static_history_i, save_dir / f"rgb_static_{env_idx:02d}_{suffix}.mp4")
        save_to_video(rgb_gripper_history_i, save_dir / f"rgb_gripper_{env_idx:02d}_{suffix}.mp4")


def transform_rgb_obs(
    rgb_static_list: list[list[np.ndarray]],
    rgb_gripper_list: list[list[np.ndarray]],
) -> np.ndarray:
    rgb_static = np.array(rgb_static_list)
    rgb_gripper = np.array(rgb_gripper_list)

    rgb_static = rgb_static.transpose(1, 0, 2, 3, 4)
    rgb_gripper = rgb_gripper.transpose(1, 0, 2, 3, 4)

    rgb_static = torch.from_numpy(rgb_static)
    rgb_gripper = torch.from_numpy(rgb_gripper)

    rgb_static = rgb_static.float() / 255.0
    rgb_gripper = rgb_gripper.float() / 255.0

    rgb_static = rgb_static.permute(0, 1, 4, 2, 3)
    rgb_gripper = rgb_gripper.permute(0, 1, 4, 2, 3)

    batch_size = rgb_static.shape[0]

    rgb_static = rgb_static.flatten(0, 1)
    rgb_gripper = rgb_gripper.flatten(0, 1)

    transform1 = T.Compose([
        T.Resize(224),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    transform2 = T.Compose([
        T.Resize(84),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


    rgb_static = transform1(rgb_static)
    rgb_gripper = transform2(rgb_gripper)

    rgb_static = rgb_static.view(batch_size, -1, *rgb_static.shape[1:])
    rgb_gripper = rgb_gripper.view(batch_size, -1, *rgb_gripper.shape[1:])

    return {"rgb_static": rgb_static, "rgb_gripper": rgb_gripper}


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "True"

    device = torch.device("cuda")

    policy: EDMDPAgent = get_mdt()
    policy.eval()
    policy.to(device)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_90"
    task_suite = benchmark_dict[task_suite_name]()

    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    succ_list = []
    for task_id in range(2, 90):
        # retrieve a specific task
        task = task_suite.get_task(task_id)
        task_description: str = task.language

        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        print("-" * 80)
        print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
            f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")
        print("-" * 80)

        init_states = task_suite.get_task_init_states(task_id)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 128,
            "camera_widths": 128
        }

        env_num = 20
        env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
        )
        env.reset()
        env.seed(10000)

        indices = np.arange(env_num) % init_states.shape[0]
        init_states = init_states[indices]

        env.set_init_state(init_states)

        # TODO: policy reset
        policy.reset()

        rgb_static_history = []
        rgb_gripper_history = []

        dummy_action = np.zeros((env_num, 7))
        for step in range(20):
            obs, reward, done, info = env.step(dummy_action)
            if step >= 18:
                rgb_static_history.append([o["agentview_image"] for o in obs])
                rgb_gripper_history.append([o["robot0_eye_in_hand_image"] for o in obs])

        task_description = task_description.strip()
        task_description = capitalize_and_period(task_description)

        print("+" * 80)
        print("task_description", task_description)
        print("+" * 80)

        dones = np.zeros(env_num, dtype=bool)

        for step in tqdm(range(600)):
            with torch.inference_mode():
                actions = policy.step(
                    obs={"rgb_obs": transform_rgb_obs(rgb_static_history[-1:], rgb_gripper_history[-1:])},
                    goal={
                        "lang": [True] * env_num,
                        "lang_text": [task_description] * env_num,
                    },
                )

            gripper_open = actions[:, -1:]
            gripper_open[gripper_open <= 0] = -1
            gripper_open[gripper_open > 0] = 1
            actions = np.concatenate([actions[:, :-1], gripper_open], axis=-1)

            obs, reward, done, info = env.step(actions)
            dones = dones | np.array(done)
            rgb_static_history.append([o["agentview_image"] for o in obs])
            rgb_gripper_history.append([o["robot0_eye_in_hand_image"] for o in obs])

            if np.all(dones):
                print("All environments are done.")
                break

            if np.any(dones):
                print("Some environments are done.", np.sum(dones))

        print("Task done:", dones)

        succ_list.append(dones.tolist())

        env.close()

        print("Saving videos...")
        save_path = Path("outputs") / "rollout" / "mdt" / time_str / f"task_{task_id}"
        save_path.mkdir(parents=True, exist_ok=True)
        save_to_videos(save_path, rgb_static_history, rgb_gripper_history, dones)

    with open(f"outputs/rollout/mdt/{time_str}/succ_list.json", "w") as f:
        json.dump(succ_list, f)

    all_succ = np.array(succ_list)

    print("Total success rate:", np.mean(all_succ))

    for i, succ in enumerate(all_succ):
        print(f"Task {i} success rate:", np.mean(succ))


if __name__ == "__main__":
    main()

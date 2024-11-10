from __future__ import annotations

import os
import hashlib
from pathlib import Path

import av
import yaml
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.rdt.rdt_model import create_model

STATE_VEC_IDX_MAPPING = {
    # [0, 10): right arm joint positions
    **{
        'arm_joint_{}_pos'.format(i): i for i in range(10)
    },
    **{
        'right_arm_joint_{}_pos'.format(i): i for i in range(10)
    },
    # [10, 15): right gripper joint positions
    **{
        'gripper_joint_{}_pos'.format(i): i + 10 for i in range(5)
    },
    **{
        'right_gripper_joint_{}_pos'.format(i): i + 10 for i in range(5)
    },
    'gripper_open': 10, # alias of right_gripper_joint_0_pos
    'right_gripper_open': 10,
    # [15, 25): right arm joint velocities
    **{
        'arm_joint_{}_vel'.format(i): i + 15 for i in range(10)
    },
    **{
        'right_arm_joint_{}_vel'.format(i): i + 15 for i in range(10)
    },
    # [25, 30): right gripper joint velocities
    **{
        'gripper_joint_{}_vel'.format(i): i + 25 for i in range(5)
    },
    **{
        'right_gripper_joint_{}_vel'.format(i): i + 25 for i in range(5)
    },
    'gripper_open_vel': 25, # alias of right_gripper_joint_0_vel
    'right_gripper_open_vel': 25,
    # [30, 33): right end effector positions
    'eef_pos_x': 30,
    'right_eef_pos_x': 30,
    'eef_pos_y': 31,
    'right_eef_pos_y': 31,
    'eef_pos_z': 32,
    'right_eef_pos_z': 32,
    # [33, 39): right end effector 6D pose
    'eef_angle_0': 33,
    'right_eef_angle_0': 33,
    'eef_angle_1': 34,
    'right_eef_angle_1': 34,
    'eef_angle_2': 35,
    'right_eef_angle_2': 35,
    'eef_angle_3': 36,
    'right_eef_angle_3': 36,
    'eef_angle_4': 37,
    'right_eef_angle_4': 37,
    'eef_angle_5': 38,
    'right_eef_angle_5': 38,
    # [39, 42): right end effector velocities
    'eef_vel_x': 39,
    'right_eef_vel_x': 39,
    'eef_vel_y': 40,
    'right_eef_vel_y': 40,
    'eef_vel_z': 41,
    'right_eef_vel_z': 41,
    # [42, 45): right end effector angular velocities
    'eef_angular_vel_roll': 42,
    'right_eef_angular_vel_roll': 42,
    'eef_angular_vel_pitch': 43,
    'right_eef_angular_vel_pitch': 43,
    'eef_angular_vel_yaw': 44,
    'right_eef_angular_vel_yaw': 44,
    # [45, 50): reserved
    # [50, 60): left arm joint positions
    **{
        'left_arm_joint_{}_pos'.format(i): i + 50 for i in range(10)
    },
    # [60, 65): left gripper joint positions
    **{
        'left_gripper_joint_{}_pos'.format(i): i + 60 for i in range(5)
    },
    'left_gripper_open': 60, # alias of left_gripper_joint_0_pos
    # [65, 75): left arm joint velocities
    **{
        'left_arm_joint_{}_vel'.format(i): i + 65 for i in range(10)
    },
    # [75, 80): left gripper joint velocities
    **{
        'left_gripper_joint_{}_vel'.format(i): i + 75 for i in range(5)
    },
    'left_gripper_open_vel': 75, # alias of left_gripper_joint_0_vel
    # [80, 83): left end effector positions
    'left_eef_pos_x': 80,
    'left_eef_pos_y': 81,
    'left_eef_pos_z': 82,
    # [83, 89): left end effector 6D pose
    'left_eef_angle_0': 83,
    'left_eef_angle_1': 84,
    'left_eef_angle_2': 85,
    'left_eef_angle_3': 86,
    'left_eef_angle_4': 87,
    'left_eef_angle_5': 88,
    # [89, 92): left end effector velocities
    'left_eef_vel_x': 89,
    'left_eef_vel_y': 90,
    'left_eef_vel_z': 91,
    # [92, 95): left end effector angular velocities
    'left_eef_angular_vel_roll': 92,
    'left_eef_angular_vel_pitch': 93,
    'left_eef_angular_vel_yaw': 94,
    # [95, 100): reserved
    # [100, 102): base linear velocities
    'base_vel_x': 100,
    'base_vel_y': 101,
    # [102, 103): base angular velocities
    'base_angular_vel': 102,
    # [103, 128): reserved
}
STATE_VEC_LEN = 128


"""
Below is a continuous 6D rotation representation adapted from
On the Continuity of Rotation Representations in Neural Networks
https://arxiv.org/pdf/1812.07035.pdf
https://github.com/papagina/RotationContinuity/blob/master/sanity_test/code/tools.py
"""
def rotation_matrix_to_ortho6d(matrix: np.ndarray) -> np.ndarray:
    """
    The orhto6d represents the first two column vectors a1 and a2 of the
    rotation matrix: [ | , |,  | ]
                     [ a1, a2, a3]
                     [ | , |,  | ]
    Input: (A1, ..., An, 3, 3)
    Output: (A1, ..., An, 6)
    """
    ortho6d = matrix[..., :, :2]
    # Transpose the last two dimension
    perm = list(range(len(ortho6d.shape)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    ortho6d = np.transpose(ortho6d, perm)
    # Flatten the last two dimension
    ortho6d = ortho6d.reshape(ortho6d.shape[:-2] + (-1,))
    return ortho6d


def normalize_vector(v: np.ndarray) -> np.ndarray:
    v_mag = np.linalg.norm(v, ord=2, axis=-1, keepdims=True)
    v_mag = np.maximum(v_mag, 1e-8)
    return v / v_mag


def ortho6d_to_rotation_matrix(ortho6d: np.ndarray) -> np.ndarray:
    """
    The orhto6d represents the first two column vectors a1 and a2 of the
    rotation matrix: [ | , |,  | ]
                     [ a1, a2, a3]
                     [ | , |,  | ]
    Input: (A1, ..., An, 6)
    Output: (A1, ..., An, 3, 3)
    """
    x_raw = ortho6d[..., 0:3]
    y_raw = ortho6d[..., 3:6]

    x = normalize_vector(x_raw)
    z = np.cross(x, y_raw)
    z = normalize_vector(z)
    y = np.cross(z, x)

    matrix = np.stack([x, y, z], axis=-1)
    return matrix


def state_vec_to_action(state_vec: np.ndarray) -> np.ndarray:
    batch_size = state_vec.shape[0]

    arm_format = "eef_vel_x,eef_vel_y,eef_vel_z,eef_angular_vel_roll,eef_angular_vel_pitch,eef_angular_vel_yaw,gripper_open"

    arm_concat = np.zeros((batch_size, len(arm_format.split(","))), dtype=np.float32)
    for i, key in enumerate(arm_format.split(",")):
        arm_concat[:, i] = state_vec[:, STATE_VEC_IDX_MAPPING[key]]

    eef_delta_pos = arm_concat[:, 0:3]
    eef_delta_ang = arm_concat[:, 3:6]
    gripper_open = (arm_concat[:, 6:7]) * 2 - 1

    gripper_open[gripper_open <= 0] = -1
    gripper_open[gripper_open > 0] = 1

    action = np.concatenate([eef_delta_pos, eef_delta_ang, gripper_open], axis=-1)
    return action


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
    container = av.open(filename, mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width = 128
    stream.height = 128
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "23"}

    for frame in frames:
        frame = np.array(frame)
        frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


class History:
    def __init__(self):
        self.agentview_image_list = []
        self.eye_in_hand_rgb_list = []
        self.qpos_list = []
        self.gripper_open_list = []
        self.eef_pos_list = []
        self.eef_ang_list = []

    def add(self, obs: dict[str, np.ndarray]):
        agentview_image = obs["agentview_image"]
        robot0_eye_in_hand_image = obs["robot0_eye_in_hand_image"]
        robot0_joint_pos = obs["robot0_joint_pos"]
        robot0_gripper_qpos = obs["robot0_gripper_qpos"]
        robot0_eef_pos = obs["robot0_eef_pos"]
        robot0_eef_quat = obs["robot0_eef_quat"]

        gripper_open = robot0_gripper_qpos[0] / 0.05184841017638091
        gripper_open = np.clip(gripper_open, 0, 1)

        eef_ang = R.from_quat(robot0_eef_quat).as_matrix()
        eef_ang = rotation_matrix_to_ortho6d(eef_ang)

        self.agentview_image_list.append(Image.fromarray(agentview_image))
        self.eye_in_hand_rgb_list.append(Image.fromarray(robot0_eye_in_hand_image))
        self.qpos_list.append(robot0_joint_pos.astype(np.float32))
        self.gripper_open_list.append(gripper_open)
        self.eef_pos_list.append(robot0_eef_pos.astype(np.float32))
        self.eef_ang_list.append(eef_ang.astype(np.float32))

    def get_model_inputs(self):
        images = [
            self.agentview_image_list[-2],
            self.eye_in_hand_rgb_list[-2],
            None,
            self.agentview_image_list[-1],
            self.eye_in_hand_rgb_list[-1],
            None,
        ]

        qpos = self.qpos_list[-1]
        gripper_open = self.gripper_open_list[-1]
        eef_pos = self.eef_pos_list[-1]
        eef_ang = self.eef_ang_list[-1]

        arm_concat = np.concatenate([qpos, [gripper_open], eef_pos, eef_ang])
        arm_format = "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,gripper_open,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5"

        state_vec = np.zeros(STATE_VEC_LEN, dtype=np.float32)
        state_mask = np.zeros(STATE_VEC_LEN, dtype=np.float32)

        for i, key in enumerate(arm_format.split(",")):
            state_vec[STATE_VEC_IDX_MAPPING[key]] = arm_concat[i]
            state_mask[STATE_VEC_IDX_MAPPING[key]] = 1

        return images, state_vec, state_mask

    def save_video(self, file_path):
        save_to_video(self.agentview_image_list, file_path)


def make_policy():
    with open("configs/base.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    # pretrained_model_name_or_path = "robotics-diffusion-transformer/rdt-1b"
    # pretrained_model_name_or_path = "/mnt/dongxu-fs2/data-hdd/mingyang/projs/RoboticsDiffusionTransformer/checkpoints/rdt-finetune-calvin-1b/checkpoint-84000"
    # pretrained_model_name_or_path = "/mnt/dongxu-fs2/data-hdd/mingyang/projs/RoboticsDiffusionTransformer/checkpoints/rdt-finetune-calvin-170m"
    # pretrained_model_name_or_path = "/mnt/dongxu-fs2/data-hdd/mingyang/projs/RoboticsDiffusionTransformer/checkpoints/rdt-finetune-calvin-170m-v2/checkpoint-72000"
    # pretrained_model_name_or_path = "/mnt/dongxu-fs2/data-hdd/mingyang/projs/RoboticsDiffusionTransformer/checkpoints/rdt-finetune-calvin-1b-v3/checkpoint-36000"
    pretrained_model_name_or_path = "robotics-diffusion-transformer/rdt-1b"
    pretrained_model_name_or_path = "/mnt/dongxu-fs2/data-hdd/mingyang/projs/RoboticsDiffusionTransformer/checkpoints/rdt-finetune-libero-1b-v1/checkpoint-23000"

    pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=config,
        dtype=torch.bfloat16,
        pretrained=pretrained_model_name_or_path,
        pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=20,
    )

    return model


def main():
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_10"
    task_suite = benchmark_dict[task_suite_name]()

    # retrieve a specific task
    task_id = 0
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_description = task.language

    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(task_id)
    init_state_id = 0
    obs = env.set_init_state(init_states[init_state_id])

    history = History()
    history.add(obs)
    history.add(obs)

    horizon = 32

    policy = make_policy()

    replacements = {
        '_': ' ',
        '1f': ' ',
        '4f': ' ',
        '-': ' ',
        '50': ' ',
        '55': ' ',
        '56': ' ',
    }

    for key, value in replacements.items():
        task_description = task_description.replace(key, value)
    task_description = task_description.strip()
    task_description = capitalize_and_period(task_description)

    sha1 = hashlib.sha1(task_description.encode()).hexdigest()

    cache_path = Path("outputs") / "cache" / f"text_embeds_{sha1}.pt"
    if cache_path.exists():
        text_embeds = torch.load(cache_path, map_location="cpu")
    else:
        text_embeds = policy.encode_instruction(task_description)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(text_embeds, cache_path)

    for step in range(360):
        if step % horizon == 0:
            images, state_vec, state_mask = history.get_model_inputs()
            with torch.inference_mode():
                future_states = policy.step(
                    state_vec=state_vec[None],
                    state_mask=state_mask[None],
                    images=images,
                    text_embeds=text_embeds,
                ).squeeze(0).cpu().numpy()

            actions = state_vec_to_action(future_states)

        obs, reward, done, info = env.step(actions[step % horizon])
        history.add(obs)

        if len(info) > 0:
            print(info)
            break

    history.save_video(f"outputs/rollout.mp4")

    env.close()


if __name__ == "__main__":
    main()

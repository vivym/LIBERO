from __future__ import annotations

import os

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

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

    dummy_action = [0.] * 7
    for step in range(2):
        history.get_model_inputs()
        obs, reward, done, info = env.step(dummy_action)
        history.add(obs)
    env.close()


if __name__ == "__main__":
    main()

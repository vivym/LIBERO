import os
from typing import List, Dict, Tuple
from functools import partial
import numpy as np
from PIL import Image
import litdata as ld
from tqdm import tqdm
import argparse

from debug.de_dataloader import print_batch


def read_data(fn: str, fns: list) -> Dict:
    ep_data: Dict = dict(np.load(fn, allow_pickle=True))
    '''
    ep_data: Dict,keys=dict_keys(['actions', 'rel_actions', 'robot_obs', 'scene_obs', 'rgb_static', 'rgb_gripper', 'rgb_tactile', 'depth_static', 'depth_gripper', 'depth_tactile'])
    actions,<class 'numpy.ndarray'>,shape=(7,)
    rel_actions,<class 'numpy.ndarray'>,shape=(7,)
    robot_obs,<class 'numpy.ndarray'>,shape=(15,)
    scene_obs,<class 'numpy.ndarray'>,shape=(24,)
    rgb_static,<class 'numpy.ndarray'>,shape=(200, 200, 3)
    rgb_gripper,<class 'numpy.ndarray'>,shape=(84, 84, 3)
    rgb_tactile,<class 'numpy.ndarray'>,shape=(160, 120, 6)
    depth_static,<class 'numpy.ndarray'>,shape=(200, 200)
    depth_gripper,<class 'numpy.ndarray'>,shape=(84, 84)
    depth_tactile,<class 'numpy.ndarray'>,shape=(160, 120, 2)
    '''
    gen_idx_offset = 3
    gen_fn: str = get_next_ep_name(fn, gen_idx_offset)
    for offset in list(range(gen_idx_offset + 1))[::-1]:
        if gen_fn in fns:
            if offset < gen_idx_offset:
                print(f'[read_data][Warning] {fn} uses offset={offset}')
            break
        gen_fn = get_next_ep_name(fn, offset)
    gen_ep_data: Dict = dict(np.load(gen_fn, allow_pickle=True))
    ep_data['gen_rgb_static'] = gen_ep_data['rgb_static']
    ep_data['gen_rgb_gripper'] = gen_ep_data['rgb_gripper']
    return ep_data


def get_next_ep_name(cur: str, offset: int = 3):
    ep_idx = int(os.path.basename(cur).split('_')[1].split('.')[0])
    next_ep_name = cur.replace(f'{ep_idx:07d}', f'{ep_idx+offset:07d}')
    return next_ep_name


if __name__ == "__main__":
    """
    python mdt/datasets/litdata_scripts/litdata_optimize.py -i /home/geyuan/datasets/CALVIN/dataset/task_D_D
    """
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--in_data", type=str,
                            help="path to original data")
    arg_parser.add_argument("-o", "--out_data", type=str,
                            default='/home/geyuan/code/mdt24rss_fork/dataset/litdata',
                            help="path to optimized output data")
    args = arg_parser.parse_args()

    ''' 1. Check data format '''
    args.in_data = os.path.abspath(args.in_data)
    args.out_data = os.path.abspath(args.out_data)
    root = os.path.dirname(args.in_data)
    task = os.path.basename(args.in_data)
    sub1_dir = os.listdir(args.in_data)
    for sub1 in sub1_dir:  # 'training', 'validation'
        if sub1 == 'training':
            continue
        ''' read file names '''
        print(f'[litdata_optimize] now_dir={sub1} Reading...')
        sub1_path = os.path.join(args.in_data, sub1)
        all_files = os.listdir(sub1_path)
        npz_files = [os.path.join(sub1_path, x) for x in all_files if x.endswith('.npz') and 'episode' in x]
        other_files = [x for x in all_files if 'episode' not in x]
        print(f'[litdata_optimize] now_dir={sub1}, following files or dirs are excluded: {other_files}')

        # for ep_name in tqdm(npz_files):
        #     next_ep_name = get_next_ep_name(ep_name)
        #     if next_ep_name not in npz_files:
        #         print(f'[litdata_optimize][Warning] Next NOT exist! ep_name={ep_name}, next_name={next_ep_name}')
        # print(f'[litdata_optimize] Check existence finished.')

        ''' create out_dir '''
        sub1_out = str(os.path.join(args.out_data, task, sub1))
        os.makedirs(sub1_out, exist_ok=True)

        ''' parse episode names (e.g episode_0360804.npz) '''
        print(f'[litdata_optimize] now_dir={sub1} Create dict ep_idx_to_dataset_order.npz in {sub1_out}')
        ep_indices = [int(os.path.basename(x).split('.')[0].split('_')[-1]) for x in npz_files]
        dict_ep_idx_to_dataset_order = {ep_indices[x]: x for x in range(len(ep_indices))}
        np.save(os.path.join(sub1_out, 'dict_ep_idx_to_dataset_order'),
                dict_ep_idx_to_dataset_order, allow_pickle=True)

        print(f'[litdata_optimize] now_dir={sub1} Optimizing... saved to {sub1_out}')
        ld.optimize(
            fn=partial(read_data, fns=npz_files),  # the function applied to each input
            inputs=npz_files,  # the inputs to the function (here it's a list of numbers)
            output_dir=sub1_out,  # optimized data is stored here
            num_workers=4,  # The number of workers on the same machine
            chunk_bytes="96MB",  # size of each chunk
            mode='overwrite',
            reorder_files=False,
            compression="zstd",
        )

import os
import time
import copy
import argparse
from typing import List
import numpy as np
import litdata as ld
from litdata.streaming.sampler import ChunkedIndex
from tqdm import tqdm

from debug.de_dataloader import print_batch

import torch
import torch.multiprocessing
from torch.utils.data import Dataset, DataLoader
torch.multiprocessing.set_sharing_strategy('file_system')


class DiskDataset(Dataset):
    def __init__(self, lit_root='/home/geyuan/code/mdt24rss_fork/dataset/litdata',
                 task='task_D_D',
                 split='training',
                 target_key='rel_actions'):
        ori_root = lit_root.replace('/litdata', '/')
        self.ori_path = os.path.join(ori_root, task, split)
        ep_npz_names = [int(x.split('_')[1].split('.')[0]) for x in
                        os.listdir(self.ori_path) if 'episode' in x]
        self.ep_npz_names = ep_npz_names
        self.target_key = target_key
    def __len__(self):
        return len(self.ep_npz_names)
    def __getitem__(self, idx):
        npz = np.load(os.path.join(self.ori_path,
                                      f"episode_{self.ep_npz_names[idx]:07d}.npz"), allow_pickle=True)
        s_data = copy.deepcopy(npz[self.target_key])
        del npz
        return {
            "data": s_data,
            "idx": idx,
            "ep_npz_idx": self.ep_npz_names[idx],
        }


class LitDataset(ld.StreamingDataset):
    def __init__(self, root='/home/geyuan/code/mdt24rss_fork/dataset/litdata',
                 task='task_D_D',
                 split='training'):
        super().__init__(
            input_dir=f'local:{os.path.join(root, task, split)}',
            max_cache_size="10GB",
            shuffle=False,
            drop_last=False
        )
        data_path = os.path.join(root, task, split)
        dict_ep_idx_to_dataset_order = np.load(
            os.path.join(data_path, "dict_ep_idx_to_dataset_order.npy"), allow_pickle=True).reshape(-1)[0]
        dict_dataset_order_to_ep_idx = {v: k for k, v in dict_ep_idx_to_dataset_order.items()}
        self.dict_ep_idx_to_dataset_order = dict_ep_idx_to_dataset_order
        self.dict_dataset_order_to_ep_idx = dict_dataset_order_to_ep_idx

    def __getitem__(self, idx: ChunkedIndex):
        data = super().__getitem__(idx)
        return {
            "data": data,
            "idx": idx.index,
            "ep_npz_idx": self.dict_dataset_order_to_ep_idx[idx.index],
        }


@torch.no_grad()
def litdata_extract(
        root='/home/geyuan/code/mdt24rss_fork/dataset/litdata',
        task='task_D_D',  # calvin_debug_dataset, task_D_D
        split='training',  # training, validation
        extract_key='rel_actions',
):
    """ Extract key data from Litdata dataset"""
    lit_dataset = LitDataset(root=root, task=task, split=split)
    dataset_len = len(lit_dataset)
    print(f'[litdata_extract] dataset_len={len(lit_dataset)}, dir={os.path.join(root, task, split)}')
    bs = 128
    train_dataloader = ld.StreamingDataLoader(
        lit_dataset,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    all_data = []
    start_time = time.time()
    for idx, data in enumerate(tqdm(train_dataloader)):
        data = data['data']
        for b_idx in range(bs):
            target: torch.Tensor = data[extract_key][b_idx]  # remove batch-dim
            target = target.cpu().numpy()
            all_data.append(target)
    print(f'[litdata_load] Litdata avg Speed (bs={bs}): {len(all_data) / (time.time() - start_time)}data/s, total={len(all_data)}')

    all_data = np.stack(all_data, axis=0)
    save_path = os.path.join(root, task, split, f"ep_{extract_key}.npy")
    np.save(save_path, all_data)
    print(f'[litdata_load] Data: {all_data.shape}, saved_to={save_path}')

    # Check
    ori_root = root.replace('/litdata', '/')
    ori_path = os.path.join(ori_root, task, split)
    ep_npz_names = [int(x.split('_')[1].split('.')[0]) for x in
                    os.listdir(ori_path) if 'episode' in x]
    check_source = [0, len(ep_npz_names) // 2, len(ep_npz_names) - 1] + np.random.randint(0, len(ep_npz_names), size=10).tolist()
    check_targets = [lit_dataset.dict_ep_idx_to_dataset_order[ep_npz_names[x]] for x in check_source]
    for i in range(len(check_source)):
        si, ti = check_source[i], check_targets[i]
        s_data = np.load(os.path.join(ori_path, f"episode_{ep_npz_names[si]:07d}.npz"), allow_pickle=True)[extract_key]
        t_data = np.load(save_path)[ti]
        if (s_data - t_data).sum() != 0.:
            print(f'[litdata_load] Check ERROR! saved.npy != ori/ep_*.npz, error={(s_data - t_data).sum()}')
    print(f'[litdata_load] Check OK on {check_source}!')


@torch.no_grad()
def diskdata_extract(
        root='/home/geyuan/code/mdt24rss_fork/dataset/litdata',
        task='task_D_D',  # calvin_debug_dataset, task_D_D
        split='training',  # training, validation
        extract_key='rel_actions',
):
    """ Extract key data from Litdata dataset"""
    os.makedirs(os.path.join(root, task, split), exist_ok=True)
    disk_dataset = DiskDataset(lit_root=root, task=task, split=split)
    dataset_len = len(disk_dataset)
    print(f'[diskdata_extract] dataset_len={len(disk_dataset)}, dir={os.path.join(root, task, split)}')
    ep_npz_names = [str(x) for x in disk_dataset.ep_npz_names]
    with open(os.path.join(root, task, split, 'ep_npz_names.list'), 'w') as f:
        f.write('\n'.join(ep_npz_names))
    print(f'[diskdata_extract] ep_npz_names saved to:{os.path.join(root, task, split, "ep_npz_names.list")}, len={len(ep_npz_names)}')

    bs = 128
    train_dataloader = DataLoader(
        disk_dataset,
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=24,
    )

    all_data = []
    start_time = time.time()
    for idx, data in enumerate(tqdm(train_dataloader)):
        data = data['data']
        targets = data
        for b_idx in range(targets.shape[0]):
            target: torch.Tensor = targets[b_idx]  # remove batch-dim
            target: np.ndarray = target.cpu().numpy()
            all_data.append(copy.deepcopy(target))
            del target
    print(f'[diskdata_extract] DiskData avg Speed (bs={bs}): {len(all_data) / (time.time() - start_time)}data/s, total={len(all_data)}')

    all_data = np.stack(all_data, axis=0)
    save_path = os.path.join(root, task, split, f"ep_{extract_key}.npy")
    np.save(save_path, all_data)
    print(f'[diskdata_extract] Data: {all_data.shape}, saved_to={save_path}')

    # Check
    ori_root = root.replace('/litdata', '/')
    ori_path = os.path.join(ori_root, task, split)
    ep_npz_names = [int(x.split('_')[1].split('.')[0]) for x in
                    os.listdir(ori_path) if 'episode' in x]
    check_source = [0, len(ep_npz_names) // 2, len(ep_npz_names) - 1] + np.random.randint(0, len(ep_npz_names), size=10).tolist()
    check_targets = [si for si in check_source]
    for i in range(len(check_source)):
        si, ti = check_source[i], check_targets[i]
        s_data = np.load(os.path.join(ori_path, f"episode_{ep_npz_names[si]:07d}.npz"), allow_pickle=True)[extract_key]
        t_data = np.load(save_path)[ti]
        if (s_data - t_data).sum() != 0.:
            print(f'[diskdata_extract] Check ERROR! saved.npy != ori/ep_*.npz, error={(s_data - t_data).sum()}')
    print(f'[diskdata_extract] Check OK on {check_source}!')


if __name__ == '__main__':
    '''
    task: calvin_debug_dataset, task_D_D, task_ABC_D
    split: training, validation
    '''
    # diskdata_extract(task='calvin_debug_dataset', split='validation')
    # diskdata_extract(task='calvin_debug_dataset', split='training')
    # diskdata_extract(task='task_D_D', split='validation')
    # diskdata_extract(task='task_D_D', split='training')
    # diskdata_extract(task='task_ABC_D', split='training')
    # diskdata_extract(task='task_ABC_D', split='validation')
    diskdata_extract(task='task_ABCD_D', split='training')
    diskdata_extract(task='task_ABCD_D', split='validation')

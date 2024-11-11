import logging
import os.path
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from itertools import chain
import random

import pickle
import numpy as np
from omegaconf import DictConfig
import pyhash
import torch
from torch.utils.data import Dataset
import litdata as ld
import litdata.streaming.sampler

from mdt.datasets.utils.episode_utils import (
    get_state_info_dict,
    process_actions,
    process_depth,
    process_language,
    process_rgb,
    process_state,
    lookup_naming_pattern,
)

hasher = pyhash.fnv1_32()
logger = logging.getLogger(__name__)


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


def get_validation_window_size(idx: int, min_window_size: int, max_window_size: int) -> int:
    """
    In validation step, use hash function instead of random sampling for consistent window sizes across epochs.

    Args:
        idx: Sequence index.
        min_window_size: Minimum window size.
        max_window_size: Maximum window size.

    Returns:
        Window size computed with hash function.
    """
    window_range = max_window_size - min_window_size + 1
    return min_window_size + hasher(str(idx)) % window_range


class LitStreamDataset(ld.StreamingDataset):
    """
    A CLEAR version of disk_dataset.ExtendedDiskDataset, removing the multi-inheriting relationship.

    Args:
        datasets_dir: Path of folder containing episode files (string must contain 'validation' or 'training').
        obs_space: DictConfig of observation space.
        proprio_state: DictConfig with shape of prioprioceptive state.
        key: 'vis' or 'lang'.
        lang_folder: Name of the subdirectory of the dataset containing the language annotations.
        num_workers: Number of dataloading workers for this dataset.
        transforms: Dict with pytorch data transforms.
        batch_size: Batch size.
        min_window_size: Minimum window length of loaded sequences.
        max_window_size: Maximum window length of loaded sequences.
        pad: If True, repeat last frame such that all sequences have length 'max_window_size'.
        aux_lang_loss_window: How many sliding windows to consider for auxiliary language losses, counted from the end
            of an annotated language episode.
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(self,
            ## BaseDataset ##
            datasets_dir: Path,
            obs_space: DictConfig,
            proprio_state: DictConfig,
            key: str,
            lang_folder: str,
            num_workers: int,
            transforms: Dict = {},
            batch_size: int = 32,
            min_window_size: int = 16,
            max_window_size: int = 32,
            pad: bool = True,
            aux_lang_loss_window: int = 1,
            window_sampling_strategy: str = 'random',
            geometric_p_value: float = 0.1,
            ## DiskDataset ##
            skip_frames: int = 1,
            save_format: str = "npz",
            pretrain: bool = False,
            ## ExtendedDiskDataset ##
            obs_seq_len: int = -1,  # used:1
            action_seq_len: int = -1,  # used:10
            future_range: int = -1,  # used:29
            img_gen_frame_diff: int = 3,  # used:3
            ## ld.StreamingDataset ##
            ld_data_root: str = "",
            ld_item_loader = None,
            ld_shuffle: bool = True,
            ld_drop_last: bool = False,
            ld_seed: int = 42,
            ld_max_cache_size: Union[int, str] = "10GB",
            ld_subsample: float = 1.0,
    ):
        train_val_spilt = os.path.basename(os.path.abspath(datasets_dir))  # 'training', 'validation'
        task_name = os.path.basename(os.path.dirname(os.path.abspath(datasets_dir)))  # 'task_XXX_D'
        self.lit_input_dir = os.path.join(ld_data_root, task_name, train_val_spilt)
        print('[DEBUG] lit_input_dir:', self.lit_input_dir)
        super(LitStreamDataset, self).__init__(
            input_dir=self.lit_input_dir,
            item_loader=ld_item_loader,
            shuffle=ld_shuffle, drop_last=ld_drop_last, seed=ld_seed,
            max_cache_size=ld_max_cache_size, subsample=ld_subsample,
        )
        self.observation_space = obs_space
        self.proprio_state = proprio_state
        self.transforms = transforms
        self.with_lang = key == "lang"
        self.relative_actions = "rel_actions" in self.observation_space["actions"]
        assert window_sampling_strategy in ('random', 'geometric')
        self.window_sampling_strategy = window_sampling_strategy
        self.geometric_p_value = geometric_p_value  # only needed for geomtric sampling
        self.pad = pad
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_window_size = min_window_size  # default:21
        self.max_window_size = max_window_size  # default:50
        self.abs_datasets_dir = datasets_dir
        self.lang_folder = lang_folder  # if self.with_lang else None
        self.aux_lang_loss_window = aux_lang_loss_window
        assert "validation" in self.abs_datasets_dir.as_posix() or "training" in self.abs_datasets_dir.as_posix()
        self.validation = "validation" in self.abs_datasets_dir.as_posix()
        assert self.abs_datasets_dir.is_dir()
        logger.info(f"loading dataset at {self.abs_datasets_dir}")
        logger.info("finished loading dataset")

        ## DiskDataset ##
        self.save_format = save_format
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain
        self.skip_frames = skip_frames

        # episode_lookup: dataset_order -> ep_fn_idx, no repeat in ep_fn_idx
        if self.with_lang:
            self.episode_lookup, self.lang_lookup, self.lang_ann, self.lang_text = self._build_file_indices_lang(
                self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(self.abs_datasets_dir, self.save_format)

        ## ExtendedDiskDataset ##
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len
        self.future_range = future_range  # Number of steps into the future to sample goals
        self.ep_start_end_ids = np.load(self.abs_datasets_dir / "ep_start_end_ids.npy")  # Load sequence boundaries
        self.img_gen_frame_diff = img_gen_frame_diff
        self.random_frame_diff = False if img_gen_frame_diff > -1 else True

        ## LitData Related ##
        dict_ep_idx_to_dataset_order = np.load(
            os.path.join(self.lit_input_dir, "dict_ep_idx_to_dataset_order.npy"), allow_pickle=True
        ).reshape(-1)[0]
        self.dict_ep_idx_to_dataset_order = dict_ep_idx_to_dataset_order
        print('[DEBUG][LitStreamDataset] loaded: dict_ep_idx_to_dataset_order.npy')

    def __getitem__(self, chunk_idx: ld.streaming.sampler.ChunkedIndex) -> Dict:
        """
        Get sequence of dataset.

        Args:
            idx: Index of the sequence.

        Returns:
            Loaded sequence.
        """
        # print('[DEBUG] get_item:', type(idx), idx)
        idx = chunk_idx.index
        if isinstance(idx, int):
            # When max_ws_size and min_ws_size are equal, avoid unnecessary padding
            # acts like Constant dataset. Currently, used for language data
            if self.min_window_size == self.max_window_size:
                window_size = self.max_window_size
            elif self.min_window_size < self.max_window_size:
                window_size = self._get_window_size(idx)
            else:
                logger.error(f"min_window_size {self.min_window_size} > max_window_size {self.max_window_size}")
                raise ValueError
        else:
            idx, window_size = idx
        sequence = self._get_sequences(idx, window_size)
        if self.pad:  # used:False
            pad_size = self._get_pad_size(sequence)
            sequence = self._pad_sequence(sequence, pad_size)
        return sequence

    def _get_sequences(self, idx: int, window_size: int) -> Dict:
        """
        Load sequence of length window_size.

        Args:
            idx: Index of starting frame.
            window_size: Length of sampled episode.

        Returns:
            dict: Dictionary of tensors of loaded sequence with different input modalities and actions.
        """

        episode = self._load_episode(idx, window_size)

        seq_state_obs = process_state(episode, self.observation_space, self.transforms, self.proprio_state)
        seq_rgb_obs = process_rgb(episode, self.observation_space, self.transforms)
        seq_depth_obs = process_depth(episode, self.observation_space, self.transforms)
        seq_acts = process_actions(episode, self.observation_space, self.transforms)
        info = get_state_info_dict(episode)
        seq_lang = process_language(episode, self.transforms, self.with_lang)
        info = self._add_language_info(info, idx)
        seq_dict = {**seq_state_obs, **seq_rgb_obs, **seq_depth_obs, **seq_acts, **info, **seq_lang}  # type:ignore
        seq_dict["idx"] = idx  # type:ignore
        seq_dict['future_frame_diff'] = episode['future_frame_diff']
        return seq_dict

    def _get_window_size(self, idx: int) -> int:
        """
        Sample a window size taking into account the episode limits.

        Args:
            idx: Index of the sequence to load.

        Returns:
            Window size.
        """
        window_diff = self.max_window_size - self.min_window_size
        if len(self.episode_lookup) <= idx + window_diff:
            # last episode when required right >= len(lookup)
            max_window = self.min_window_size + len(self.episode_lookup) - idx - 1
        elif self.episode_lookup[idx + window_diff] != self.episode_lookup[idx] + window_diff:
            # less than max_episode steps until next episode
            steps_to_next_episode = int(
                np.nonzero(
                    self.episode_lookup[idx: idx + window_diff + 1]
                    - (self.episode_lookup[idx] + np.arange(window_diff + 1))
                )[0][0]
            )
            max_window = min(self.max_window_size, (self.min_window_size + steps_to_next_episode - 1))
        else:
            max_window = self.max_window_size

        if self.validation:
            # in validation step, repeat the window sizes for each epoch.
            return get_validation_window_size(idx, self.min_window_size, max_window)
        else:
            if self.window_sampling_strategy == 'geometric':  # used
                p = self.geometric_p_value  # Choose a suitable value for p
                while True:
                    sampled_window_size = 1 + np.random.geometric(p)  # E(G(p))=10
                    if self.min_window_size <= sampled_window_size <= max_window:  # if in [21,50], avg~=28.98
                        return sampled_window_size
            else:  # 'random'
                return np.random.randint(self.min_window_size, max_window + 1)

    def __len__(self) -> int:
        """
        Returns:
            Size of the dataset.
        """
        return len(self.episode_lookup)

    def _get_pad_size(self, sequence: Dict) -> int:
        """
        Determine how many frames to append to end of the sequence

        Args:
            sequence: Loaded sequence.

        Returns:
            Number of frames to pad.
        """
        return self.max_window_size - len(sequence["actions"])

    def _pad_sequence(self, seq: Dict, pad_size: int) -> Dict:
        """
        Pad a sequence by repeating the last frame.

        Args:
            seq: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded sequence.
        """
        seq.update({"robot_obs": self._pad_with_repetition(seq["robot_obs"], pad_size)})
        seq.update({"rgb_obs": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["rgb_obs"].items()}})
        seq.update({"depth_obs": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["depth_obs"].items()}})
        #  todo: find better way of distinguishing rk and play action spaces
        if not self.relative_actions:
            # repeat action for world coordinates action space
            seq.update({"actions": self._pad_with_repetition(seq["actions"], pad_size)})
        else:
            # for relative actions zero pad all but the last action dims and repeat last action dim (gripper action)
            seq_acts = torch.cat(
                [
                    self._pad_with_zeros(seq["actions"][..., :-1], pad_size),
                    self._pad_with_repetition(seq["actions"][..., -1:], pad_size),
                ],
                dim=-1,
            )
            seq.update({"actions": seq_acts})
        seq.update({"state_info": {k: self._pad_with_repetition(v, pad_size) for k, v in seq["state_info"].items()}})
        return seq

    @staticmethod
    def _pad_with_repetition(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a sequence Tensor by repeating last element pad_size times.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        last_repeated = torch.repeat_interleave(torch.unsqueeze(input_tensor[-1], dim=0), repeats=pad_size, dim=0)
        padded = torch.vstack((input_tensor, last_repeated))
        return padded

    @staticmethod
    def _pad_with_zeros(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
        """
        Pad a Tensor with zeros.

        Args:
            input_tensor: Sequence to pad.
            pad_size: Number of frames to pad.

        Returns:
            Padded Tensor.
        """
        zeros_repeated = torch.repeat_interleave(
            torch.unsqueeze(torch.zeros(input_tensor.shape[-1]), dim=0), repeats=pad_size, dim=0
        )
        padded = torch.vstack((input_tensor, zeros_repeated))
        return padded

    def _add_language_info(self, info: Dict, idx: int) -> Dict:
        """
        If dataset contains language, add info to determine if this sequence will be used for the auxiliary losses.

        Args:
            info: Info dictionary.
            idx: Sequence index.

        Returns:
            Info dictionary with updated information.
        """
        if not self.with_lang:
            return info
        use_for_aux_lang_loss = (
                idx + self.aux_lang_loss_window >= len(self.lang_lookup)
                or self.lang_lookup[idx] < self.lang_lookup[idx + self.aux_lang_loss_window]
        )
        info["use_for_aux_lang_loss"] = use_for_aux_lang_loss
        return info

    #################### End ####################

    ## Copied from disk_dataset.DiskDataset ##
    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.

        Args:
            file_idx: index of starting frame.

        Returns:
            Path to file.
        """
        return Path(f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def _build_file_indices_lang(self, abs_datasets_dir: Path) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        try:
            print("trying to load lang data from: ", abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        except Exception:
            print("Exception, trying to load lang data from: ", abs_datasets_dir / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of (second - first) are <= 64, vis:[5124],lang:[1011]
        print('[DEBUG] LANG ep_start_end_ids:', len(ep_start_end_ids), ep_start_end_ids[0], ep_start_end_ids[1],
              ep_start_end_ids[2])
        lang_ann = lang_data["language"]["emb"]  # length total number of annotations
        lang_text = lang_data["language"]["ann"]  # length total number of annotations
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.pretrain:
                start_idx = max(start_idx, end_idx + 1 - self.min_window_size - self.aux_lang_loss_window)
            assert end_idx >= self.max_window_size
            cnt = 0
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        print('[DEBUG] LANG episode_lookup len=', len(episode_lookup))
        return np.array(episode_lookup), lang_lookup, lang_ann, lang_text

    #################### End ####################

    ## Copied from disk_dataset.ExtendedDiskDataset ##
    def find_sequence_boundaries(self, idx: int) -> Tuple[int, int]:
        for start_idx, end_idx in self.ep_start_end_ids:
            if start_idx <= idx < end_idx:
                return start_idx, end_idx
        raise ValueError(f"Index {idx} does not belong to any sequence.")

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.

        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]
        end_idx = start_idx + self.action_seq_len + self.obs_seq_len - 1
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")

        ''' op1. vanilla load from disk '''
        # print('[DEBUG] ep_start_end:', start_idx, '-', end_idx, f'={end_idx-start_idx}')
        # print('[DEBUG] _get_episode_name:', [self._get_episode_name(file_idx) for file_idx in range(start_idx, end_idx)])
        # episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        ''' op2. litdata loading '''
        file_indices = [file_idx for file_idx in range(start_idx, end_idx)]
        episodes = [super(LitStreamDataset, self).__getitem__(self.dict_ep_idx_to_dataset_order[x]) for x in file_indices]

        # Modify the episode dict to only include the specified sequence lengths
        if self.random_frame_diff:
            img_gen_frame_diff = random.randint(0, self.action_seq_len - 1)
        else:
            img_gen_frame_diff = self.img_gen_frame_diff

        episode = {}
        for key in keys:
            if 'gen' in key:
                continue

            stacked_data = np.stack([ep[key] for ep in episodes])
            if key == "rel_actions" or key == 'actions':
                episode[key] = stacked_data[(self.obs_seq_len - 1):((self.obs_seq_len - 1) + self.action_seq_len), :]
            else:
                if key == 'rgb_static':
                    gen_img_static = stacked_data[self.obs_seq_len + img_gen_frame_diff - 1, :]
                elif key == 'rgb_gripper':
                    gen_img_gripper = stacked_data[self.obs_seq_len + img_gen_frame_diff - 1, :]
                episode[key] = stacked_data[:self.obs_seq_len, :]

        if self.with_lang:
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]  # TODO check  [0]
            episode["language_text"] = self.lang_text[self.lang_lookup[idx]]  # [0]  # TODO check  [0]

        # get the random future state as goal
        goal_idx = end_idx + window_size
        # print(start_idx, end_idx, goal_idx)
        eps_start_idx, eps_end_idx = self.find_sequence_boundaries(end_idx)

        # Check if future goal can be sampled

        if eps_end_idx < goal_idx:
            goal_idx = eps_end_idx

        goal_episodes = self.load_file(self._get_episode_name(goal_idx))
        goal_episode = {}
        for key in keys:
            if 'gen' in key:
                continue
            goal_stacked_data = np.stack([goal_episodes[key]])
            if key == "rel_actions" or key == 'actions':
                pass
            else:
                goal_episode[key] = goal_stacked_data[:self.obs_seq_len, :]
        # store for merging

        episode = self.merge_episodes(episode, goal_episode)
        episode['gen_static'] = gen_img_static
        episode['gen_gripper'] = gen_img_gripper
        episode['future_frame_diff'] = np.array(img_gen_frame_diff)
        '''
        VIS: Dict,keys=dict_keys(['robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'lang', 'idx', 'future_frame_diff'])
        robot_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 8])
        rgb_obs:Dict,keys=dict_keys(['rgb_static', 'rgb_gripper', 'gen_static', 'gen_gripper'])
        -rgb_static,<class 'torch.Tensor'>,shape=torch.Size([2, 3, 224, 224])
        -rgb_gripper,<class 'torch.Tensor'>,shape=torch.Size([2, 3, 84, 84])
        -gen_static,<class 'torch.Tensor'>,shape=torch.Size([1, 3, 112, 112])
        -gen_gripper,<class 'torch.Tensor'>,shape=torch.Size([1, 3, 112, 112])
        depth_obs:Dict,keys=dict_keys([])
        actions,<class 'torch.Tensor'>,shape=torch.Size([10, 7])
        state_info:Dict,keys=dict_keys(['robot_obs', 'scene_obs'])
        -robot_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 15])
        -scene_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 24])
        lang,<class 'torch.Tensor'>,shape=torch.Size([0])
        idx:<class 'int'>,0
        future_frame_diff,<class 'numpy.ndarray'>,shape=(), 3

        Lang batch: Dict,keys=dict_keys(['robot_obs', 'rgb_obs', 'depth_obs', 'actions', 'state_info', 'use_for_aux_lang_loss', 'lang', 'lang_text', 'idx', 'future_frame_diff'])
        robot_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 8])
        rgb_obs: Dict,keys=dict_keys(['rgb_static', 'rgb_gripper', 'gen_static', 'gen_gripper'])
        -rgb_static,<class 'torch.Tensor'>,shape=torch.Size([2, 3, 224, 224])
        -rgb_gripper,<class 'torch.Tensor'>,shape=torch.Size([2, 3, 84, 84])
        -gen_static,<class 'torch.Tensor'>,shape=torch.Size([1, 3, 112, 112])
        -gen_gripper,<class 'torch.Tensor'>,shape=torch.Size([1, 3, 112, 112])
        depth_obs: Dict,keys=dict_keys([])
        actions,<class 'torch.Tensor'>,shape=torch.Size([10, 7])
        state_info: Dict,keys=dict_keys(['robot_obs', 'scene_obs'])
        -robot_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 15])
        -scene_obs,<class 'torch.Tensor'>,shape=torch.Size([2, 24])
        use_for_aux_lang_loss:<class 'bool'>,False
        lang,<class 'torch.Tensor'>,shape=torch.Size([1024])
        lang_text:<class 'str'>,len=30
        idx:<class 'int'>,0
        future_frame_diff,<class 'numpy.ndarray'>,shape=(), 3
        '''
        return episode

    def merge_episodes(self, episode1: Dict[str, np.ndarray], episode2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        merged_episode = {}
        all_keys = set(episode1.keys()).union(set(episode2.keys()))
        for key in all_keys:
            if key in episode1 and key in episode2:
                # Merge logic here, for example:
                merged_episode[key] = np.concatenate([episode1[key], episode2[key]], axis=0)
            elif key in episode1:
                merged_episode[key] = episode1[key]
            else:
                merged_episode[key] = episode2[key]
        return merged_episode

    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        print('[DEBUG] VIS ep_start_end_ids:', ep_start_end_ids.shape)
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        print('[DEBUG] VIS episode_lookup len=', len(episode_lookup))
        return np.array(episode_lookup)

    #################### End ####################

import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader, Dataset
import torchvision

import mdt
from mdt.datasets.utils.episode_utils import load_dataset_statistics
from mdt.datasets.utils.shared_memory_utils import load_shm_lookup, save_shm_lookup, SharedMemoryLoader

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})
ONE_EP_DATASET_URL = "http://www.informatik.uni-freiburg.de/~meeso/50steps.tar.xz"


class MergedDataset(Dataset):
    def __init__(self, s_dataset, t_dataset):
        super(MergedDataset, self).__init__()
        self.s_dataset = s_dataset
        self.t_dataset = t_dataset
        self.batch_size = s_dataset.batch_size
        self.num_workers = s_dataset.num_workers
    def __getitem__(self, index):
        return {
            "source": self.s_dataset[index],
            "target": self.t_dataset[index],
        }
    def __len__(self):
        return max(self.s_dataset.__len__(), self.t_dataset.__len__())


class HulcDomainAdaptDataModule(pl.LightningDataModule):
    """
    Config: conf/datamodule/calvin_da.yaml
    """
    def __init__(
        self,
        datasets: DictConfig,  # source and target use the same vision-language config
        source_root_data_dir: str = "XXX/task_ABC_D/",
        target_root_data_dir: str = "XXX/task_D_D/",
        target_train_ratio: float = 1.,
        num_workers: int = 8,
        transforms: DictConfig = DEFAULT_TRANSFORM,
        shuffle_val: bool = False,
        **kwargs: Dict,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        self.train_sampler = None
        self.val_sampler = None
        self.num_workers = num_workers
        source_root_data_path = Path(source_root_data_dir)
        target_root_data_path = Path(target_root_data_dir)
        if not source_root_data_path.is_absolute():
            source_root_data_path = Path(mdt.__file__).parent / source_root_data_path
        if not target_root_data_path.is_absolute():
            target_root_data_path = Path(mdt.__file__).parent / target_root_data_path
        self.s_training_dir = source_root_data_path / "training"
        self.s_val_dir = source_root_data_path / "validation"
        self.t_training_dir = target_root_data_path / "training"
        self.t_val_dir = target_root_data_path / "validation"
        self.training_dirs = [self.s_training_dir, self.t_training_dir]
        self.val_dirs = [self.s_val_dir, self.t_val_dir]

        self.target_train_ratio = float(target_train_ratio)

        self.shuffle_val = shuffle_val
        self.modalities: List[str] = []
        self.transforms = transforms

        if 'lang_dataset' in self.datasets_cfg:
            if "shm_dataset" in self.datasets_cfg.lang_dataset._target_:
                self.use_shm = "shm_dataset" in self.datasets_cfg.lang_dataset._target_
            else:
                self.use_shm = False
        elif 'shm_dataset' in self.datasets_cfg.vision_dataset._target_:
            self.use_shm = True
        else:
            self.use_shm = False

    def prepare_data(self, *args, **kwargs):
        for training_dir, val_dir in zip(self.training_dirs, self.val_dirs):  # 0-it:source, 1-it:target
            # check if files already exist
            dataset_exist = np.any([len(list(training_dir.glob(extension))) for extension in ["*.npz", "*.pkl"]])

            # download and unpack images
            if not dataset_exist:
                if "CI" not in os.environ:
                    print(f"No dataset found in {training_dir}.")
                    print("For information how to download to full CALVIN dataset, please visit")
                    print("https://github.com/mees/calvin/tree/main/dataset")
                    print("Do you wish to download small debug dataset to continue training?")
                    s = input("YES / no")
                    if s == "no":
                        exit()
                logger.info(f"downloading dataset to {training_dir} and {val_dir}")
                torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, training_dir)
                torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, val_dir)

            if self.use_shm:
                # When using shared memory dataset, initialize lookups
                train_shmem_loader = SharedMemoryLoader(self.datasets_cfg, training_dir)
                train_shm_lookup = train_shmem_loader.load_data_in_shared_memory()

                val_shmem_loader = SharedMemoryLoader(self.datasets_cfg, val_dir)
                val_shm_lookup = val_shmem_loader.load_data_in_shared_memory()

                save_shm_lookup(train_shm_lookup, val_shm_lookup)

        print('[DEBUG] HulcDomainAdaptDataModule prepare_data finished.')

    def setup(self, stage=None):
        """
        Called by trainer.fit()
        """
        # transforms = load_dataset_statistics(self.training_dir, self.val_dir, self.transforms)
        s_transforms = load_dataset_statistics(self.s_training_dir, self.s_val_dir, self.transforms)
        t_transforms = load_dataset_statistics(self.t_training_dir, self.t_val_dir, self.transforms)

        self.s_train_transforms = {}
        self.t_train_transforms = {}
        for cam in s_transforms.train:
            # print("Processing camera:", cam)
            cam_transforms = []
            for transform in s_transforms.train[cam]:
                # print("Instantiating transform for camera", cam, ":", transform)
                if transform._target_ == "torchvision.transforms.ColorJitter":
                    instantiated_transform = torchvision.transforms.ColorJitter(
                        brightness=transform.brightness,
                        contrast=tuple(transform.contrast),
                        saturation=tuple(transform.saturation),
                    )
                else:
                    instantiated_transform = hydra.utils.instantiate(transform)
                cam_transforms.append(instantiated_transform)
            self.s_train_transforms[cam] = cam_transforms
        for cam in t_transforms.train:
            # print("Processing camera:", cam)
            cam_transforms = []
            for transform in t_transforms.train[cam]:
                # print("Instantiating transform for camera", cam, ":", transform)
                if transform._target_ == "torchvision.transforms.ColorJitter":
                    instantiated_transform = torchvision.transforms.ColorJitter(
                        brightness=transform.brightness,
                        contrast=tuple(transform.contrast),
                        saturation=tuple(transform.saturation),
                    )
                else:
                    instantiated_transform = hydra.utils.instantiate(transform)
                cam_transforms.append(instantiated_transform)
            self.t_train_transforms[cam] = cam_transforms

        self.s_val_transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in s_transforms.val[cam]] for cam in s_transforms.val
        }
        self.s_train_transforms = {
            key: torchvision.transforms.Compose(val) for key, val in self.s_train_transforms.items()
        }
        self.s_val_transforms = {
            key: torchvision.transforms.Compose(val) for key, val in self.s_val_transforms.items()
        }
        self.t_val_transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in t_transforms.val[cam]] for cam in t_transforms.val
        }
        self.t_train_transforms = {
            key: torchvision.transforms.Compose(val) for key, val in self.t_train_transforms.items()
        }
        self.t_val_transforms = {
            key: torchvision.transforms.Compose(val) for key, val in self.t_val_transforms.items()
        }

        self.train_datasets, self.train_sampler, self.val_datasets, self.val_sampler = {}, {}, {}, {}

        if self.use_shm:
            train_shm_lookup, val_shm_lookup = load_shm_lookup()  # maybe not support source target

        for _, dataset in self.datasets_cfg.items():  # keys:'lang_dataset','vision_dataset', not used, only for loop
            if dataset == 'lang_paraphrase-MiniLM-L3-v2':
                continue
            else:
                t_train_dataset = hydra.utils.instantiate(
                    dataset, datasets_dir=self.t_training_dir, transforms=self.t_train_transforms,
                    chose_ratio=self.target_train_ratio,
                )
                t_val_dataset = hydra.utils.instantiate(
                    dataset, datasets_dir=self.t_val_dir, transforms=self.t_val_transforms
                )
                s_train_dataset = hydra.utils.instantiate(
                    dataset, datasets_dir=self.s_training_dir, transforms=self.s_train_transforms,
                    # max_len=len(t_train_dataset)
                )
                s_val_dataset = hydra.utils.instantiate(
                    dataset, datasets_dir=self.s_val_dir, transforms=self.s_val_transforms
                )
                if self.use_shm:
                    s_train_dataset.setup_shm_lookup(train_shm_lookup)
                    s_val_dataset.setup_shm_lookup(val_shm_lookup)
                    t_train_dataset.setup_shm_lookup(train_shm_lookup)
                    t_val_dataset.setup_shm_lookup(val_shm_lookup)
                    print('[Warning] HulcDomainAdaptDataModule shm dataset may have bug!')
                key = dataset.key  # "lang", "vis"
                self.train_datasets[f"{key}_source"] = s_train_dataset
                self.train_datasets[f"{key}_target"] = t_train_dataset
                self.val_datasets[f"{key}_source"] = s_val_dataset
                self.val_datasets[f"{key}_target"] = t_val_dataset  # to avoid 1 thread accessing 2 different dirs
                self.modalities.append(key)  # "lang", "vis"
                logger.info(f'HulcDomainAdaptDataModule: train_{key}_s_len={len(s_train_dataset)}, '
                            f'train_{key}_t_len={len(t_train_dataset)}, chose_ratio={self.target_train_ratio * 100:.3f}%')
        print(f'[DEBUG] HulcDomainAdaptDataModule setup finished. '
              f'train.keys:{self.train_datasets.keys()}, '
              f'val.keys:{self.val_datasets.keys()}.')

    def train_dataloader(self):
        return CombinedLoader({
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
                shuffle=True,
                drop_last=False,
            )
            for key, dataset in self.train_datasets.items()
        }, "max_size_cycle")

    def test_dataloader(self):  # just for debug
        return CombinedLoader({
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
                shuffle=False,
            )
            for key, dataset in self.train_datasets.items()
        }, "max_size_cycle")

    def val_dataloader(self):
        val_dataloaders = {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
            )
            for key, dataset in self.val_datasets.items()
        }
        # combined_val_loaders = val_dataloaders['vis']
        combined_val_loaders = CombinedLoader(val_dataloaders, "max_size_cycle")
        return combined_val_loaders

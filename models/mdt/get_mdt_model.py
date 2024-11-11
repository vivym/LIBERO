from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Union, List, Dict, Tuple
import importlib

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import hydra
import omegaconf
from omegaconf import OmegaConf
import numpy as np
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm
import wandb
import torch.distributed as dist


logger = logging.getLogger(__name__)


def get_all_checkpoints(experiment_folder: Path) -> List:
    if experiment_folder.is_dir():
        print('[DEBUG] experiment_folder.is_dir()')
        checkpoint_folder = experiment_folder / "saved_models"
        if not os.path.exists(checkpoint_folder):
            skip_folders = ["wandb", ".hydra"]
            for level1_folder in os.listdir(experiment_folder):
                if level1_folder not in skip_folders:
                    checkpoint_folder = experiment_folder / level1_folder
                    if checkpoint_folder.is_dir():
                        break

        print("[DEBUG] checkpoint_folder:", checkpoint_folder)

        # if checkpoint_folder.is_dir():
        #     print('[DEBUG] checkpoint_folder.is_dir()')
        #     checkpoints = sorted(Path(checkpoint_folder).iterdir(), key=lambda chk: chk.stat().st_mtime)
        #     print('[DEBUG] checkpoints.len:', len(checkpoints))
        #     if len(checkpoints):
        #         print('[DEBUG] checkpoints:', checkpoints)
        #         return [chk for chk in checkpoints if chk.suffix == ".ckpt"]

        if checkpoint_folder.is_dir():
            level2_folder = os.listdir(checkpoint_folder)[0]
            level3_folder = checkpoint_folder / level2_folder / "checkpoints"
            checkpoints = sorted(Path(level3_folder).iterdir())
            if len(checkpoints):
                return [chk for chk in checkpoints if chk.suffix == ".ckpt"]
    return []


def get_last_checkpoint(experiment_folder: Path) -> Union[Path, None]:
    # return newest checkpoint according to creation time
    print('[DEBUG] experiment_folder:', experiment_folder)
    checkpoints = get_all_checkpoints(experiment_folder)
    print('[DEBUG] checkpoints:', checkpoints)
    if len(checkpoints):
        return checkpoints[-1]
    return None


def get_checkpoint_i_from_dir(dir, i: int = -1):
    ckpt_paths = list(dir.rglob("*.ckpt"))
    if i == -1:
        for ckpt_path in ckpt_paths:
            if ckpt_path.stem == "last":
                return ckpt_path

    # Search for ckpt of epoch i
    for ckpt_path in ckpt_paths:
        split_path = str(ckpt_path).split("_")
        for k, word in enumerate(split_path):
            if word == "epoch":
                if int(split_path[k + 1]) == i:
                    return ckpt_path

    sorted(ckpt_paths, key=lambda f: f.stat().st_mtime)
    return ckpt_paths[i]


def get_config_from_dir(dir):
    dir = Path(dir)
    config_yaml = list(dir.rglob("*.hydra/config.yaml"))[0]
    return OmegaConf.load(config_yaml)


def load_class(name):
    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def load_pl_module_from_checkpoint(
        filepath: Union[Path, str],
        epoch: int = 1,
        overwrite_cfg: dict = {},
        use_ema_weights: bool = False
):
    if isinstance(filepath, str):
        filepath = Path(filepath)

    if filepath.is_dir():
        filedir = filepath
        ckpt_path = get_checkpoint_i_from_dir(dir=filedir, i=epoch)
    elif filepath.is_file():
        assert filepath.suffix == ".ckpt", "File must have .ckpt extension"
        ckpt_path = filepath
        filedir = filepath.parents[0]
    else:
        raise ValueError(f"not valid file path: {str(filepath)}")
    config = get_config_from_dir(filedir)
    class_name = config.model.pop("_target_")
    if "_recursive_" in config.model:
        del config.model["_recursive_"]
    print(f"class_name {class_name}")
    module_class = load_class(class_name)
    print(f"Loading model from {ckpt_path}")
    load_cfg = {**config.model, **overwrite_cfg}
    model = module_class.load_from_checkpoint(ckpt_path, **load_cfg)
    # Load EMA weights if they exist and the flag is set
    if use_ema_weights:
        checkpoint_data = torch.load(ckpt_path)
        if "ema_weights" in checkpoint_data['callbacks']['EMA']:
            ema_weights_list = checkpoint_data['callbacks']['EMA']['ema_weights']

            # Convert list of tensors to a state_dict format
            ema_weights_dict = {name: ema_weights_list[i] for i, (name, _) in enumerate(model.named_parameters())}

            model.load_state_dict(ema_weights_dict)
            print("Successfully loaded EMA weights from checkpoint!")
        else:
            print("Warning: No EMA weights found in checkpoint!")

    print(f"Finished loading model {ckpt_path}")
    return model


def add_parent_for_cfg(ori_cfg: omegaconf.DictConfig,
                       target_key: str = "_target_", target_value: str = "mdt.",
                       prefix_value: str = "models."):
    for key, value in ori_cfg.items():
        if key in ("logger", ):
            continue
        if OmegaConf.is_dict(value):
            add_parent_for_cfg(value, target_key, target_value, prefix_value)  # recursive
        else:
            if key == target_key and isinstance(value, str) and value.startswith(target_value):
                ori_cfg.update({key: f"{prefix_value}{value}"})
    return ori_cfg


def get_model_from_hydra_configs(train_folder, checkpoint, device_id=0,
                             eval_cfg_overwrite={}):
    train_cfg_path = Path(train_folder) / ".hydra/config.yaml"

    def_cfg = OmegaConf.load(train_cfg_path)
    eval_override_cfg = OmegaConf.create(eval_cfg_overwrite)
    cfg = OmegaConf.merge(def_cfg, eval_override_cfg)

    # print(cfg)

    data_module = hydra.utils.instantiate(cfg.datamodule, num_workers=0)
    data_module.prepare_data()
    data_module.setup()

    if device_id != 'cpu':
        device = torch.device(f"cuda:{device_id}")
    else:
        device = 'cpu'

    print(f"Loading model from {checkpoint}")

    # new stuff
    epoch = cfg.epoch_to_load if "epoch_to_load" in cfg else -1
    overwrite_cfg = cfg.overwrite_module_cfg if "overwrite_module_cfg" in cfg else {}
    module_path = str(Path(train_folder).expanduser())
    model = load_pl_module_from_checkpoint(
        module_path,
        epoch=epoch,
        overwrite_cfg=overwrite_cfg,
        use_ema_weights=True
    )
    model.freeze()
    model = model.cuda(device)
    print("Successfully loaded model.")

    return model, data_module


def get_mdt():
    cfg = OmegaConf.create()
    cfg.train_folder = "/mnt/dongxu-fs1/data-ssd/geyuan/code/mdt24rss_fork/logs/runs/2024-11-11/02-50-40"
    cfg.weight_folder = "libero90"
    cfg.sampler_type = "ddim"
    cfg.multistep = 10
    cfg.num_sampling_steps = 10
    cfg.sigma_min = 0.001
    cfg.sigma_max = 80
    cfg.noise_scheduler = "exponential"

    checkpoint = get_last_checkpoint(Path(cfg.train_folder))

    model, _ = get_model_from_hydra_configs(cfg.train_folder, checkpoint, device_id=0)
    model.num_sampling_steps = cfg.num_sampling_steps
    model.sampler_type = cfg.sampler_type
    model.multistep = cfg.multistep
    if cfg.sigma_min is not None:
        model.sigma_min = cfg.sigma_min
    if cfg.sigma_max is not None:
        model.sigma_max = cfg.sigma_max
    if cfg.noise_scheduler is not None:
        model.noise_scheduler = cfg.noise_scheduler

    model.eval()
    return model


def main():
    get_mdt()


if __name__ == '__main__':
    main()


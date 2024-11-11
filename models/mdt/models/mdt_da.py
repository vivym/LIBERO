import logging
import os
from typing import Any, Dict, NamedTuple, Optional, Tuple, List, Union
from functools import partial
import copy

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch.distributed as dist
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
import einops
import torch.optim as optim
import wandb

from mdt.models.edm_diffusion.gc_sampling import *
from mdt.models.edm_diffusion.utils import append_dims
from mdt.utils.lr_schedulers.tri_stage_scheduler import TriStageLRScheduler
from mdt.models.perceptual_encoders.no_encoder import NoEncoder
from mdt.models.networks.transformers.transformer_blocks import ClipStyleProjection
from mdt.callbacks.ema import EMA
from mdt.models.perceptual_encoders.resnets import BesoResNetEncoder

logger = logging.getLogger(__name__)


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

    for name, submodule in model.named_modules():
        # Adjusting the condition to capture the desired layers
        if '.' not in name or name.count('.') <= 10:  # Can be adjusted based on your model structure
            # Counting parameters including submodules
            submodule_params = sum(p.numel() for p in submodule.parameters())
            if submodule_params > 0:
                print(f"{name} - Total Params: {submodule_params}")


class MDTDomainAdaptVisualEncoder(pl.LightningModule):
    """
    The lightning module used for training.
    """

    def __init__(
            self,
            language_goal: DictConfig,
            visual_goal: DictConfig,
            img_gen: DictConfig,
            model: DictConfig,
            domain_adapt: DictConfig,
            optimizer: DictConfig,
            lr_scheduler: DictConfig,
            latent_dim: int = 512,
            multistep: int = 10,
            sampler_type: str = 'ddim',
            num_sampling_steps: int = 10,
            sigma_data: float = 0.5,
            sigma_min: float = 0.001,
            sigma_max: float = 80,
            noise_scheduler: str = 'exponential',
            sigma_sample_density_type: str = 'loglogistic',
            use_lr_scheduler: bool = True,
            act_window_size: int = 10,
            cont_alpha: int = 1,
            masked_beta: float = 1,
            use_distributed_clip: bool = False,
            use_text_not_embedding: bool = False,
            ckpt_path=None,
            seed: int = 42,
            debug_diff_loss: bool = False,
            debug_tsne: bool = False,
            shuffle_target_goal: bool = True,
            cfg_drop_ratio: float = 0.,
            reg_source_diff_loss: bool = False,
    ):
        super(MDTDomainAdaptVisualEncoder, self).__init__()
        self.automatic_optimization = False  # manually backward
        print('[MDTDomainAdaptVisualEncoder] Set automatic optimization to False!')
        self.latent_dim = latent_dim
        img_gen['context_dim'] = self.latent_dim
        self.static_resnet = BesoResNetEncoder(self.latent_dim)
        self.gripper_resnet = BesoResNetEncoder(self.latent_dim)
        self.act_window_size = act_window_size
        self.gen_img = hydra.utils.instantiate(img_gen).to(self.device)
        self.seed = seed
        self.use_lr_scheduler = use_lr_scheduler
        # goal encoders
        self.visual_goal = hydra.utils.instantiate(visual_goal).to(self.device)
        self.language_goal = hydra.utils.instantiate(language_goal) if language_goal else None
        # policy network
        self.model = hydra.utils.instantiate(model).to(self.device)
        self.modality_scope = "vis"
        self.optimizer_config = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_hyperparameters()
        self.masked_beta = masked_beta
        # diffusion stuff
        self.sampler_type = sampler_type
        self.num_sampling_steps = num_sampling_steps
        self.noise_scheduler = noise_scheduler
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_sample_density_type = sigma_sample_density_type
        # for inference
        self.rollout_step_counter = 0
        self.multistep = multistep
        self.latent_goal = None
        self.plan = None
        self.state_recons = False
        self.cont_alpha = cont_alpha
        self.use_text_not_embedding = use_text_not_embedding
        # print_model_parameters(self.perceptual_encoder.perceiver_resampler)
        # for clip loss ground truth plot
        self.cont_loss = self.clip_auxiliary_loss
        self.cont_loss_type = 'infonce'
        self.use_distributed_clip = use_distributed_clip
        self.clip_proj = ClipStyleProjection(
            clip_style='single_token',
            token_dim=self.latent_dim,
            clip_token_index=1,
            num_token=3,
        )
        self.clip_loss_type = 'symmetric'
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.ema_callback_idx = None

        # Load pretrained checkpoint
        if ckpt_path is not None:
            self.load_pretrained_parameters(ckpt_path)
        else:
            pass
            # raise ValueError('[MDTDomainAdaptVisualEncoder] ckpt_path must be provided!')

        # Create model copies for domain adaptation AFTER loading pretrained weights
        self.source_static_resnet = copy.deepcopy(self.static_resnet)
        self.source_gripper_resnet = copy.deepcopy(self.gripper_resnet)
        self.source_model = copy.deepcopy(self.model)
        self.placeholder_param = torch.nn.Parameter(torch.ones([1]))
        # For domain adaptation
        self.use_da_vis1: bool = domain_adapt.use_da_visual in ('both', 'static')
        self.use_da_vis2: bool = domain_adapt.use_da_visual in ('both', 'gripper')
        self.use_da_act: bool = domain_adapt.use_da_act
        self.act_loss_from: str = domain_adapt.act_loss_from
        self.act_layers: int = domain_adapt.act_layers
        self.act_weights: str = domain_adapt.act_weights
        # Register Adapter blocks for trainable modules
        if 'adapter' in domain_adapt.act_weights:
            self.model.inner_model.decoder.register_adapter()
        if debug_diff_loss:
            self.use_da_vis1 = self.use_da_vis2 = False
        if self.use_da_vis1:
            self.da_vis1_loss = hydra.utils.instantiate(domain_adapt.visual_da).to(self.device)
        if self.use_da_vis2:
            self.da_vis2_loss = hydra.utils.instantiate(domain_adapt.visual_da).to(self.device)
        if self.use_da_act:
            self.da_act_loss = hydra.utils.instantiate(domain_adapt.action_da).to(self.device)
        self.cache_da_d_loss = 0.
        self.cache_wdist = 0.
        self.cache_da_g_loss = 0.
        self.shuffle_target_goal = shuffle_target_goal
        self.cfg_drop_ratio = cfg_drop_ratio
        self.reg_source_diff_loss = reg_source_diff_loss
        # For visualization
        self.cache_s_vis1 = []
        self.cache_t_vis1 = []
        self.cache_s_vis2 = []
        self.cache_t_vis2 = []
        self.cache_s_emb = []
        self.cache_t_emb = []
        self.cache_s_pred_a0 = []
        self.cache_t_pred_a0 = []
        self.cache_s_action_gt = []
        self.cache_t_action_gt = []
        self.cache_s_ca = []
        self.cache_t_ca = []
        self.cache_s_k = []
        self.cache_t_k = []
        self.cache_s_v = []
        self.cache_t_v = []
        self.cache_s_q = []
        self.cache_t_q = []
        # For debug
        self.debug_diff_loss = debug_diff_loss
        if self.debug_diff_loss:
            self.shuffle_target_goal = False
            self.cfg_drop_ratio = 0.
        self.debug_tsne = debug_tsne
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def load_pretrained_parameters(self, ckpt_path):
        """
        Load the pretrained parameters from the provided path.
        """
        print("Loading pretrained parameters")
        checkpoint_data = torch.load(ckpt_path)
        '''if 'callbacks'''
        if "ema_weights" in checkpoint_data['callbacks']['EMA']:
            ema_weights_list = checkpoint_data['callbacks']['EMA']['ema_weights']

            # Model's params dict
            model_weights_dict = {name: v for name, v in self.named_parameters()}
            popped_tiny_names = ['cross_att.bias', 'attn.bias']
            model_need_loading_weights_dict = {name: None for name, v in self.named_parameters()}
            missing_keys, unexpected_keys = [], []
            for tiny_name in popped_tiny_names:
                for name in model_need_loading_weights_dict.keys():
                    if tiny_name in name:
                        model_need_loading_weights_dict.pop(name)
                        missing_keys.append(name)

            # Convert list of tensors to a state_dict format
            ema_weights_dict = {name: ema_weights_list[i] for i, (name, _) in enumerate(model_need_loading_weights_dict.items())}
            # missing_keys, unexpected_keys = self.load_state_dict(ema_weights_dict)
            for name, v in ema_weights_dict.items():
                if name in model_weights_dict.keys():
                    model_weights_dict[name] = v
                else:
                    unexpected_keys.append(name)
            self.load_state_dict(model_weights_dict)
            print(f"Successfully loaded EMA weights from checkpoint! "
                  f"missing: {missing_keys}, unexpected: {unexpected_keys}")
        else:
            self.load_state_dict(checkpoint_data['state_dict'])
        print("Successfully loaded weights from checkpoint!")

    def configure_optimizers(self):
        """
        Initialize optimizers and learning rate schedulers based on model configuration.
        """
        # Configuration for models using transformer weight decay
        g_vis1_optim_groups = []
        d_vis1_optim_groups = []
        g_vis2_optim_groups = []
        d_vis2_optim_groups = []
        g_act_optim_groups = []
        d_act_optim_groups = []

        ''' Frozen modules '''
        self.set_requires_grad(self.visual_goal, False)
        self.set_requires_grad(self.source_static_resnet, False)
        self.set_requires_grad(self.source_gripper_resnet, False)
        self.set_requires_grad(self.source_model, False)

        self.set_requires_grad(self.gen_img, False)
        # self.set_requires_grad(self.clip_proj, False)
        # self.logit_scale.requires_grad = False

        ''' Visual Encoder '''
        self.set_requires_grad(self.static_resnet, False)
        if self.use_da_vis1:
            self.set_requires_grad(self.static_resnet, True)
            self.static_resnet.freeze_backbone()
            g_vis1_optim_groups.extend([
                {"params": self.static_resnet.trainable_params(), "lr": self.optimizer_config.vis1_lr},
            ])
        else:
            self.set_requires_grad(self.static_resnet, False)
            g_vis1_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])  # placeholder

        self.set_requires_grad(self.gripper_resnet, False)
        if self.use_da_vis2:
            self.set_requires_grad(self.gripper_resnet, True)
            self.gripper_resnet.freeze_backbone()
            g_vis2_optim_groups.extend([
                {"params": self.gripper_resnet.trainable_params(), "lr": self.optimizer_config.vis2_lr},
            ])
        else:
            self.set_requires_grad(self.gripper_resnet, False)
            g_vis2_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])  # placeholder

        # g_visual_optim_groups.extend([
        #     # {"params": self.clip_proj.parameters(), "weight_decay": self.optimizer_config.obs_encoder_weight_decay},
        #     # {"params": self.logit_scale, "weight_decay": self.optimizer_config.obs_encoder_weight_decay},
        # ])

        ''' Transformer Encoder & Decoder '''
        self.set_requires_grad(self.model, False)
        if self.use_da_act:
            self.set_requires_grad(self.model, True)
            unfreeze_ca = "ca" in self.act_weights
            unfreeze_adapter = "adapter" in self.act_weights
            if not self.debug_diff_loss:  # when NOT debug diff loss, finetuning CA params of diffusion policy
                self.model.inner_model.freeze_backbone(
                    unfreeze_params=self.act_weights
                )
            else:
                # Debug diff loss
                self.model.inner_model.freeze_backbone(
                    unfreeze_params=self.act_weights
                )  # using the same setting with da_act
                pass  # finetuning all params
            g_act_optim_groups.extend([
                {"params": self.model.inner_model.trainable_params(), "lr": self.optimizer_config.act_lr},
                # {"params": self.gen_img.parameters(), "weight_decay": self.optimizer_config.transformer_weight_decay},
            ])
        else:
            g_act_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])  # placeholder

        ''' Adaptation Discriminator '''
        if self.use_da_vis1:
            self.set_requires_grad(self.da_vis1_loss, True)
            d_vis1_optim_groups.extend([
                {"params": self.da_vis1_loss.parameters(), "lr": self.optimizer_config.vis1_lr},
            ])
        else:
            d_vis1_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])

        if self.use_da_vis2:
            self.set_requires_grad(self.da_vis2_loss, True)
            d_vis2_optim_groups.extend([
                {"params": self.da_vis2_loss.parameters(), "lr": self.optimizer_config.vis2_lr},
            ])
        else:
            d_vis2_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])

        if self.use_da_act:
            self.set_requires_grad(self.da_act_loss, True)
            d_act_optim_groups.extend([
                {"params": self.da_act_loss.parameters(),
                 "lr": self.optimizer_config.act_lr},
            ])
        else:
            d_act_optim_groups.extend([{"params": self.placeholder_param, "lr": 0.}])

        ''' Optimizer '''
        g_vis1_optimizer = torch.optim.AdamW(g_vis1_optim_groups, weight_decay=self.optimizer_config.obs_encoder_weight_decay,
                                             betas=self.optimizer_config.betas)
        d_vis1_optimizer = torch.optim.AdamW(d_vis1_optim_groups, weight_decay=self.optimizer_config.obs_encoder_weight_decay,
                                             betas=self.optimizer_config.betas)

        g_vis2_optimizer = torch.optim.AdamW(g_vis2_optim_groups, weight_decay=self.optimizer_config.obs_encoder_weight_decay,
                                             betas=self.optimizer_config.betas)
        d_vis2_optimizer = torch.optim.AdamW(d_vis2_optim_groups, weight_decay=self.optimizer_config.obs_encoder_weight_decay,
                                             betas=self.optimizer_config.betas)

        g_act_optimizer = torch.optim.AdamW(g_act_optim_groups, weight_decay=self.optimizer_config.transformer_weight_decay,
                                            betas=self.optimizer_config.betas)
        d_act_optimizer = torch.optim.AdamW(d_act_optim_groups, weight_decay=self.optimizer_config.transformer_weight_decay,
                                            betas=self.optimizer_config.betas)


        ''' Learning Rate Scheduler '''
        # Optionally initialize the scheduler
        if self.use_lr_scheduler:
            vis1_lr_configs = OmegaConf.create(self.lr_scheduler.vis1_lr_scheduler)
            vis2_lr_configs = OmegaConf.create(self.lr_scheduler.vis2_lr_scheduler)
            act_lr_configs = OmegaConf.create(self.lr_scheduler.act_lr_scheduler)

            g_vis1_scheduler = TriStageLRScheduler(g_vis1_optimizer, vis1_lr_configs)
            g_vis1_lr_scheduler = {
                "scheduler": g_vis1_scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            d_vis1_scheduler = TriStageLRScheduler(d_vis1_optimizer, vis1_lr_configs)
            d_vis1_lr_scheduler = {
                "scheduler": d_vis1_scheduler,
                "interval": 'step',
                "frequency": 1,
            }

            g_vis2_scheduler = TriStageLRScheduler(g_vis2_optimizer, vis2_lr_configs)
            g_vis2_lr_scheduler = {
                "scheduler": g_vis2_scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            d_vis2_scheduler = TriStageLRScheduler(d_vis2_optimizer, vis2_lr_configs)
            d_vis2_lr_scheduler = {
                "scheduler": d_vis2_scheduler,
                "interval": 'step',
                "frequency": 1,
            }

            g_act_scheduler = TriStageLRScheduler(g_act_optimizer, act_lr_configs)
            g_act_lr_scheduler = {
                "scheduler": g_act_scheduler,
                "interval": 'step',
                "frequency": 1,
            }
            d_act_scheduler = TriStageLRScheduler(d_act_optimizer, act_lr_configs)
            d_act_lr_scheduler = {
                "scheduler": d_act_scheduler,
                "interval": 'step',
                "frequency": 1,
            }

            return (
                {"optimizer": g_vis1_optimizer, "lr_scheduler": g_vis1_lr_scheduler},
                {"optimizer": d_vis1_optimizer, "lr_scheduler": d_vis1_lr_scheduler},
                {"optimizer": g_vis2_optimizer, "lr_scheduler": g_vis2_lr_scheduler},
                {"optimizer": d_vis2_optimizer, "lr_scheduler": d_vis2_lr_scheduler},
                {"optimizer": g_act_optimizer, "lr_scheduler": g_act_lr_scheduler},
                {"optimizer": d_act_optimizer, "lr_scheduler": d_act_lr_scheduler},
            )
        else:
            return (g_vis1_optimizer, d_vis1_optimizer,
                    g_vis2_optimizer, d_vis2_optimizer,
                    g_act_optimizer, d_act_optimizer,
                    )

    @staticmethod
    def set_requires_grad(model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad
        if requires_grad:
            model.train()
        else:
            model.eval()

    def calc_grad_and_param_norm(self, module: Union[nn.Module, List[nn.Module]],
                                 sqrt_out: bool = True,
                                 ):
        total_grad_norm = 0.0
        total_param_norm = 0.0
        total_ratio_norm = 0.0
        if isinstance(module, list):
            for m in module:
                m_grad, m_param, m_ratio = self.calc_grad_and_param_norm(m, sqrt_out=False)  # recursive
                total_grad_norm += m_grad
                total_param_norm += m_param
                total_ratio_norm += m_ratio
        else:
            assert isinstance(module, nn.Module)
            for name, p in module.named_parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.norm().item() ** 2
                    total_ratio_norm += (p.grad.norm().item() / (1e-8 + p.data.norm().item())) ** 2
                total_param_norm += p.norm().item() ** 2
        if sqrt_out:
            total_grad_norm = total_grad_norm ** 0.5
            total_param_norm = total_param_norm ** 0.5
            total_ratio_norm = total_ratio_norm ** 0.5
        return total_grad_norm, total_param_norm, total_ratio_norm

    def on_before_zero_grad(self, optimizer=None):
        vis1_grad_norm, vis1_total_norm, _ = self.calc_grad_and_param_norm(self.static_resnet)
        vis2_grad_norm, vis2_total_norm, _ = self.calc_grad_and_param_norm(self.gripper_resnet)
        act_grad_norm, act_total_norm, _ = self.calc_grad_and_param_norm(self.model)

        if self.debug_diff_loss:
            q_layers = [layer.cross_att.query for layer in self.model.inner_model.decoder.blocks]
            k_layers = [layer.cross_att.key for layer in self.model.inner_model.decoder.blocks]
            v_layers = [layer.cross_att.value for layer in self.model.inner_model.decoder.blocks]
            _, _, q_grad_norm = self.calc_grad_and_param_norm(q_layers)
            _, _, k_grad_norm = self.calc_grad_and_param_norm(k_layers)
            _, _, v_grad_norm = self.calc_grad_and_param_norm(v_layers)
            self.log("grad/ca_q", q_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
            self.log("grad/ca_k", k_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
            self.log("grad/ca_v", v_grad_norm, on_step=True, on_epoch=False, sync_dist=True)

            q_layers = [layer.attn.query for layer in self.model.inner_model.decoder.blocks]
            k_layers = [layer.attn.key for layer in self.model.inner_model.decoder.blocks]
            v_layers = [layer.attn.value for layer in self.model.inner_model.decoder.blocks]
            _, _, q_grad_norm = self.calc_grad_and_param_norm(q_layers)
            _, _, k_grad_norm = self.calc_grad_and_param_norm(k_layers)
            _, _, v_grad_norm = self.calc_grad_and_param_norm(v_layers)
            self.log("grad/sa_q", q_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
            self.log("grad/sa_k", k_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
            self.log("grad/sa_v", v_grad_norm, on_step=True, on_epoch=False, sync_dist=True)

            ca_layers = [layer.cross_att for layer in self.model.inner_model.decoder.blocks]
            sa_layers = [layer.attn for layer in self.model.inner_model.decoder.blocks]
            mlp_layers = [layer.mlp for layer in self.model.inner_model.decoder.blocks]
            _, _, ca_grad_norm= self.calc_grad_and_param_norm(ca_layers)
            _, _, sa_grad_norm = self.calc_grad_and_param_norm(sa_layers)
            _, _, mlp_grad_norm = self.calc_grad_and_param_norm(mlp_layers)
            self.log("grad/ca_all", ca_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
            self.log("grad/sa_all", sa_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
            self.log("grad/mlp", mlp_grad_norm, on_step=True, on_epoch=False, sync_dist=True)

        self.log("train/vis1_grad_norm", vis1_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/vis1_total_norm", vis1_total_norm, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/vis2_grad_norm", vis2_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/vis2_total_norm", vis2_total_norm, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/act_grad_norm", act_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/act_total_norm", act_total_norm, on_step=True, on_epoch=False, sync_dist=True)

        total_grad_norm = vis1_grad_norm + vis2_grad_norm + act_grad_norm
        total_param_norm = vis1_total_norm + vis2_total_norm + act_total_norm
        self.log("train/grad_norm", total_grad_norm, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/param_norm", total_param_norm, on_step=True, on_epoch=False, sync_dist=True)

    def clip_extra_forward(self, perceptual_emb, latent_goal, actions, sigmas, noise):

        self.model.train()
        noised_input = actions + noise * append_dims(sigmas, actions.ndim)
        context = self.model.forward_context_only(perceptual_emb, noised_input, latent_goal, sigmas)
        return context

    def shuffle_tensor(self, x: torch.Tensor, dim: int = 0):
        if self.shuffle_target_goal:
            idx = torch.randperm(x.size(dim)).to(x.device)
            return x.index_select(dim, idx)
        else:
            return x  # do nothing

    @torch.no_grad()
    def log_kl_loss(self, target, source, suffix: str, total_bs):
        kl_loss = self.kl(F.log_softmax(target, dim=-1), F.softmax(source, dim=-1))
        self.log(f"kl_loss/{suffix}", kl_loss.clone().detach(),
                 on_step=True, on_epoch=False, sync_dist=True, batch_size=total_bs)
        return

    def training_step(self, batch: Dict[str, Dict], batch_idx: int,
                      dataloader_idx: int = 0) -> torch.Tensor:  # type: ignore
        """
        Compute and return the training loss for the MDT Agent.
        The training loss consists of the score matching loss of the diffusion model
        and the contrastive loss of the CLIP model for the multimodal encoder.

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            loss tensor
        """
        (g_vis1_opt, d_vis1_opt,
         g_vis2_opt, d_vis2_opt,
         g_act_opt, d_act_opt) = self.optimizers(use_pl_optimizer=True)  # PL optimizer handles grad_scaling
        if self.use_lr_scheduler:
            (g_vis1_sch, d_vis1_sch,
             g_vis2_sch, d_vis2_sch,
             g_act_sch, d_act_sch) = self.lr_schedulers()
        else:
            class EmptySch(object):
                def step(self):
                    pass
            (g_vis1_sch, d_vis1_sch,
             g_vis2_sch, d_vis2_sch,
             g_act_sch, d_act_sch) = EmptySch(), EmptySch(), EmptySch(), EmptySch(), EmptySch(), EmptySch()

        (total_loss, action_loss, cont_loss, id_loss, img_gen_loss,
            da_d1_loss, da_g1_loss, da_d2_loss, da_g2_loss, da_d_act_loss, da_g_act_loss,
            w_dist_1, gp_1, w_dist_2, gp_2, w_dist_act, gp_act) = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )
        losses = {
            'total_loss': total_loss,
            'action_loss': action_loss,
            'cont_loss': cont_loss,
            'img_gen_loss': img_gen_loss,
            'da_d1_loss': da_d1_loss,
            'da_g1_loss': da_g1_loss,
            'da_d2_loss': da_d2_loss,
            'da_g2_loss': da_g2_loss,
            'da_d_act_loss': da_d_act_loss,
            'da_g_act_loss': da_g_act_loss,
            'w_dist_1': w_dist_1,
            'gp_1': gp_1,
            'w_dist_2': w_dist_2,
            'gp_2': gp_2,
            'w_dist_act': w_dist_act,
            'gp_act': gp_act,
        }
        encoders_dict = {}
        batch_size: Dict[str, int] = {}
        s_batch_len = 0
        t_batch_len = 0
        total_bs = 0

        s_latent_static_emb_dict = {}
        t_latent_static_emb_dict = {}
        s_latent_gripper_emb_dict = {}
        t_latent_gripper_emb_dict = {}
        s_latent_encoder_emb_dict = {}
        t_latent_encoder_emb_dict = {}
        s_latent_action_emb_dict = {}
        t_latent_action_emb_dict = {}
        s_pred_a0_dict = {}
        t_pred_a0_dict = {}
        s_action_gt_dict = {}
        t_action_gt_dict = {}
        s_sa_dict = {}
        t_sa_dict = {}
        s_ca_dict = {}
        t_ca_dict = {}
        s_ks_dict: Dict[str, List[torch.Tensor]] = {}
        t_ks_dict: Dict[str, List[torch.Tensor]] = {}
        s_vs_dict: Dict[str, List[torch.Tensor]] = {}
        t_vs_dict: Dict[str, List[torch.Tensor]] = {}
        s_qs_dict: Dict[str, List[torch.Tensor]] = {}
        t_qs_dict: Dict[str, List[torch.Tensor]] = {}
        s_qk_dict = {}
        t_qk_dict = {}
        s_qkv_dict = {}
        t_qkv_dict = {}
        s_mlp_dict = {}
        t_mlp_dict = {}

        source_act_0 = None
        common_noise = None
        common_sigmas = None
        common_sigma_emb = None
        max_bs = None
        use_zero_goal = np.random.uniform(0, 1) <= self.cfg_drop_ratio
        for self.modality_scope, dataset_batch in batch.items():  # order:lang_source,lang_target,vis_source,vis_target
            # if 'lang' in self.modality_scope:  # TODO: skip:'lang_source', 'lang_target'
            #     continue
            # if 'source' in self.modality_scope:
            #     continue
            if dataset_batch is not None:
                # rand_noise = torch.randn_like(dataset_batch["actions"]) if rand_noise is None else rand_noise
                cur_bs = dataset_batch["actions"].shape[0]
                if max_bs is None:
                    max_bs = cur_bs
                elif cur_bs < max_bs:
                    print(f'[Warning] data dropped, bs={cur_bs}<max_bs={max_bs}, modality={self.modality_scope}')
                    continue
            else:
                print(f'[Warning] dataset_batch is None for {self.modality_scope}')

            if 'source' in self.modality_scope:
                # Compute the required embeddings
                s_perceptual_emb, latent_goal, image_latent_goal = self.compute_input_embeddings(
                    dataset_batch, is_target=False)
                latent_goal = latent_goal if not use_zero_goal else torch.zeros_like(latent_goal)
                # 'static' or 'gripper':(bs,1,512)
                s_batch_len += 1

                # Compute diffusion loss without actions, just for sigmas
                source_act_0 = dataset_batch['actions']
                _, sigmas, noise, pred_a0 = self.diffusion_loss(
                    s_perceptual_emb,
                    latent_goal,  # (64,512)
                    source_act_0,  # no need to calculate loss
                    is_target=False,
                    sigmas=common_sigmas,
                    is_da=False,
                )  # will call enc_only_forward() and dec_only_forward()
                common_noise = noise  # S and T can use different noise
                common_sigmas = sigmas if common_sigmas is None else common_sigmas  # only assign sigmas once
                latent_encoder_emb = self.source_model.inner_model.latent_encoder_emb
                latent_action_emb = self.source_model.inner_model.cache_action_emb
                common_one_modal_sigma_emb = self.source_model.inner_model.cache_sigma_emb
                assert common_one_modal_sigma_emb is not None
                if common_sigma_emb is None:
                    common_sigma_emb = common_one_modal_sigma_emb
                else:
                    common_sigma_emb = torch.cat((common_sigma_emb, common_one_modal_sigma_emb), dim=0)  # repeat
                action_output = pred_a0
                sa_output = self.source_model.inner_model.cache_sa_output
                ca_output = self.source_model.inner_model.cache_ca_output  # [B,10,512]*6
                k_output = self.source_model.inner_model.cache_k_output  # [B,8,3,64]*6
                v_output = self.source_model.inner_model.cache_v_output
                q_output = self.source_model.inner_model.cache_q_output  # [B,8,10,64]*6
                qk_output = self.source_model.inner_model.cache_qk_output  # (B,8,10,3)*6
                qkv_output = self.source_model.inner_model.cache_qkv_output  # [B,10,512]
                mlp_out = self.source_model.inner_model.cache_mlp_output  # (B,10,512)*6

                save_key = self.modality_scope[:-len('_source')]
                s_latent_static_emb_dict[save_key] = s_perceptual_emb['static']  # (bs,1,512)
                s_latent_gripper_emb_dict[save_key] = s_perceptual_emb['gripper']
                s_latent_encoder_emb_dict[save_key] = latent_encoder_emb
                s_latent_action_emb_dict[save_key] = latent_action_emb
                s_pred_a0_dict[save_key] = action_output
                s_action_gt_dict[save_key] = dataset_batch["actions"]
                s_sa_dict[save_key] = sa_output
                s_ca_dict[save_key] = ca_output  # [B,10,512]*6
                s_ks_dict[save_key] = [feat.reshape(feat.shape[0], -1) for feat in k_output]  # (B,8,3,64)->(B,1536)
                s_vs_dict[save_key] = [feat.reshape(feat.shape[0], -1) for feat in v_output]
                s_qs_dict[save_key] = [einops.rearrange(feat, 'b h t d -> b t (h d)') for feat in q_output]  # (B,8,10,64)->(B,10,512)
                s_qk_dict[save_key] = qk_output
                s_qkv_dict[save_key] = qkv_output  # [B,10,512]*6
                s_mlp_dict[save_key] = mlp_out

                if self.reg_source_diff_loss > 0.:
                    # DEBUG: train target with source data
                    t_perceptual_emb, latent_goal, image_latent_goal = self.compute_input_embeddings(
                        dataset_batch, is_target=True)
                    # t_perceptual_emb = {k: v.clone.detach() for k, v in t_perceptual_emb.items()}
                    diff_loss, sigmas, noise, pred_a0 = self.diffusion_loss(
                        t_perceptual_emb,
                        latent_goal,
                        dataset_batch['actions'],
                        is_target=True,
                        is_da=False,
                    )
                    losses['action_loss'] += diff_loss * self.reg_source_diff_loss

            elif 'target' in self.modality_scope:
                t_perceptual_emb, latent_goal, image_latent_goal = self.compute_input_embeddings(
                    dataset_batch, is_target=True)
                latent_goal = latent_goal if not use_zero_goal else torch.zeros_like(latent_goal)
                t_batch_len += 1

                # Compute diffusion loss without actions, just for sigmas
                rand_noise = torch.randn_like(common_noise)  # TODO: re-rand
                shuffled_goal = self.shuffle_tensor(latent_goal)  # TODO: setting in config
                assert source_act_0 is not None
                _, sigmas, noise, pred_a0 = self.diffusion_loss(
                    t_perceptual_emb,
                    shuffled_goal,
                    source_act_0,  # TODO: no need to calculate loss, ori:rand_noise
                    is_target=True,
                    sigmas=common_sigmas,
                    is_da=False,
                    noise=rand_noise,
                )
                latent_encoder_emb = self.model.inner_model.latent_encoder_emb
                latent_action_emb = self.model.inner_model.cache_action_emb
                action_output = pred_a0
                sa_output = self.model.inner_model.cache_sa_output
                ca_output = self.model.inner_model.cache_ca_output  # [B,10,512]*6
                k_output = self.model.inner_model.cache_k_output
                v_output = self.model.inner_model.cache_v_output
                q_output = self.model.inner_model.cache_q_output
                qk_output = self.model.inner_model.cache_qk_output
                qkv_output = self.model.inner_model.cache_qkv_output
                mlp_out = self.model.inner_model.cache_mlp_output  # (B,10,512)*6

                # Compute diffusion loss for DEBUG (DO NOT use in method!)
                if self.debug_diff_loss:
                    diff_loss, sigmas, noise, pred_a0 = self.diffusion_loss(
                        t_perceptual_emb,
                        latent_goal,
                        dataset_batch['actions'],
                        is_target=True,
                        is_da=False,
                    )
                    losses['action_loss'] += diff_loss

                # # Compute the masked generative foresight loss (only for target)
                # if not isinstance(self.gen_img, NoEncoder):
                #     rgb_static_goal = dataset_batch["rgb_obs"]['gen_static']
                #     rgb_gripper_goal = dataset_batch["rgb_obs"]['gen_gripper']
                #     img_gen_frame_diff = dataset_batch[
                #         'future_frame_diff'] if "future_frame_diff" in dataset_batch else 3
                #     # combine both goal images
                #     rgb_pred_goal = torch.cat([rgb_static_goal, rgb_gripper_goal], dim=1)
                #     img_gen_embed = latent_encoder_emb
                #     img_gen_loss_part = self.compute_img_gen_loss(img_gen_embed, rgb_pred_goal,
                #                                                   img_gen_frame_diff=img_gen_frame_diff)
                #     img_gen_loss += img_gen_loss_part * self.masked_beta

                # # Compute the Contrastive Latent Alignment Loss (only for target)
                # cont_loss_part = self.compute_contrastive_loss(
                #     t_perceptual_emb,
                #     latent_goal,
                #     image_latent_goal,
                #     dataset_batch,
                #     sigmas,
                #     noise
                # )
                # cont_loss += self.cont_alpha * cont_loss_part

                save_key = self.modality_scope[:-len('_target')]
                t_latent_static_emb_dict[save_key] = t_perceptual_emb['static']  # (bs,1,512)
                t_latent_gripper_emb_dict[save_key] = t_perceptual_emb['gripper']
                t_latent_encoder_emb_dict[save_key] = latent_encoder_emb
                t_latent_action_emb_dict[save_key] = latent_action_emb  # (bs,10,512)
                t_pred_a0_dict[save_key] = action_output
                t_action_gt_dict[save_key] = dataset_batch["actions"]
                t_sa_dict[save_key] = sa_output
                t_ca_dict[save_key] = ca_output # [B,10,512]*6
                t_ks_dict[save_key] = [feat.reshape(feat.shape[0], -1) for feat in k_output]  # (B,8,3,64)->(B,1536)
                t_vs_dict[save_key] = [feat.reshape(feat.shape[0], -1) for feat in v_output]
                t_qs_dict[save_key] = [einops.rearrange(feat, 'b h t d -> b t (h d)') for feat in q_output]  # (B,8,10,64)
                t_qk_dict[save_key] = qk_output
                t_qkv_dict[save_key] = qkv_output  # [B,10,512]*6
                t_mlp_dict[save_key] = mlp_out

            else:
                raise KeyError(f'[MDTDomainAdaptVisualEncoder] batch key:{self.modality_scope} not supported')

            batch_size[self.modality_scope] = dataset_batch["actions"].shape[0]
            total_bs += dataset_batch["actions"].shape[0]

        # divide accumulated gradients by number of datasets
        batch_len = s_batch_len + t_batch_len

        # sort dict
        def sort_dict(dict1):
            return {key: dict1[key] for key in sorted(dict1.keys())}

        t_latent_static_emb_dict = sort_dict(t_latent_static_emb_dict)
        s_latent_static_emb_dict = sort_dict(s_latent_static_emb_dict)
        t_latent_gripper_emb_dict = sort_dict(t_latent_gripper_emb_dict)
        s_latent_gripper_emb_dict = sort_dict(s_latent_gripper_emb_dict)
        t_latent_encoder_emb_dict = sort_dict(t_latent_encoder_emb_dict)
        s_latent_encoder_emb_dict = sort_dict(s_latent_encoder_emb_dict)
        t_latent_action_emb_dict = sort_dict(t_latent_action_emb_dict)
        s_latent_action_emb_dict = sort_dict(s_latent_action_emb_dict)
        t_sa_dict = sort_dict(t_sa_dict)
        s_sa_dict = sort_dict(s_sa_dict)
        t_ca_dict = sort_dict(t_ca_dict)
        s_ca_dict = sort_dict(s_ca_dict)
        t_ks_dict = sort_dict(t_ks_dict)
        s_ks_dict = sort_dict(s_ks_dict)
        t_vs_dict = sort_dict(t_vs_dict)
        s_vs_dict = sort_dict(s_vs_dict)
        t_qs_dict = sort_dict(t_qs_dict)
        s_qs_dict = sort_dict(s_qs_dict)
        t_qk_dict = sort_dict(t_qk_dict)
        s_qk_dict = sort_dict(s_qk_dict)
        t_qkv_dict = sort_dict(t_qkv_dict)
        s_qkv_dict = sort_dict(s_qkv_dict)
        t_mlp_dict = sort_dict(t_mlp_dict)
        s_mlp_dict = sort_dict(s_mlp_dict)
        t_pred_a0_dict = sort_dict(t_pred_a0_dict)
        s_pred_a0_dict = sort_dict(s_pred_a0_dict)

        t_feat_for_da_vis1 = torch.cat([v for v in t_latent_static_emb_dict.values()], dim=0)
        s_feat_for_da_vis1 = torch.cat([v for v in s_latent_static_emb_dict.values()], dim=0)
        t_feat_for_da_vis2 = torch.cat([v for v in t_latent_gripper_emb_dict.values()], dim=0)
        s_feat_for_da_vis2 = torch.cat([v for v in s_latent_gripper_emb_dict.values()], dim=0)
        t_feat_for_da_enc = torch.cat([v for v in t_latent_encoder_emb_dict.values()], dim=0)
        s_feat_for_da_enc = torch.cat([v for v in s_latent_encoder_emb_dict.values()], dim=0)

        def cat_list_tensor(llt: List[List[torch.Tensor]], dim=0) -> List[torch.Tensor]:
            lt1 = llt[0]
            for i in range(len(lt1)):
                lt1[i] = torch.cat([lt[i] for lt in llt], dim=dim)
            return lt1

        # t_feat_for_da_act = torch.cat([v for v in t_latent_action_emb_dict.values()], dim=0)
        # s_feat_for_da_act = torch.cat([v for v in s_latent_action_emb_dict.values()], dim=0)
        t_sa_for_da_act = cat_list_tensor(list(t_sa_dict.values()))  # v:[B,10,512]*6, out:[2*B,10,512]*6
        s_sa_for_da_act = cat_list_tensor(list(s_sa_dict.values()))
        t_ca_for_da_act = cat_list_tensor(list(t_ca_dict.values()))
        s_ca_for_da_act = cat_list_tensor(list(s_ca_dict.values()))

        # for i in range(len(t_sa_for_da_act)):
        #     print(f"layer = {i}")
        #     print(t_sa_for_da_act[i].shape)
        #     print(s_sa_for_da_act[i].shape)
        #     print(t_ca_for_da_act[i].shape)
        #     print(s_ca_for_da_act[i].shape)
        # exit()

        t_pred_a0_for_da_act = torch.cat([v for v in t_pred_a0_dict.values()], dim=0)
        s_pred_a0_for_da_act = torch.cat([v for v in s_pred_a0_dict.values()], dim=0)  # (B,10,7)

        t_k_for_da_act: List[torch.Tensor] = []
        s_k_for_da_act: List[torch.Tensor] = []
        t_v_for_da_act: List[torch.Tensor] = []  # 6*[(B,1536)]
        s_v_for_da_act: List[torch.Tensor] = []  # 6*[(B,1536)]
        t_q_for_da_act: List[torch.Tensor] = []  # 6*[(B,5120)]
        s_q_for_da_act: List[torch.Tensor] = []
        t_qk_for_da_act = []
        s_qk_for_da_act = []
        t_qkv_for_da_act = []
        s_qkv_for_da_act = []
        t_mlp_for_da_act = []
        s_mlp_for_da_act = []
        num_layers = len(list(t_ks_dict.values())[0])
        for l_idx in range(num_layers):
            t_k_for_da_act.append(torch.cat([fs[l_idx] for fs in t_ks_dict.values()], dim=0))
            s_k_for_da_act.append(torch.cat([fs[l_idx] for fs in s_ks_dict.values()], dim=0))
            t_v_for_da_act.append(torch.cat([fs[l_idx] for fs in t_vs_dict.values()], dim=0))
            s_v_for_da_act.append(torch.cat([fs[l_idx] for fs in s_vs_dict.values()], dim=0))
            t_q_for_da_act.append(torch.cat([fs[l_idx] for fs in t_qs_dict.values()], dim=0))
            s_q_for_da_act.append(torch.cat([fs[l_idx] for fs in s_qs_dict.values()], dim=0))
            t_qk_for_da_act.append(torch.cat([fs[l_idx] for fs in t_qk_dict.values()], dim=0))
            s_qk_for_da_act.append(torch.cat([fs[l_idx] for fs in s_qk_dict.values()], dim=0))
            t_qkv_for_da_act.append(torch.cat([fs[l_idx] for fs in t_qkv_dict.values()], dim=0))
            s_qkv_for_da_act.append(torch.cat([fs[l_idx] for fs in s_qkv_dict.values()], dim=0))
            t_mlp_for_da_act.append(torch.cat([fs[l_idx] for fs in t_mlp_dict.values()], dim=0))
            s_mlp_for_da_act.append(torch.cat([fs[l_idx] for fs in s_mlp_dict.values()], dim=0))

        if self.debug_diff_loss:
            for l_idx in range(num_layers):
                t_k, s_k = t_k_for_da_act[l_idx], s_k_for_da_act[l_idx]
                t_v, s_v = t_v_for_da_act[l_idx], s_v_for_da_act[l_idx]
                t_q, s_q = t_q_for_da_act[l_idx], s_q_for_da_act[l_idx]
                self.log_kl_loss(t_k, s_k, f'k_layer{l_idx:02d}', total_bs)
                self.log_kl_loss(t_v, s_v, f'v_layer{l_idx:02d}', total_bs)
                self.log_kl_loss(t_q, s_q, f'q_layer{l_idx:02d}', total_bs)

                half_shape = s_k.shape[0] // 2
                self.log_kl_loss(s_k[:half_shape], s_k[half_shape:], f'source_k_layer{l_idx:02d}', total_bs)
                half_shape = s_v.shape[0] // 2
                self.log_kl_loss(s_v[:half_shape], s_v[half_shape:], f'source_v_layer{l_idx:02d}', total_bs)
                half_shape = s_q.shape[0] // 2
                self.log_kl_loss(s_q[:half_shape], s_q[half_shape:], f'source_q_layer{l_idx:02d}', total_bs)

        if self.act_layers < 0:
            left, right = self.act_layers, None
        else:
            left, right = None, self.act_layers
        t_feat_for_da_act = []
        s_feat_for_da_act = []
        if 'v' in self.act_loss_from:
            t_feat_for_da_act.extend(t_v_for_da_act[left:right])
            s_feat_for_da_act.extend(s_v_for_da_act[left:right])
        if 'k' in self.act_loss_from:
            t_feat_for_da_act.extend(t_k_for_da_act[left:right])
            s_feat_for_da_act.extend(s_k_for_da_act[left:right])
        if 'q' in self.act_loss_from:
            t_feat_for_da_act.extend(t_q_for_da_act[left:right])  # only last layer
            s_feat_for_da_act.extend(s_q_for_da_act[left:right])
        if 'softmax' in self.act_loss_from:
            t_feat_for_da_act.extend(t_qk_for_da_act[left:right])
            s_feat_for_da_act.extend(s_qk_for_da_act[left:right])
        if 'attn' in self.act_loss_from:
            t_feat_for_da_act.extend(t_qkv_for_da_act[left:right])
            s_feat_for_da_act.extend(s_qkv_for_da_act[left:right])
        if 'mlp' in self.act_loss_from:
            t_feat_for_da_act.extend(t_mlp_for_da_act[left:right])
            s_feat_for_da_act.extend(s_mlp_for_da_act[left:right])
        if 'sa' in self.act_loss_from:
            t_feat_for_da_act.extend(t_sa_for_da_act)
            s_feat_for_da_act.extend(s_sa_for_da_act)
        if 'ca' in self.act_loss_from:
            t_feat_for_da_act.extend(t_ca_for_da_act)
            s_feat_for_da_act.extend(s_ca_for_da_act)

        ''' 1. Update discriminator '''
        tsne_batch_nums = 10
        if len(self.cache_t_emb) < tsne_batch_nums:
            t_keys = list(t_latent_action_emb_dict.keys())
            s_keys = list(s_latent_action_emb_dict.keys())
            t_key = t_keys[-1]
            bs = t_latent_action_emb_dict[t_key].shape[0]
            last_dim = t_latent_action_emb_dict[t_key].shape[-1]
            # print(t_keys, s_keys, t_latent_action_emb_dict[t_key].shape)
            self.cache_t_vis1.append(t_latent_static_emb_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_s_vis1.append(s_latent_static_emb_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_t_vis2.append(t_latent_gripper_emb_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_s_vis2.append(s_latent_gripper_emb_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_t_emb.append(t_latent_action_emb_dict[t_key].detach().cpu().reshape(bs, -1).numpy())
            self.cache_s_emb.append(s_latent_action_emb_dict[t_key].detach().cpu().reshape(bs, -1).numpy())
            # self.cache_t_emb.append(t_latent_action_emb_dict[t_key].detach().cpu().reshape(-1, last_dim).numpy())
            # self.cache_s_emb.append(s_latent_action_emb_dict[t_key].detach().cpu().reshape(-1, last_dim).numpy())
            self.cache_t_pred_a0.append(t_pred_a0_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_s_pred_a0.append(s_pred_a0_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_t_action_gt.append(t_action_gt_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_s_action_gt.append(s_action_gt_dict[t_key].detach().float().cpu().reshape(bs, -1).numpy())

            self.cache_t_ca.append([x.detach().float().cpu().reshape(bs, -1).numpy() for x in t_ca_dict[t_key]])
            self.cache_s_ca.append([x.detach().float().cpu().reshape(bs, -1).numpy() for x in s_ca_dict[t_key]])

            # Only show the 1st/3rd/last layer
            self.cache_t_k.append(t_ks_dict[t_key][3].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_s_k.append(s_ks_dict[t_key][3].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_t_v.append(t_vs_dict[t_key][3].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_s_v.append(s_vs_dict[t_key][3].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_t_q.append(t_qs_dict[t_key][3].detach().float().cpu().reshape(bs, -1).numpy())
            self.cache_s_q.append(s_qs_dict[t_key][3].detach().float().cpu().reshape(bs, -1).numpy())

        from mdt.datasets.utils.debug_utils import TSNEHelper
        if self.debug_tsne and (os.environ.get("LOCAL_RANK", "0") == "0" and batch_idx % 200 == 100 and
                len(self.cache_t_emb) >= tsne_batch_nums):
            epoch_idx = self.current_epoch

            # tsne_inputs = np.concatenate(self.cache_t_vis1 + self.cache_s_vis1, axis=0)  # [(B,D)]*20 + [(B,D)]*20
            # helper = TSNEHelper(tsne_inputs)
            # helper.plot_tsne(f'vis1_enc_{epoch_idx:02d}_{batch_idx:05d}')
            #
            # tsne_inputs = np.concatenate(self.cache_t_vis2 + self.cache_s_vis2, axis=0)  # [(B,D)]*20 + [(B,D)]*20
            # helper = TSNEHelper(tsne_inputs)
            # helper.plot_tsne(f'vis2_enc_{epoch_idx:02d}_{batch_idx:05d}')
            #
            # tsne_inputs = np.concatenate(self.cache_t_emb + self.cache_s_emb, axis=0)
            # helper = TSNEHelper(tsne_inputs)
            # helper.plot_tsne(f'action_embedding_{epoch_idx:02d}_{batch_idx:05d}')

            tsne_inputs = np.concatenate(self.cache_t_k + self.cache_s_k, axis=0)
            helper = TSNEHelper(tsne_inputs)
            helper.plot_tsne(f'ca_k_{epoch_idx:02d}_{batch_idx:05d}')

            tsne_inputs = np.concatenate(self.cache_t_v + self.cache_s_v, axis=0)
            helper = TSNEHelper(tsne_inputs)
            helper.plot_tsne(f'ca_v_{epoch_idx:02d}_{batch_idx:05d}')

            tsne_inputs = np.concatenate(self.cache_t_q + self.cache_s_q, axis=0)
            helper = TSNEHelper(tsne_inputs)
            helper.plot_tsne(f'ca_q_{epoch_idx:02d}_{batch_idx:05d}')

            # tsne_inputs = np.concatenate(self.cache_t_pred_a0 + self.cache_s_pred_a0, axis=0)
            # helper = TSNEHelper(tsne_inputs)
            # helper.plot_tsne(f'action_pred_a0_{epoch_idx:02d}_{batch_idx:05d}')

            # tsne_inputs = np.concatenate(self.cache_t_action_gt + self.cache_s_action_gt, axis=0)
            # helper = TSNEHelper(tsne_inputs)
            # helper.plot_tsne(f'action_gt_{epoch_idx:02d}_{batch_idx:05d}')

            # b_len = len(self.cache_t_ca)
            # ca_len = len(self.cache_t_ca[0])
            # for ca_idx in range(ca_len):
            #     tsne_t_inputs, tsne_s_inputs = [], []
            #     for b_idx in range(b_len):
            #         tsne_t_inputs.append(self.cache_t_ca[b_idx][ca_idx])  # each is (B,10*512)
            #         tsne_s_inputs.append(self.cache_s_ca[b_idx][ca_idx])
            #     tsne_inputs = np.concatenate(tsne_t_inputs + tsne_s_inputs, axis=0)
            #     helper = TSNEHelper(tsne_inputs)
            #     helper.plot_tsne(f'ca_layer{ca_idx}_{epoch_idx:02d}_{batch_idx:05d}')

            self.cache_t_vis1 = []
            self.cache_s_vis1 = []
            self.cache_t_vis2 = []
            self.cache_s_vis2 = []
            self.cache_t_action_gt = []
            self.cache_s_action_gt = []
            self.cache_t_pred_a0 = []
            self.cache_s_pred_a0 = []
            self.cache_t_k = []
            self.cache_s_k = []
            self.cache_t_v = []
            self.cache_s_v = []
            self.cache_t_q = []
            self.cache_s_q = []
            self.cache_t_emb = []
            self.cache_s_emb = []

        if self.use_da_vis1:
            da_loss_dict = self.da_vis1_loss.forward(
                [t_feat_for_da_vis1.clone().detach()],  # avoid grad of G_target
                [s_feat_for_da_vis1.detach()],  # avoid grad of G_source
                is_discriminator_batch=True,
            )
            da_d_1_loss = da_loss_dict['loss']
            w_dist = da_loss_dict['w_dist']
            gp = da_loss_dict['gp']  # just for log
            losses['da_d1_loss'] += da_d_1_loss / 1
            losses['w_dist_1'] += w_dist
            losses['gp_1'] += gp

            d_vis1_opt.zero_grad()
            self.manual_backward(losses['da_d1_loss'], retain_graph=False)  # no need to retrain graph
            d_vis1_opt.step()
            d_vis1_sch.step()

        if self.use_da_vis2:
            da_2_loss_dict = self.da_vis2_loss.forward(
                [t_feat_for_da_vis2.clone().detach()],  # avoid grad of G_target
                [s_feat_for_da_vis2.detach()],  # avoid grad of G_source
                is_discriminator_batch=True,
            )
            da_d_2_loss = da_2_loss_dict['loss']
            w_dist = da_2_loss_dict['w_dist']
            gp = da_2_loss_dict['gp']  # just for log
            losses['da_d2_loss'] += da_d_2_loss / 1
            losses['w_dist_2'] += w_dist
            losses['gp_2'] += gp

            d_vis2_opt.zero_grad()
            self.manual_backward(losses['da_d2_loss'], retain_graph=False)  # no need to retrain graph
            d_vis2_opt.step()
            d_vis2_sch.step()

        if self.use_da_act:
            da_act_loss_dict = self.da_act_loss.forward(
                [x.clone().detach() for x in t_feat_for_da_act],  # avoid grad of G_target
                [x.clone().detach() for x in s_feat_for_da_act],  # avoid grad of G_source
                is_discriminator_batch=True,
                sigmas=common_sigma_emb,
                conditions=[t_feat_for_da_enc.clone().detach(),
                            s_feat_for_da_enc.clone().detach()],  # concat with sigmas_emb
            )
            da_d_act_loss = da_act_loss_dict['loss']
            w_dist = da_act_loss_dict['w_dist']
            gp = da_act_loss_dict['gp']  # just for log
            losses['da_d_act_loss'] += da_d_act_loss / 1
            losses['w_dist_act'] += w_dist
            losses['gp_act'] += gp

            d_act_opt.zero_grad()
            self.manual_backward(losses['da_d_act_loss'], retain_graph=False)  # no need to retrain graph
            d_act_opt.step()
            d_act_sch.step()

        ''' 2. Update generator '''
        losses['cont_loss'] += cont_loss / t_batch_len  # used
        losses['action_loss'] += action_loss / batch_len  # NOT used
        losses['img_gen_loss'] += img_gen_loss / t_batch_len  # used
        backward_loss = losses["img_gen_loss"] + losses['cont_loss'] + losses['action_loss']

        if self.use_da_act:
            da_act_loss_dict = self.da_act_loss.forward(
                t_feat_for_da_act,  # update G_target
                s_feat_for_da_act,  # avoid grad of G_source
                is_discriminator_batch=False,
                sigmas=common_sigma_emb,
                conditions=[t_feat_for_da_enc.clone().detach(),
                            s_feat_for_da_enc.clone().detach()],  # concat with sigmas_emb
            )
            da_g_act_loss = da_act_loss_dict['loss']
            gp = da_act_loss_dict['gp']  # just for log
            losses['da_g_act_loss'] += da_g_act_loss / 1

            g_act_opt.zero_grad()
            retain_graph = self.use_da_vis1 or self.use_da_vis2  # Keep backward graph for later modules
            if not self.debug_diff_loss:
                act_back_loss = losses['da_g_act_loss'] + losses['action_loss']
                self.manual_backward(act_back_loss, retain_graph=retain_graph)
            elif self.current_epoch >= 1 or batch_idx > 10:  # Only for debug
                self.manual_backward(backward_loss)
            g_act_opt.step()
            g_act_sch.step()

        if self.use_da_vis1:
            da_loss_dict = self.da_vis1_loss.forward(
                [t_feat_for_da_vis1],  # update G_target
                [s_feat_for_da_vis1],  # avoid grad of G_source
                is_discriminator_batch=False,
            )
            da_g1_loss = da_loss_dict['loss']
            gp = da_loss_dict['gp']  # just for log
            losses['da_g1_loss'] += da_g1_loss / 1

            backward_loss += losses['da_g1_loss']
            g_vis1_opt.zero_grad()

        if self.use_da_vis2:
            da_2_loss_dict = self.da_vis2_loss.forward(
                [t_feat_for_da_vis2],  # update G_target
                [s_feat_for_da_vis2],  # avoid grad of G_source
                is_discriminator_batch=False,
            )
            da_g_2_loss = da_2_loss_dict['loss']
            gp = da_2_loss_dict['gp']  # just for log
            losses['da_g2_loss'] += da_g_2_loss / 1

            backward_loss += losses['da_g2_loss']
            g_vis2_opt.zero_grad()

        losses['total_loss'] += backward_loss + losses['da_g_act_loss']
        if self.use_da_vis1 or self.use_da_vis2:
            self.manual_backward(backward_loss)  # backward vis1 and vis2 together

        # if batch_idx >= 110 or self.current_epoch >= 1:  # TODO: update after vis tsne
        if self.use_da_vis1:
            g_vis1_opt.step()
        if self.use_da_vis2:
            g_vis2_opt.step()
        g_vis1_sch.step()
        g_vis2_sch.step()

        if not self.automatic_optimization:
            self.on_before_zero_grad()
        # Log the metrics
        self._log_training_metrics(losses, total_bs)

        return total_loss

    @torch.no_grad()
    def validation_step(self, batch: Dict[str, Dict], batch_idx: int, dataloader_idx: int = 0) -> Dict[
        str, torch.Tensor]:  # type: ignore
        """
        Compute and log the validation losses and additional metrics.
        During the validation step, the diffusion model predicts the next action sequence given the current state

        Args:
            batch: Dictionary containing the batch data for each modality.
            batch_idx: Index of the batch. used for compatibility with pytorch lightning.
            dataloader_idx: Index of the dataloader. used for compatibility with pytorch lightning.

        Returns:
            Dictionary containing the sampled plans of plan recognition and plan proposal networks, as well as the
            episode indices.
        """
        output = {}
        val_total_act_loss_pp = torch.tensor(0.0).to(self.device)
        for self.modality_scope, dataset_batch in batch.items():
            if "source" in self.modality_scope:
                continue
            # Compute the required embeddings
            perceptual_emb, latent_goal, image_latent_goal = self.compute_input_embeddings(dataset_batch)

            # predict the next action sequence
            action_pred = self.denoise_actions(
                torch.zeros_like(latent_goal).to(latent_goal.device),
                perceptual_emb,
                latent_goal,
                inference=True,
            )
            # compute the mse action loss
            pred_loss = torch.nn.functional.mse_loss(action_pred, dataset_batch["actions"])
            latent_encoder_emb = self.model.inner_model.latent_encoder_emb
            val_total_act_loss_pp += pred_loss

            # next compute the image generation loss
            if not isinstance(self.gen_img, NoEncoder):
                rgb_static_goal = dataset_batch["rgb_obs"]['gen_static']
                rgb_gripper_goal = dataset_batch["rgb_obs"]['gen_gripper']
                img_gen_frame_diff = dataset_batch['future_frame_diff'] if "future_frame_diff" in dataset_batch else 3
                # combine both goal images
                rgb_pred_goal = torch.cat([rgb_static_goal, rgb_gripper_goal], dim=1)

                img_gen_embed = latent_encoder_emb

                img_gen_loss = self.compute_img_gen_loss(
                    img_gen_embed,
                    rgb_pred_goal,
                    store_img=False,
                    batch_idx=batch_idx,
                    img_gen_frame_diff=img_gen_frame_diff,
                )
            else:
                img_gen_loss = torch.tensor(0.0).to(self.device)

            self._log_validation_metrics(pred_loss, img_gen_loss, val_total_act_loss_pp)

            output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]
            output["validation_loss"] = val_total_act_loss_pp
        return output

    def compute_input_embeddings(self, dataset_batch, is_target: bool = True):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        # 1. extract the revelant visual observations
        latent_goal = None
        rgb_static_goal = dataset_batch["rgb_obs"]['rgb_static'][:, -1]
        rgb_static = dataset_batch["rgb_obs"]['rgb_static'][:, :-1]

        rgb_gripper = dataset_batch["rgb_obs"]['rgb_gripper'][:, :-1]
        modality = "vis"

        # 2. Compute the latent goal embedding for the visual goal
        if not isinstance(self.visual_goal, NoEncoder):
            latent_goal = self.visual_goal(rgb_static_goal).to(rgb_static.dtype)

        lang_text = dataset_batch["lang_text"] if "lang" in self.modality_scope else None

        # 3. we compute the language goal if the language modality is in the scope
        if "lang" in self.modality_scope:
            modality = "lang"
            image_latent_goal = latent_goal.to(rgb_static.dtype)
            if self.use_text_not_embedding:
                latent_goal = self.language_goal(dataset_batch["lang_text"]).to(rgb_static.dtype)
            else:
                latent_goal = self.language_goal(dataset_batch["lang"]).to(rgb_static.dtype)
        else:
            image_latent_goal = None

        perceptual_emb = self.embed_visual_obs(rgb_static, rgb_gripper, is_target)
        perceptual_emb['modality'] = modality
        return perceptual_emb, latent_goal, image_latent_goal

    def embed_visual_obs(self, rgb_static, rgb_gripper, is_target=True):
        # reshape rgb_static and rgb_gripper
        rgb_static = einops.rearrange(rgb_static, 'b t c h w -> (b t) c h w')
        rgb_gripper = einops.rearrange(rgb_gripper, 'b t c h w -> (b t) c h w')
        if is_target:
            static_resnet = self.static_resnet
            gripper_resnet = self.gripper_resnet
        else:
            static_resnet = self.source_static_resnet
            gripper_resnet = self.source_gripper_resnet

        static_tokens = static_resnet(rgb_static)
        gripper_tokens = gripper_resnet(rgb_gripper)
        static_tokens = einops.rearrange(static_tokens, 'b (t d) -> b t d', t=1)
        gripper_tokens = einops.rearrange(gripper_tokens, 'b (t d) -> b t d', t=1)
        token_seq = {
            'static': static_tokens,
            'gripper': gripper_tokens,
        }
        return token_seq

    def clip_extra_forward(self, perceptual_emb, latent_goal, actions, sigmas, noise):
        self.model.train()
        noised_input = actions + noise * append_dims(sigmas, actions.ndim)
        context = self.model.forward_context_only(perceptual_emb, noised_input, latent_goal, sigmas)
        return context

    def compute_img_gen_loss(self, latent_embeddings, goal_img, store_img=False, img_gen_frame_diff=3, batch_idx=0):
        """
        Compute the image generation loss based on the provided embeddings and dataset batch.
        """
        if len(goal_img.shape) == 5:
            goal_img = goal_img.squeeze(1)
            # the goal is not to reconstruct all the details but to get the general shape
        # 1. predict the future image patches
        img_gen_pred, mask, restore_idxs, visible_patches = self.gen_img(latent_embeddings, goal_img,
                                                                         img_gen_frame_diff)
        # 2. compute the loss
        img_gen_loss = self.gen_img.compute_loss(goal_img, img_gen_pred, mask, restore_idxs)
        if store_img:
            file_path = os.getcwd() + f'/img_gen_pred_{batch_idx}.png'
            self.gen_img.reconstruct_image(
                predictions=img_gen_pred,
                goal_images=goal_img,
                mask=mask,
                restore_idxs=restore_idxs,
                file_path=file_path,
            )
            try:
                self.logger.experiment.log({f"generated_img_{batch_idx}": wandb.Image(os.path.abspath(file_path))})
            except Exception as e:
                print(f"An error occurred while saving or logging image: {e}")
                # Optionally, you can log the error to wandb as well
                self.logger.experiment.log({"error": str(e)})

        return img_gen_loss

    def compute_contrastive_loss(self, perceptual_emb, latent_goal, image_latent_goal, dataset_batch, sigma, noise):
        """
        Compute the contrastive loss based on the provided embeddings and dataset batch.
        """
        if "lang" in self.modality_scope:
            latent_language_embed = self.model.inner_model.latent_encoder_emb

            latent_vis_embed = self.clip_extra_forward(
                perceptual_emb,
                image_latent_goal,
                dataset_batch["actions"],
                sigma,  # Assuming you don't need sigmas and noise here
                noise
            )
            latent_language_embed = self.clip_proj(latent_language_embed)
            latent_vis_embed = self.clip_proj(latent_vis_embed)

            is_distributed = self.trainer.global_rank >= 0 and dist.is_initialized()

            if is_distributed and self.use_distributed_clip:

                all_latent_vis_embed = self.all_gather(latent_vis_embed, sync_grads=True)
                all_latent_language_embed = self.all_gather(latent_language_embed, sync_grads=True)
                all_latent_language_embed = einops.rearrange(all_latent_language_embed, 'n b d -> (n b) d')
                all_latent_vis_embed = einops.rearrange(all_latent_vis_embed, 'n b d -> (n b) d')

            else:
                all_latent_vis_embed = latent_vis_embed
                all_latent_language_embed = latent_language_embed

            lang_text = dataset_batch["lang_text"] if "lang_text" in dataset_batch else None

            # Compute contrastive loss with gathered embeddings
            cont_loss_part = self.cont_loss(
                all_latent_vis_embed,
                all_latent_language_embed,
                mode=self.clip_loss_type,
                lang_text=lang_text
            )

            return cont_loss_part
        else:
            return torch.tensor(0.0).to(self.device)  # Return a zero tensor if "lang" is not in the modality scope

    def _log_training_metrics(self, log_dict, total_bs):
        """
        Log the training metrics.
        """
        for k, v in log_dict.items():
            self.log(f"train/{k}", v.clone().detach(),
                     on_step=True, on_epoch=False, sync_dist=True, batch_size=total_bs)
        # self.logger.log_metrics(log_dict, step=self.global_step)
        # self.log()

    def _log_validation_metrics(self, pred_loss, img_gen_loss, val_total_act_loss_pp):
        """
        Log the validation metrics.
        """
        self.log(f"val_act/{self.modality_scope}_act_loss_pp", pred_loss, sync_dist=True)
        self.log(
            "val_act/action_loss",
            val_total_act_loss_pp / len(self.trainer.datamodule.modalities),  # type:ignore
            sync_dist=True,
        )
        self.log(f"val_act/img_gen_loss_pp", img_gen_loss, sync_dist=True)

    def diffusion_loss(
            self,
            perceptual_emb: dict,
            latent_goal: torch.Tensor,
            actions: torch.Tensor,  # gt
            is_target: bool = True,
            is_da: bool = True,
            sigmas: torch.Tensor = None,
            noise: torch.Tensor = None,
    ):
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        bs = perceptual_emb['static'].shape[0]
        if bs != actions.shape[0]:
            actions = actions[:bs]
        if not is_target:  # source
            self.source_model.eval()
            sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device) if sigmas is None else sigmas
            noise = torch.randn_like(actions).to(self.device) if noise is None else noise
            loss, pred_a0 = self.source_model.loss(perceptual_emb, actions, latent_goal, noise, sigmas, is_da=is_da)
        else:
            self.model.train()
            # assert sigmas is not None
            # assert noise is not None
            sigmas = self.make_sample_density()(shape=(len(actions),), device=self.device).to(self.device) if sigmas is None else sigmas
            noise = torch.randn_like(actions).to(self.device) if noise is None else noise  # TODO: do not re-rand
            loss, pred_a0 = self.model.loss(perceptual_emb, actions, latent_goal, noise, sigmas, is_da=is_da)
        return loss, sigmas, noise, pred_a0

    def denoise_actions(  # type: ignore
            self,
            latent_plan: torch.Tensor,
            perceptual_emb: torch.Tensor,
            latent_goal: torch.Tensor,
            inference: Optional[bool] = False,
            extra_args={}
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the next sequence of actions
        """
        if inference:
            sampling_steps = self.num_sampling_steps
        else:
            sampling_steps = 10
        self.model.eval()
        if len(latent_goal.shape) < len(
                perceptual_emb['static'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape):
            latent_goal = latent_goal.unsqueeze(1)  # .expand(-1, seq_len, -1)
        input_state = perceptual_emb
        sigmas = self.get_noise_schedule(sampling_steps, self.noise_scheduler)
        if len(latent_goal.shape) == 2:
            goal = einops.rearrange(goal, 'b d -> 1 b d')

        x = torch.randn((len(latent_goal), self.act_window_size, 7), device=self.device) * self.sigma_max

        actions = self.sample_loop(sigmas, x, input_state, latent_goal, latent_plan, self.sampler_type, extra_args)

        return actions

    def make_sample_density(self, is_da=False):
        """
        Generate a sample density function based on the desired type for training the model
        We mostly use log-logistic as it has no additional hyperparameters to tune.
        """
        sd_config = []
        if self.sigma_sample_density_type == 'lognormal':
            loc = self.sigma_sample_density_mean  # if 'mean' in sd_config else sd_config['loc']
            scale = self.sigma_sample_density_std  # if 'std' in sd_config else sd_config['scale']
            return partial(utils.rand_log_normal, loc=loc, scale=scale)

        if self.sigma_sample_density_type == 'loglogistic':
            loc = sd_config['loc'] if 'loc' in sd_config else math.log(self.sigma_data)
            scale = sd_config['scale'] if 'scale' in sd_config else 0.5
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            min_value = max_value if is_da else min_value
            return partial(utils.rand_log_logistic, loc=loc, scale=scale, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'loguniform':
            min_value = sd_config['min_value'] if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_log_uniform, min_value=min_value, max_value=max_value)

        if self.sigma_sample_density_type == 'uniform':
            return partial(utils.rand_uniform, min_value=self.sigma_min, max_value=self.sigma_max)

        if self.sigma_sample_density_type == 'v-diffusion':
            min_value = self.min_value if 'min_value' in sd_config else self.sigma_min
            max_value = sd_config['max_value'] if 'max_value' in sd_config else self.sigma_max
            return partial(utils.rand_v_diffusion, sigma_data=self.sigma_data, min_value=min_value, max_value=max_value)
        if self.sigma_sample_density_type == 'discrete':
            sigmas = self.get_noise_schedule(self.num_sampling_steps * 1e5, 'exponential')
            return partial(utils.rand_discrete, values=sigmas)
        if self.sigma_sample_density_type == 'split-lognormal':
            loc = sd_config['mean'] if 'mean' in sd_config else sd_config['loc']
            scale_1 = sd_config['std_1'] if 'std_1' in sd_config else sd_config['scale_1']
            scale_2 = sd_config['std_2'] if 'std_2' in sd_config else sd_config['scale_2']
            return partial(utils.rand_split_log_normal, loc=loc, scale_1=scale_1, scale_2=scale_2)
        else:
            raise ValueError('Unknown sample density type')

    def sample_loop(
            self,
            sigmas,
            x_t: torch.Tensor,
            state: torch.Tensor,
            goal: torch.Tensor,
            latent_plan: torch.Tensor,
            sampler_type: str,
            extra_args={},
    ):
        """
        Main method to generate samples depending on the chosen sampler type. DDIM is the default as it works well in all settings.
        """
        s_churn = extra_args['s_churn'] if 's_churn' in extra_args else 0
        s_min = extra_args['s_min'] if 's_min' in extra_args else 0
        use_scaler = extra_args['use_scaler'] if 'use_scaler' in extra_args else False
        keys = ['s_churn', 'keep_last_actions']
        if bool(extra_args):
            reduced_args = {x: extra_args[x] for x in keys}
        else:
            reduced_args = {}
        if use_scaler:
            scaler = self.scaler
        else:
            scaler = None
        # ODE deterministic
        if sampler_type == 'lms':
            x_0 = sample_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True, extra_args=reduced_args)
        # ODE deterministic can be made stochastic by S_churn != 0
        elif sampler_type == 'heun':
            x_0 = sample_heun(self.model, state, x_t, goal, sigmas, scaler=scaler, s_churn=s_churn, s_tmin=s_min,
                              disable=True)
        # ODE deterministic
        elif sampler_type == 'euler':
            x_0 = sample_euler(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # SDE stochastic
        elif sampler_type == 'ancestral':
            x_0 = sample_dpm_2_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
            # SDE stochastic: combines an ODE euler step with an stochastic noise correcting step
        elif sampler_type == 'euler_ancestral':
            x_0 = sample_euler_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm':
            x_0 = sample_dpm_2(self.model, state, x_t, goal, sigmas, disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_adaptive':
            x_0 = sample_dpm_adaptive(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), disable=True)
        # ODE deterministic
        elif sampler_type == 'dpm_fast':
            x_0 = sample_dpm_fast(self.model, state, x_t, goal, sigmas[-2].item(), sigmas[0].item(), len(sigmas),
                                  disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2s_ancestral':
            x_0 = sample_dpmpp_2s_ancestral(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        # 2nd order solver
        elif sampler_type == 'dpmpp_2m':
            x_0 = sample_dpmpp_2m(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2m_sde':
            x_0 = sample_dpmpp_sde(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'ddim':
            x_0 = sample_ddim(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2s':
            x_0 = sample_dpmpp_2s(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        elif sampler_type == 'dpmpp_2_with_lms':
            x_0 = sample_dpmpp_2_with_lms(self.model, state, x_t, goal, sigmas, scaler=scaler, disable=True)
        else:
            raise ValueError('desired sampler type not found!')
        return x_0

    def get_noise_schedule(self, n_sampling_steps, noise_schedule_type):
        """
        Get the noise schedule for the sampling steps. Describes the distribution over the noise levels from sigma_min to sigma_max.
        """
        if noise_schedule_type == 'karras':
            return get_sigmas_karras(n_sampling_steps, self.sigma_min, self.sigma_max, 7,
                                     self.device)  # rho=7 is the default from EDM karras
        elif noise_schedule_type == 'exponential':
            return get_sigmas_exponential(n_sampling_steps, self.sigma_min, self.sigma_max, self.device)
        elif noise_schedule_type == 'vp':
            return get_sigmas_vp(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 'linear':
            return get_sigmas_linear(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'cosine_beta':
            return cosine_beta_schedule(n_sampling_steps, device=self.device)
        elif noise_schedule_type == 've':
            return get_sigmas_ve(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        elif noise_schedule_type == 'iddpm':
            return get_iddpm_sigmas(n_sampling_steps, self.sigma_min, self.sigma_max, device=self.device)
        raise ValueError('Unknown noise schedule type')

    def reset(self):
        """
        Call this at the beginning of a new rollout when doing inference.
        """
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0

    def forward(self, obs, goal):
        """
        Method for doing inference with the model.
        """
        if 'lang' in goal:
            modality = 'lang'
            if self.use_text_not_embedding:
                # print(goal.keys())
                latent_goal = self.language_goal(goal["lang_text"])
                latent_goal = latent_goal.to(torch.float32)
            else:
                latent_goal = self.language_goal(goal["lang"]).unsqueeze(0).to(torch.float32).to(
                    obs["rgb_obs"]['rgb_static'].device)
        else:
            modality = 'vis'
            if self.use_delta_goal:
                perceptual_goal_emb = self.visual_goal(obs["rgb_obs"]['rgb_static'].squeeze(0))
            else:
                perceptual_goal_emb = self.visual_goal(obs["rgb_obs"]['rgb_static'][:, -1]).unsqueeze(1)  # [:, -1])

            latent_goal = perceptual_goal_emb

        rgb_static = obs["rgb_obs"]['rgb_static']
        rgb_gripper = obs["rgb_obs"]['rgb_gripper']

        perceptual_emb = self.embed_visual_obs(rgb_static, rgb_gripper)
        perceptual_emb['modality'] = modality

        act_seq = self.denoise_actions(
            torch.zeros_like(latent_goal).to(latent_goal.device),
            perceptual_emb,
            latent_goal,
            inference=True,
        )
        return act_seq

    def step(self, obs, goal):
        """
        Do one step of inference with the model. THis method handles the action chunking case.
        Our model is trained to predict a sequence of actions.
        We only compute the sequence once every self.multistep steps.

        Args:
            obs (dict): Observation from environment.
            goal (dict): Goal as visual observation or embedded language instruction.

        Returns:
            Predicted action.
        """
        if self.rollout_step_counter % self.multistep == 0:
            pred_action_seq = self(obs, goal)

            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]
        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')
        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0

        return current_action

    def on_train_start(self) -> None:

        self.source_model.to(dtype=self.dtype)
        self.source_static_resnet.to(dtype=self.dtype)
        self.source_gripper_resnet.to(dtype=self.dtype)
        self.model.to(dtype=self.dtype)
        self.static_resnet.to(dtype=self.dtype)
        self.gripper_resnet.to(dtype=self.dtype)
        self.language_goal.to(dtype=self.dtype)
        self.visual_goal.to(dtype=self.dtype)
        self.gen_img.to(dtype=self.dtype)

        for idx, callback in enumerate(self.trainer.callbacks):
            if isinstance(callback, EMA):
                self.ema_callback_idx = idx
                break

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def clip_auxiliary_loss(self, image_features, lang_features, mode='symmetric', lang_text=None):
        # Normalize the features
        image_features = F.normalize(image_features, dim=-1)
        lang_features = F.normalize(lang_features, dim=-1)
        logit_scale = self.logit_scale.exp()

        # Compute the cosine similarity
        similarity_matrix = logit_scale * image_features @ lang_features.t()

        # InfoNCE loss
        labels = torch.arange(similarity_matrix.shape[0], device=image_features.device)
        infonce_loss = F.cross_entropy(similarity_matrix, labels)

        if mode == 'symmetric':
            similarity_matrix_lang_img = logit_scale * lang_features @ image_features.t()
            # similarity_matrix_lang_img.masked_fill_(~unique_mask, float('-inf'))
            infonce_loss_lang_img = F.cross_entropy(similarity_matrix_lang_img, labels)
            infonce_loss = (infonce_loss + infonce_loss_lang_img) / 2
        elif mode == 'img_to_text':
            pass  # already computed above
        elif mode == 'text_to_img':
            similarity_matrix = similarity_matrix.t()  # transpose for text-to-image
            infonce_loss = F.cross_entropy(similarity_matrix, labels)
        else:
            raise ValueError("Invalid mode. Expected one of: 'symmetric', 'img_to_text', 'text_to_img'.")
        return infonce_loss

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_train_epoch_end(self, unused: Optional = None) -> None:  # type: ignore
        logger.info(f"Finished training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_end(self) -> None:
        logger.info(f"Finished validation epoch {self.current_epoch}")

    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)

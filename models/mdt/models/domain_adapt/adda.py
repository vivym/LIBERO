"""
Implements ADDA:
Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
Code referred from: https://github.com/corenel/pytorch-adda
"""
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm, trange


class ADDALoss(nn.Module):
    def __init__(self, in_dim: int = 3 * 512):
        super(ADDALoss, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),  # only one encoder
            nn.GELU(),
            nn.Linear(in_dim // 2, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 2),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, target_feat, source_feat=None, is_discriminator_batch: bool = True, gt_labels=None,):
        if source_feat is None:
            assert is_discriminator_batch, "source_feat should be given when is_discriminator_batch=True"
            source_feat = target_feat
        if source_feat.shape[0] > target_feat.shape[0]:
            source_feat = source_feat[:target_feat.shape[0]]
        elif target_feat.shape[0] > source_feat.shape[0]:
            target_feat = target_feat[:source_feat.shape[0]]
        bs = source_feat.shape[0]
        source_feat = source_feat.view(bs, -1)
        target_feat = target_feat.view(bs, -1)
        device = source_feat.device

        if is_discriminator_batch:
            in_feat = torch.cat([source_feat, target_feat])
            if gt_labels is None:
                gt_labels = torch.cat([torch.ones(source_feat.shape[0], device=device),
                                       torch.zeros(target_feat.shape[0], device=device)])
        else:
            in_feat = target_feat
            if gt_labels is None:
                gt_labels = torch.ones(source_feat.shape[0], device=device)  # flipped labels
        preds = self.discriminator(in_feat)
        loss = self.criterion(preds, gt_labels.long())
        return {
            'loss': loss,
            'w_dist': 0.,
            'gp': 0.,
        }

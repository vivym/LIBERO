import torch
import torch.nn as nn
from torch.autograd import Variable
import time as t
import os
from torchvision import utils
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()

    def forward(self, t_feat_list, s_feat_list, t_cond_list, s_cond_list):
        for i in range(len(t_feat_list)):
            t_feat = t_feat_list[i]  # (B,D)
            s_feat = s_feat_list[i]
            t_cond = t_cond_list[i]  # (B,C)
            s_cond = s_cond_list[i]



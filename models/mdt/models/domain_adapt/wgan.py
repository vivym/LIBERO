import einops
import torch
import torch.nn as nn
from sympy import discriminant
from torch.autograd import Variable
import time as t
import os
from typing import List, Tuple
from torchvision import utils
import torch.nn.functional as F


class Discriminator(torch.nn.Module):
    def __init__(self, channels: int = 3 * 384):
        super(Discriminator, self).__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        # self.main_module = nn.Sequential(
        #     # Image (Cx32x32)
        #     nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(num_features=256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     # State (256x16x16)
        #     nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(num_features=512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #
        #     # State (512x8x8)
        #     nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm2d(num_features=1024),
        #     nn.LeakyReLU(0.2, inplace=True))
        #     # output of main module --> State (1024x4x4)
        self.main_module = nn.Sequential(
            nn.Linear(in_features=channels, out_features=384, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=384, out_features=384, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=384, out_features=256, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # self.output = nn.Sequential(
        #     # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
        #     nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding=0))
        self.output = nn.Linear(in_features=256, out_features=1, bias=False)

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)


class WGAN_CP(torch.nn.Module):
    """
    Code refers to: https://github.dev/aadhithya/gan-zoo-pytorch/blob/master/models/wgan_gp.py
    """
    def __init__(self):
        super(WGAN_CP, self).__init__()
        self.discriminator = Discriminator()

    def forward(self, target_feat, source_feat=None, is_discriminator_batch: bool = True, gt_labels=None,):
        if source_feat is None:
            assert is_discriminator_batch, "source_feat should be given when is_discriminator_batch=True"
            source_feat = target_feat
        bs = source_feat.shape[0]
        source_feat = source_feat.view(bs, -1)
        target_feat = target_feat.view(bs, -1)
        device = source_feat.device

        if is_discriminator_batch:
            d_loss_real = self.discriminator(source_feat)
            d_loss_real = d_loss_real.mean(0).view(1)
            d_loss_fake = self.discriminator(target_feat)
            d_loss_fake = d_loss_fake.mean(0).view(1)
            loss = d_loss_real - d_loss_fake
        else:
            g_loss = self.discriminator(target_feat)
            g_loss = g_loss.mean().mean(0).view(1)
            loss = g_loss
        return loss


# StyleGAN2
def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1., bias = True):
        super(EqualLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class StyleVectorizer(nn.Module):
    def __init__(self, in_dim, out_dim, depth, lr_mul = 0.1):
        super(StyleVectorizer, self).__init__()

        layers = [
            # torch.nn.LayerNorm(in_dim, eps=1e-6),
            nn.Linear(in_dim, out_dim),  # reduce dim
            leaky_relu(),
        ]
        for i in range(depth - 1):
            layers.extend([EqualLinear(out_dim, out_dim, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x = F.normalize(x, dim=1)
        return self.net(x)
#*********************************************


class Discriminator1dStyleGAN(torch.nn.Module):
    def __init__(self, in_dim: int, reduce_scale: int, dropout=0.2):
        super(Discriminator1dStyleGAN, self).__init__()
        self.style_mlp = StyleVectorizer(in_dim, in_dim // reduce_scale, depth=2, lr_mul=1)
        self.dropout = nn.Dropout(dropout)
        self.logit_out = nn.Linear(in_dim // reduce_scale, 1)
    def forward(self, x):
        return self.logit_out(self.dropout(self.style_mlp(x)))


class AdaLNZero(nn.Module):
    def __init__(self, hidden_size, in_dim=512):
        super(AdaLNZero, self).__init__()
        self.parts = 3
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, self.parts * hidden_size, bias=True)
        )
    def init_weight(self):
        # Initialize weights and biases to zero
        weight_shape = list(self.modulation[1].weight.data.shape)
        weight_shape[0] = weight_shape[0] // self.parts
        # self.modulation[1].weight = torch.nn.Parameter(torch.cat(
        #     [torch.zeros(weight_shape) for _ in range(self.parts)]
        # ))  # shift=0,scale=0,gate=0
        nn.init.normal_(self.modulation[1].weight, std=0.02)
        nn.init.zeros_(self.modulation[1].bias)
    def forward(self, c):
        return self.modulation(c).chunk(self.parts, dim=-1)  # shift, scale, gate


class Discriminator1d(torch.nn.Module):
    def __init__(self, in_dim: int, inner_dim=64, dropout=0.2, use_ada=False, use_cond_dist=False,
                 ndim: int = 2, time_dim: int = 10,
                 ):
        super(Discriminator1d, self).__init__()
        down_scale = min(4, in_dim // 64)
        stem_in_dim = 1 if ndim == 2 else time_dim  # (B,1,D) or (B,T,D)
        self.stem = nn.Sequential(
            nn.Conv1d(stem_in_dim, inner_dim, 4, down_scale, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(inner_dim, inner_dim * 2, 4, 4, 1, bias=False),
        )
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(inner_dim * 2, inner_dim * 4, 4, 4, 1, bias=False),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(inner_dim * 4, inner_dim * 8, 4, 4, 1, bias=False),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
            ),
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(inner_dim * 2),
            nn.BatchNorm1d(inner_dim * 4),
            nn.BatchNorm1d(inner_dim * 8),
        ])
        self.dropout = nn.Dropout(dropout)
        if in_dim < 256:
            self.logit_out = nn.Linear(inner_dim * 8 * 1, 1, bias=False)
        else:
            self.logit_out = nn.Linear(inner_dim * 8 * (in_dim // 256), 1, bias=False)

        self.use_ada = use_ada
        if use_ada:
            sigma_dim = 512 if not use_cond_dist else 512 * 2
            self.cond_mapping = nn.ModuleList([
                AdaLNZero(inner_dim * 2, in_dim=sigma_dim),
                AdaLNZero(inner_dim * 4, in_dim=sigma_dim),
                AdaLNZero(inner_dim * 8, in_dim=sigma_dim),
            ])

        self.init_weight()

    def forward(self, x, sigmas=None):  # x:(B,D), s:(B,512)
        if x.ndim == 2:  # (B,D)
            x = x.unsqueeze(1)  # (B,1,D)
        elif x.ndim == 3:  # (B,T,D)
            if sigmas is not None:
                sigmas = einops.repeat(sigmas, "b d -> (b t) d", t=x.shape[1])  # (B*T,D)
            x = x.reshape(-1, x.shape[-1])  # remove time dimension, (B*T,D)
            x = x.unsqueeze(1)  # (B*T,1,D)
        x = self.stem(x)

        for i in range(len(self.convs)):
            x = self.norms[i](x)
            if self.use_ada: assert sigmas is not None
            if sigmas is not None:
                c_shift, c_scale, c_gate = self.cond_mapping[i](sigmas)  # sigma:(B,emb_dim)
                c_gate = c_gate.unsqueeze(-1)
                x = (1 - c_gate) * x + c_gate * (c_shift.unsqueeze(-1) + x * (c_scale.unsqueeze(-1) + 1.))
            x = self.convs[i](x)

        x = x.reshape(x.size(0), -1)
        output = self.logit_out(self.dropout(x))
        return output

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.use_ada:
            for ada in self.cond_mapping:
                ada.init_weight()

    def calc_params(self):
        num_params = sum(p.numel() for p in self.parameters())
        ret_str = f"{num_params/1024/1024:.2f}M Params"
        return ret_str


class CondDistMapping(torch.nn.Module):
    def __init__(self, in_dim: int = 3 * 512, out_dim: int = 512):
        super(CondDistMapping, self).__init__()
        self.dist_emb = nn.Linear(in_dim, out_dim)

    def forward(self, t_cond, s_cond):  # t/s_cond is (B,768)
        dist = t_cond - s_cond
        dist_emb = self.dist_emb(dist)
        return dist_emb


class FiLMBlock(nn.Module):
    def __init__(self, in_dim, out_dim, use_ada: bool = False):
        super(FiLMBlock, self).__init__()
        self.use_ada = use_ada

        self.conv1 = nn.Conv1d(in_dim, out_dim, 4, 4, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv1d(out_dim, out_dim, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_dim)
        if self.use_ada:
            self.film = AdaLNZero(hidden_size=1)  # TODO: 1 or out_dim ?
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, sigma_emb):
        # conv1 > bn1 > relu1 > conv2 > bn2 > film > relu2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        if self.use_ada:
            # sigma:(B,emb_dim)->(B,emb_dim,1)
            c_shift, c_scale, c_gate = [c.unsqueeze(-1) for c in self.film(sigma_emb)]
        else:
            c_shift, c_scale, c_gate = 0., 0., 0.

        identity = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = (1 - c_gate) * x + c_gate * (c_shift + x * (c_scale + 1))
        x = self.relu2(x)
        return identity + x

    def init_weight(self):
        if self.use_ada:
            self.film.init_weight()


class DiscriminatorFiLM1d(nn.Module):
    def __init__(self, in_dim: int, inner_dim=64, dropout=0.2, use_ada=False, use_cond_dist=False):
        super(DiscriminatorFiLM1d, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, inner_dim, 4, 4, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True))
        self.blocks = nn.ModuleList([
            FiLMBlock(inner_dim, inner_dim * 2, use_ada=use_ada),
            FiLMBlock(inner_dim * 2, inner_dim * 4, use_ada=use_ada),
            FiLMBlock(inner_dim * 4, inner_dim * 8, use_ada=use_ada),
        ])
        self.dropout = nn.Dropout(dropout)
        self.logit_out = nn.Linear(inner_dim * 8 * (in_dim // 256), 1, bias=False)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for film_block in self.blocks:
            film_block.init_weight()

    def forward(self, x, sigma_emb=None):
        if x.ndim == 2:  # (B,D)
            x = x.unsqueeze(1)  # (B,1,D)
        x = self.stem(x)

        for block in self.blocks:
            x = block(x, sigma_emb)

        x = x.reshape(x.size(0), -1)
        output = self.logit_out(self.dropout(x))
        return output

    def calc_params(self):
        num_params = sum(p.numel() for p in self.parameters())
        ret_str = f"{num_params/1024/1024:.2f}M Params"
        return ret_str


class Discriminator2d(torch.nn.Module):
    def __init__(self, in_dim: int, inner_dim=64, dropout=0.2, use_ada=False, use_cond_dist=False,
                 ndim: int = 3, time_dim: int = 10,
                 ):
        super(Discriminator2d, self).__init__()
        down_scale = min(4, in_dim // 64)
        self.stem = nn.Sequential(
            nn.Conv2d(1, inner_dim, (3, 4), (1, down_scale), 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(inner_dim, inner_dim * 2, (3, 4), (2, 4), 1, bias=False),
        )
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(inner_dim * 2, inner_dim * 4, (3, 4), (2, 4), 1, bias=False),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(inner_dim * 4, inner_dim * 8, (1, 4), (1, 4), (0, 1), bias=False),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
            ),
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(inner_dim * 2),
            nn.BatchNorm2d(inner_dim * 4),
            nn.BatchNorm2d(inner_dim * 8),
        ])
        self.dropout = nn.Dropout(dropout)
        if in_dim < 256:
            self.logit_out = nn.Linear(inner_dim * 8 * 1, 1, bias=False)
        else:
            self.logit_out = nn.Linear(inner_dim * 8 * (in_dim // 256) * 3, 1, bias=False)

        self.use_ada = use_ada
        if use_ada:
            sigma_dim = 512 if not use_cond_dist else 512 * 2
            self.cond_mapping = nn.ModuleList([
                AdaLNZero(inner_dim * 2, in_dim=sigma_dim),
                AdaLNZero(inner_dim * 4, in_dim=sigma_dim),
                AdaLNZero(inner_dim * 8, in_dim=sigma_dim),
            ])

        self.init_weight()

    def forward(self, x, sigmas=None):  # x:(B,T,D), s:(B,512)
        if x.ndim == 3:  # (B,T,D)
            x = x.unsqueeze(1)  # (B,1,T,D)
        x = self.stem(x)

        for i in range(len(self.convs)):
            x = self.norms[i](x)
            if self.use_ada: assert sigmas is not None
            if sigmas is not None:
                c_shift, c_scale, c_gate = self.cond_mapping[i](sigmas)  # sigma:(B,emb_dim)
                c_shift, c_scale, c_gate = [c.unsqueeze(-1).unsqueeze(-1) for c in [c_shift, c_scale, c_gate]]
                x = (1 - c_gate) * x + c_gate * (c_shift + x * (c_scale + 1.))
            x = self.convs[i](x)

        x = x.reshape(x.size(0), -1)
        output = self.logit_out(self.dropout(x))
        return output

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.use_ada:
            for ada in self.cond_mapping:
                ada.init_weight()

    def calc_params(self):
        num_params = sum(p.numel() for p in self.parameters())
        ret_str = f"{num_params/1024/1024:.2f}M Params"
        return ret_str


class Discriminator2dSeqGAN(torch.nn.Module):
    def __init__(self, in_dim, reduce_scale, dropout_prob=0.2):
        super(Discriminator2dSeqGAN, self).__init__()
        filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        filter_dims = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100]
        assert len(filter_sizes) == len(filter_dims)
        inner_dim = in_dim // reduce_scale
        self.embed = nn.Linear(in_dim, inner_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, dim, (f_size, inner_dim)) for f_size, dim in zip(filter_sizes, filter_dims)
        ])
        self.highway = nn.Linear(sum(filter_dims), sum(filter_dims))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(sum(filter_dims), 1)

    def forward(self, x):
        """
        Inputs: x
            - x: (B,T,D)
        Outputs: out
            - out: (B,1)
        """
        emb = self.embed(x).unsqueeze(1)  # (B,1,T,D)
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * seq_len]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        out = torch.cat(pools, 1)  # batch_size * sum(num_filters)
        highway = self.highway(out)
        transform = F.sigmoid(highway)
        out = transform * F.relu(highway) + (1. - transform) * out  # sets C = 1 - T
        # out = F.log_softmax(self.fc(self.dropout(out)), dim=1)  # batch * num_classes
        out = self.fc(self.dropout(out))
        return out


class Discriminator3d(torch.nn.Module):
    def __init__(self, in_dim: int, inner_dim=64, dropout=0.2, use_ada=False, use_cond_dist=False,
                 ndim: int = 4, time_dim: int = 10,
                 ):
        super(Discriminator3d, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim, inner_dim, (1, 1), (1, 1), 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(inner_dim, inner_dim * 2, (3, 3), (2, 1), 1, bias=False),
        )
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(inner_dim * 2, inner_dim * 4, (3, 1), (2, 1), (1, 0), bias=False),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(inner_dim * 4, inner_dim * 8, (3, 3), (2, 2), 1, bias=False),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
            ),
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(inner_dim * 2),
            nn.BatchNorm2d(inner_dim * 4),
            nn.BatchNorm2d(inner_dim * 8),
        ])
        self.dropout = nn.Dropout(dropout)
        if in_dim < 256:
            self.logit_out = nn.Linear(inner_dim * 8 * 4, 1, bias=False)
        else:
            self.logit_out = nn.Linear(inner_dim * 8 * (in_dim // 256) * 3, 1, bias=False)

        self.use_ada = use_ada
        if use_ada:
            sigma_dim = 512 if not use_cond_dist else 512 * 2
            self.cond_mapping = nn.ModuleList([
                AdaLNZero(inner_dim * 2, in_dim=sigma_dim),
                AdaLNZero(inner_dim * 4, in_dim=sigma_dim),
                AdaLNZero(inner_dim * 8, in_dim=sigma_dim),
            ])

        self.init_weight()

    def forward(self, x, sigmas=None):  # x:(B,Nh,T,Tc), s:(B,512)
        if x.ndim == 3:  # (B,T,D)
            x = x.unsqueeze(1)  # (B,1,T,D)
        x = self.stem(x)
        for i in range(len(self.convs)):
            x = self.norms[i](x)
            if self.use_ada: assert sigmas is not None
            if sigmas is not None:
                c_shift, c_scale, c_gate = self.cond_mapping[i](sigmas)  # sigma:(B,emb_dim)
                c_shift, c_scale, c_gate = [c.unsqueeze(-1).unsqueeze(-1) for c in [c_shift, c_scale, c_gate]]
                x = (1 - c_gate) * x + c_gate * (c_shift + x * (c_scale + 1.))
            x = self.convs[i](x)
        x = x.reshape(x.size(0), -1)
        output = self.logit_out(self.dropout(x))
        return output

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if self.use_ada:
            for ada in self.cond_mapping:
                ada.init_weight()

    def calc_params(self):
        num_params = sum(p.numel() for p in self.parameters())
        ret_str = f"{num_params/1024/1024:.2f}M Params"
        return ret_str


from torch.autograd import grad
class WGAN_GP(torch.nn.Module):
    def __init__(self,
                 in_dim: str = "1536*6,",
                 in_ndim: str = "2*6,",
                 time_dim: int = 10,
                 inner_dim: int = 64,
                 gamma: float = 10,
                 num_layers: int = 1,
                 use_ada: bool = False,
                 use_cond_dist: bool = False,
                 ):
        super(WGAN_GP, self).__init__()
        self.num_layers = num_layers

        discriminators = []
        in_dims: List[int] = self.process_in_dim_str(in_dim)
        in_ndims: List[int] = self.process_in_dim_str(in_ndim)
        assert len(in_dims) == num_layers
        for l in range(self.num_layers):
            d_net = self.get_discriminators(in_ndims[l], in_dim=in_dims[l], inner_dim=inner_dim,
                                            use_ada=use_ada, use_cond_dist=use_cond_dist,
                                            time_dim=time_dim)
            discriminators.append(d_net)
        self.discriminators = nn.ModuleList(discriminators)

        self.gamma = gamma
        self.wd_clf = 1

        self.use_cond_dist = use_cond_dist
        if self.use_cond_dist:
            self.cond_dist_mapping = CondDistMapping()  # in_dim=condition_dim, out_dim=sigmas_dim

        self.cache_wdists = [0. for _ in range(self.num_layers)]
        self.cache_gps = [0. for _ in range(self.num_layers)]

    @staticmethod
    def process_in_dim_str(in_dim: str):
        discriminators = in_dim.split(',')
        dims_list = []
        for discriminator in discriminators:
            if discriminator == '': continue
            dim, layer = [int(x) for x in discriminator.split('*')]
            dims_list.extend([dim] * layer)
        return dims_list

    @staticmethod
    def get_discriminators(ndim: int, **kwargs):
        if ndim == 2:  # (B,D)
            return Discriminator1d(ndim=ndim, **kwargs)
        elif ndim == 3:  # (B,T,D)
            return Discriminator2d(ndim=ndim, **kwargs)
        elif ndim == 4:  # (B,Nh,T,Tc)
            return Discriminator3d(ndim=ndim, **kwargs)
        else:
            raise NotImplementedError(f"{ndim} not supported!")

    def forward(self, target_feats, source_feats=None, is_discriminator_batch: bool = True,
                sigmas: torch.Tensor = None,
                conditions: List[torch.Tensor] = None,  # len=2, each is (B,3*512), order: `t_cond` and `s_cond`
            ):
        if source_feats is None:
            assert not is_discriminator_batch, "source_feat should be given when is_discriminator_batch=True"
            source_feats = target_feats

        assert len(target_feats) == len(source_feats) == self.num_layers

        # Calculate condition distance and map it into latent
        if self.use_cond_dist:
            assert len(conditions) == 2
            conditions = [cond.reshape(cond.shape[0], -1) for cond in conditions]  # (B,3,512)->(B,3*512)
            dist_emb = self.cond_dist_mapping(conditions[0], conditions[1])  # T and S, cat with sigmas_emb
            sigmas = torch.cat([sigmas, dist_emb], dim=-1)  # (B,512+512)

        loss = 0.
        for l in range(self.num_layers):
            layer_discriminator = self.discriminators[l]
            target_feat = target_feats[l]
            source_feat = source_feats[l]

            # Check shape
            if source_feat.shape[0] > target_feat.shape[0]:
                # source_feat = source_feat[:target_feat.shape[0]]  # use former features
                source_feat = source_feat[-target_feat.shape[0]:]  # use last features
                print('[Warning] target < source feat')
            elif target_feat.shape[0] > source_feat.shape[0]:
                # target_feat = target_feat[:source_feat.shape[0]]  # use former features
                target_feat = target_feat[-source_feat.shape[0]:]  # use last features
                print('[Warning] target > source feat')

            bs = source_feat.shape[0]
            device = source_feat.device

            if is_discriminator_batch:
                self.cache_gps[l] = gp = self.gradient_penalty(
                    layer_discriminator, source_feat, target_feat, device,
                    sigmas=sigmas,
                )
                d_source = layer_discriminator(source_feat.detach(), sigmas)            # avoid grad of G_source
                d_target = layer_discriminator(target_feat.clone().detach(), sigmas)    # avoid grad of G_target
                self.cache_wdists[l] = wasserstein_distance = d_source.mean() - d_target.mean()
                critic_cost = -wasserstein_distance + self.gamma * gp
                loss += critic_cost
            else:
                d_target = layer_discriminator(target_feat, sigmas)  # larger:more real
                d_target_neg_logit = -d_target.mean()
                loss += self.wd_clf * d_target_neg_logit

        loss = loss / self.num_layers

        return {
            'loss': loss,
            'w_dist': sum(self.cache_wdists) / self.num_layers,
            'gp': sum(self.cache_gps) / self.num_layers,
        }

    def gradient_penalty(self, critic, h_s, h_t, device, sigmas):
        # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
        alpha = torch.rand(h_s.size(0)).to(device)
        while alpha.ndim < h_s.ndim:
            alpha = alpha.unsqueeze(-1)
        differences = h_t - h_s
        interpolates = h_s + (alpha * differences)
        interpolates.requires_grad_(True)
        # interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

        preds = critic(interpolates, sigmas)
        gradients = grad(preds, interpolates,
                         grad_outputs=torch.ones_like(preds),
                         retain_graph=True, create_graph=True)[0]
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty

""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at https://github.com/google-research/vision_transformer
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020, Ross Wightman
"""
import math
import logging
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, checkpoint_seq
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch8_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    'vit_large_patch14_224': _cfg(url=''),
    'vit_huge_patch14_224': _cfg(url=''),
    'vit_giant_patch14_224': _cfg(url=''),
    'vit_gigantic_patch14_224': _cfg(url=''),

    'vit_base2_patch32_256': _cfg(url='', input_size=(3, 256, 256), crop_pct=0.95),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch8_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub_id='timm/vit_huge_patch14_224_in21k',
        num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_224_sam': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz'),
    'vit_base_patch16_224_sam': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz'),

    # DINO pretrained - https://arxiv.org/abs/2104.14294 (no classifier head, for fine-tune only)
    'vit_small_patch16_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_small_patch8_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch16_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),
    'vit_base_patch8_224_dino': _cfg(
        url='https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, num_classes=0),


    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),

    # experimental
    'vit_small_patch16_36x1_224': _cfg(url=''),
    'vit_small_patch16_18x2_224': _cfg(url=''),
    'vit_base_patch16_18x2_224': _cfg(url=''),
}




class Attention_LoRA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.rank = r

        self.lora_A_k = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_k = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.lora_A_v = nn.ModuleList([nn.Linear(dim, r, bias=False) for _ in range(n_tasks)])
        self.lora_B_v = nn.ModuleList([nn.Linear(r, dim, bias=False) for _ in range(n_tasks)])
        self.rank = r

        self.matrix = torch.zeros(dim ,dim)
        self.n_matrix = 0
        self.cur_matrix = torch.zeros(dim ,dim)
        self.n_cur_matrix = 0

    def init_param(self):
        for t in range(len(self.lora_A_k)):
            nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B_k[t].weight)
            nn.init.zeros_(self.lora_B_v[t].weight)

    def init_param_ada(self, t, r):
        self.lora_A_k[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_k[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)
        self.lora_A_v[t] = nn.Linear(self.dim, r, bias=False).to(self.qkv.weight.device)
        self.lora_B_v[t] = nn.Linear(r, self.dim, bias=False).to(self.qkv.weight.device)

        nn.init.kaiming_uniform_(self.lora_A_k[t].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_v[t].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_k[t].weight)
        nn.init.zeros_(self.lora_B_v[t].weight)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, task, register_hook=False, get_feat=False,get_cur_feat=False):
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            self.cur_matrix = (self.cur_matrix*self.n_cur_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_cur_matrix + x.shape[0]*x.shape[1])
            self.n_cur_matrix += x.shape[0]*x.shape[1]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # insert lora
        if task > -0.5:
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task+1)], dim=0).sum(dim=0)
            k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def get_matrix(self, task):
        matrix_k = torch.mm(self.lora_B_k[task].weight, self.lora_A_k[task].weight)
        matrix_v = torch.mm(self.lora_B_v[task].weight, self.lora_A_v[task].weight)
        return matrix_k, matrix_v
    
    def get_pre_matrix(self, task):
        with torch.no_grad():
            weight_k = torch.stack([torch.mm(self.lora_B_k[t].weight, self.lora_A_k[t].weight) for t in range(task)], dim=0).sum(dim=0)
            weight_v = torch.stack([torch.mm(self.lora_B_v[t].weight, self.lora_A_v[t].weight) for t in range(task)], dim=0).sum(dim=0)
        return weight_k, weight_v
    

def FFT_SHIFT(matrix):
        m_clone = matrix.clone()
        m,n = m_clone.shape
        m = int(m / 2)
        n = int(n / 2)

        for i in range(m):
            for j in range(n):
                m_clone[i][j] = matrix[m+i][n+j]
                m_clone[m+i][n+j] = matrix[i][j]
                m_clone[m+i][j] = matrix[i][j+n]
                m_clone[i][j+n] = matrix[m+i][j]
        return m_clone

class ParameterWrapper(nn.Module):
    def __init__(self, param):
        super(ParameterWrapper, self).__init__()
        self.weight = param
    
    def forward(self, x):
        # print('x, param', x.device(), self.pram.device())
        return F.linear(x, torch.diag(self.weight))

class Attention_LoRA_FFT(Attention_LoRA):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., r=64, n_tasks=10, n_frq=3000):
        super().__init__(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, r, n_tasks)

        self.n_frq = n_frq
        self.coef_k = nn.ParameterList([nn.Parameter(torch.randn(self.n_frq), requires_grad=True) for _ in range(n_tasks)]).to(self.qkv.weight.device)
        self.coef_v = nn.ParameterList([nn.Parameter(torch.randn(self.n_frq), requires_grad=True) for _ in range(n_tasks)]).to(self.qkv.weight.device)

        # self.S_trans_k = nn.ModuleList([ParameterWrapper(nn.Parameter(torch.Tensor([0.0 for _ in range(self.dim)]))) for _ in range(n_tasks)])
        # self.S_trans_v = nn.ModuleList([ParameterWrapper(nn.Parameter(torch.Tensor([0.0 for _ in range(self.dim)]))) for _ in range(n_tasks)])

        self.indices = [self.select_pos(t, self.dim).to(self.qkv.weight.device) for t in range(n_tasks)]
        b = self.dct_matrix(self.dim)
        self.bases = [b for _ in range(n_tasks)]
        # self.bases_trans = [torch.zeros(self.dim, self.dim) for _ in range(n_tasks)]

        self.MoE = False
        if self.MoE:
            self.gate = nn.Linear(self.dim, n_tasks)

        # 新增：用于累积当前任务的统计信息
        self.new_cur_matrix = None
        self.n_new_cur_matrix = 0

        # 新增，统计量
        self.fft_cur_matrix = None
        self.n_fft_cur_matrix = 0

        # 新增：安全频点掩码
        self.safe_mask = None
        
        # 新增：存储ATL相关的梯度
        self.atl_grad_k = None
        self.atl_grad_v = None
        
        # 新增：存储所有任务的统计信息
        self.saved_list = []

        # 新增：存储所有任务的列范数
        self.all_col_norms = {}
           
    def dct_matrix(self, n):
        """生成n×n的DCT变换矩阵"""
        matrix = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                if i == 0:
                    matrix[i, j] = math.sqrt(1/n)
                else:
                    matrix[i, j] = math.sqrt(2/n) * math.cos(math.pi * i * (2*j + 1) / (2*n))
        
        return matrix
    
    # ---------- saving at old task end ----------
    def save_task_statistics(self, B_ref=None, mode='full', rank=None):
        """
        Called at end of old task to save compact info for later influence computation.
        X_t: training data for task t in shape (M_t, d)
        B_ref: (d, k_r) or None (identity)
        B_t:   (d, k_t)  (task basis) - only needed if you want extra things
        mode: 'full'|'diag'|'lowrank'
        Returns a dict to save to disk.
        """
        X_t = self.new_cur_matrix
        device = X_t.device
        d = X_t.shape[1]
        if B_ref is None:
            # B_ref == I_d
            k_r = d
            # C = B_ref^T X^T X B_ref = X^T X  (d x d)
            # D = B_ref^T B_ref = I_d
            # Compute XtX = X_t^T @ X_t
            XtX = X_t.T @ X_t  # (d,d)
            if mode == 'full':
                self.saved_list.append({'C': XtX.cpu(), 'D': torch.eye(k_r).cpu(), 'mode':'full'})
            elif mode == 'diag':
                self.saved_list.append({'C_diag': torch.diag(XtX).cpu(), 'D_diag': torch.ones(k_r).cpu(), 'mode':'diag'})
            elif mode == 'lowrank':
                U, s, Vh = torch.linalg.svd(XtX, full_matrices=False)
                r = min(rank, s.shape[0])
                self.saved_list.append({'U': U[:, :r].cpu(), 's': s[:r].cpu(), 'Vh': Vh[:r,:].cpu(), 'mode':'lowrank'})
        else:
            # general B_ref: compute C = B_ref^T X^T X B_ref
            k_r = B_ref.shape[1]
            BR = B_ref.to(device)
            # Compute XBR = X @ BR  -> (M_t, k_r)
            XBR = X_t @ BR
            C = XBR.T @ XBR  # (k_r, k_r)
            D = BR.T @ BR    # (k_r, k_r)
            if mode == 'full':
                self.saved_list.append({'C': C.cpu(), 'D': D.cpu(), 'mode':'full'})
            elif mode == 'diag':
                self.saved_list.append({'C_diag': torch.diag(C).cpu(), 'D_diag': torch.diag(D).cpu(), 'mode':'diag'})
            elif mode == 'lowrank':
                U, s, Vh = torch.linalg.svd(C, full_matrices=False)
                r = min(rank, s.shape[0])
                self.saved_list.append({'U': U[:, :r].cpu(), 's': s[:r].cpu(), 'Vh': Vh[:r,:].cpu(), 'D': D.cpu(), 'mode':'lowrank'})
        
        # 重置统计
        self.new_cur_matrix.zero_()
        self.n_new_cur_matrix = 0

    def compute_S(self, task_id):
        assert self.new_cur_matrix is not None, "cur_matrix 为空，请先累积统计信息"
        # 计算当前任务范数
        B = self.bases[task_id].to(self.qkv.weight.device)  # (dim, k)
        U = self.new_cur_matrix @ B # (patch, k)
        alpha = (U * U).sum(dim=0)   # (k,)
        beta = (B * B).sum(dim=0)  # (k,)
        current_S2 = alpha.unsqueeze(1) * beta.unsqueeze(0)  # (k, k)  ; S2[p,q] = alpha[p] * beta[q]
        # 重置统计
        self.new_cur_matrix.zero_()
        self.n_new_cur_matrix = 0
        return current_S2

    # ---------- utils ----------
    def compute_R(self, task_id, B_ref=None, method='pinv'):
        # B_ref: (d, k_r) or None -> identity
        B_t = self.bases[task_id]
        # B_t:   (d, k_t)
        if B_ref is None:
            # Reference is identity: R = B_t (shape d x k_t), but we'll keep k_r = d
            return B_t  # shape (d, k_t)
        else:
            # Solve B_ref @ R = B_t  => R = pinv(B_ref) @ B_t  (shape k_r x k_t)
            if method == 'pinv':
                return torch.linalg.pinv(B_ref) @ B_t
            else:
                return torch.linalg.lstsq(B_ref, B_t).solution
    
    # ---------- in new task: compute influence & select ----------
    def compute_old_influence_for_current(self, R_curr, mode='full'):
        """
        R_curr: (k_r, k_c) if B_ref given; if B_ref None, R_curr = B_curr (d x k_c)
        saved_list: list of dicts returned from save_task_statistics for each old task.
        Returns aggregated influence tensor I_old (k_c, k_c) for the new task.
        mode: how saved stats were stored: 'full'|'diag'|'lowrank' (should match saved)
        """
        # R_curr: shape (k_r, k_c)  OR (d, k_c) if identity (we'll treat as (k_r, k_c))
        device = R_curr.device
        k_r, k_c = R_curr.shape
        # Precompute D or D_diag if available (D = B_ref^T B_ref). If B_ref is None, D = I
        # We'll compute per saved entry and sum/aggregate afterwards.
        # We'll compute for each old task matrix of size (k_c, ), namely
        # A_p = r_p^T C r_p  for all p -> vector a of length k_c
        # B_q = r_q^T D r_q  for all q -> vector b of length k_c
        # Then I_{pq}^2 = a_p * b_q  (outer product)
        # We'll aggregate a and b over tasks (e.g., max or sum)
        # Initialize aggregated a_agg and b_agg
        # We'll use sum-of-squares aggregation by default (user can choose)
        a_agg = torch.zeros(k_c, device=device)
        b_agg = torch.zeros(k_c, device=device)
        for s in self.saved_list:
            m = s['mode']
            if m == 'full':
                C = s['C'].to(device)   # (k_r, k_r)
                D = s['D'].to(device)   # (k_r, k_r)
                # compute a_p = diag(R^T C R)  where R is R_curr
                # R_curr: (k_r, k_c) => R^T C R => (k_c, k_c), diag gives a vector
                # but more efficient: compute (C R) -> (k_r, k_c) then elementwise (R * (C R)).sum(dim=0)
                CR = C @ R_curr   # (k_r, k_c)
                a = (R_curr * CR).sum(dim=0)  # (k_c,)
                DR = D @ R_curr
                b = (R_curr * DR).sum(dim=0)
            elif m == 'diag':
                Cdiag = s['C_diag'].to(device)  # (k_r,)
                Ddiag = s['D_diag'].to(device)
                # approximate r^T C r ≈ sum_i Cii * r_i^2
                a = (R_curr**2).T @ Cdiag   # (k_c,)
                b = (R_curr**2).T @ Ddiag
            elif m == 'lowrank':
                # C ≈ U S U^T
                # r^T C r = (U^T r)^T diag(s) (U^T r) = sum_j s_j * (u_j^T r)^2
                U = s['U'].to(device)   # (k_r, r)
                sval = s['s'].to(device) # (r,)
                Vh = s.get('Vh', None)
                # compute projections P = U^T R_curr  -> (r, k_c)
                P = U.T @ R_curr
                a = (sval.unsqueeze(1) * (P**2)).sum(dim=0)  # (k_c,)
                # for D we may or may not have; try D if provided else assume diag approx
                if 'D' in s:
                    D = s['D'].to(device)
                    DR = D @ R_curr
                    b = (R_curr * DR).sum(dim=0)
                else:
                    raise ValueError("lowrank saved without D")
            else:
                raise ValueError("unknown saved mode")
            # aggregate - here we do sum of squared influences (can change to max)
            a_agg += a  # sum
            b_agg += b
        # Now aggregated I_old^2 = outer(a_agg, b_agg)
        I_old2 = torch.outer(a_agg/len(self.saved_list), b_agg/len(self.saved_list))  # shape (k_c, k_c)
        # I_old = torch.sqrt(I_old2 + 1e-12)
        return I_old2
    
    def select_candidates(self, task_id, topK_ratio=0.3, lambda_reg=1e-6):
        """
        Combine current sensitivity S_curr (k_c,k_c) and old influence I_old (k_c,k_c)
        to produce a score and select topK positions to update.
        S_curr: current task sensitivity magnitude matrix (not squared)  (k_c,k_c)
        I_old:   computed aggregated old influence matrix (k_c,k_c)
        """
        topK = int(self.bases[task_id].shape[1] * self.bases[task_id].shape[1] * topK_ratio)
        S_curr = self.compute_S(task_id)
        R_curr = self.compute_R(task_id)
        I_old = self.compute_old_influence_for_current(R_curr)
        I_old = I_old.to(S_curr.device)

        score = S_curr / (lambda_reg + I_old)
        flat = score.reshape(-1)
        vals, idx = torch.topk(flat, k=topK, largest=True)
        k_c = S_curr.shape[0]
        ps = (idx // k_c).cpu().long()
        qs = (idx % k_c).cpu().long()

        # 创建安全频点掩码
        # 创建二维安全掩码矩阵
        self.safe_mask_2d = torch.zeros((k_c, k_c), device=self.coef_k[0].device)
        self.safe_mask_2d[ps, qs] = 1

        self.safe_mask = self.safe_mask_2d[self.indices[task_id][0], self.indices[task_id][1]]
        logging.info(f"Task {task_id}: Selected top-{self.safe_mask.sum()} safe frequency positions for ATL.")

    def save_col_norms(self, task_id):
        assert self.fft_cur_matrix is not None, "fft_cur_matrix 为空，请先累积统计信息"

        # 计算列范数
        B_x = self.fft_cur_matrix / max(1, self.n_fft_cur_matrix)
        col_norms = B_x.abs().norm(dim=0)  # shape [dim]

        # 保存当前任务的列范数
        self.all_col_norms[task_id] = col_norms.cpu()

        # 重置统计
        self.fft_cur_matrix.zero_()
        self.n_fft_cur_matrix = 0
    
    def compute_agg_col_norms(self):
        assert self.fft_cur_matrix is not None, "fft_cur_matrix 为空，请先累积统计信息"
        # 计算当前任务列范数
        B_x = self.fft_cur_matrix / max(1, self.n_fft_cur_matrix)
        current_col_norms = B_x.abs().norm(dim=0).cpu()  # shape [dim]
        # ===== 聚合策略 =====
        # 取所有任务的列范数的最大值（保守）
        # agg_col_norms = torch.stack(list(self.all_col_norms.values()), dim=0).max(dim=0)[0]
        # 取所有任务的列范数的平均值（折中）
        if not hasattr(self, "all_col_norms") or len(self.all_col_norms) == 0:
            return -1 * current_col_norms  # 如果没有历史任务，返回负当前列范数，表示优先选择当前最小的列
        agg_col_norms = torch.stack(list(self.all_col_norms.values()), dim=0).mean(dim=0)
        agg_col_norms = agg_col_norms - current_col_norms
        return agg_col_norms

    # 改进的方法：计算相对安全列
    def finalize_safe_mask(self, task_id, r=50):
        """
        计算相对安全列（取范数最小的 r 列）
        """
        assert task_id > 0, "任务ID应从1开始"
        
        agg_col_norms = self.compute_agg_col_norms()
        # 找到最小的 r 列
        safe_cols = torch.argsort(agg_col_norms)[:r]
        # 创建安全频点掩码
        # 创建二维安全掩码矩阵
        self.safe_mask_2d = torch.zeros((self.dim, self.dim), device=self.coef_k[0].device)
        # 对于每个安全列q，标记行q为安全
        for q in safe_cols:
            self.safe_mask_2d[q, ] = 1

        self.safe_mask = self.safe_mask_2d[self.indices[task_id][0], self.indices[task_id][1]]
        logging.info(f'[Task {task_id}] 选择 {r} 个最安全的行，对应 {int(self.safe_mask.sum().item())} 个安全频点')

        # 重置统计
        self.fft_cur_matrix.zero_()
        self.n_fft_cur_matrix = 0

    # 保存ATL相关的梯度
    def save_atl_grad(self, task_id):
        """保存ATL相关的梯度"""
        if self.coef_k[task_id].grad is not None:
            self.atl_grad_k = self.coef_k[task_id].grad.clone()
        if self.coef_v[task_id].grad is not None:
            self.atl_grad_v = self.coef_v[task_id].grad.clone()
    
    # 应用安全频点掩码到ATL梯度
    def apply_safe_mask_to_atl_grad(self, task_id):
        """应用安全频点掩码到ATL梯度"""
        if self.atl_grad_k is not None and self.safe_mask is not None:
            # 应用安全掩码到ATL梯度
            safe_atl_grad_k = self.atl_grad_k * self.safe_mask
            
            # 如果已经有其他梯度，需要合并
            if self.coef_k[task_id].grad is not None:
                # 减去原始的ATL梯度，加上安全版本的ATL梯度
                self.coef_k[task_id].grad = self.coef_k[task_id].grad - self.atl_grad_k + safe_atl_grad_k
            else:
                self.coef_k[task_id].grad = safe_atl_grad_k
                
        if self.atl_grad_v is not None and self.safe_mask is not None:
            # 应用安全掩码到ATL梯度
            safe_atl_grad_v = self.atl_grad_v * self.safe_mask
            
            # 如果已经有其他梯度，需要合并
            if self.coef_v[task_id].grad is not None:
                # 减去原始的ATL梯度，加上安全版本的ATL梯度
                self.coef_v[task_id].grad = self.coef_v[task_id].grad - self.atl_grad_v + safe_atl_grad_v
            else:
                self.coef_v[task_id].grad = safe_atl_grad_v
                
        # 清空ATL梯度
        self.atl_grad_k = None
        self.atl_grad_v = None

    def init_param(self):
        for t in range(len(self.coef_k)):
            nn.init.zeros_(self.coef_k[t])
        for t in range(len(self.coef_v)):
            nn.init.zeros_(self.coef_v[t])

    def add_task_parameters(self, task_id, n_frq):
        """为指定任务初始化参数"""
            
        # 初始化当前任务的参数
        device = self.qkv.weight.device
        coef_k = nn.Parameter(torch.zeros(n_frq, device=device), requires_grad=True)
        coef_v = nn.Parameter(torch.zeros(n_frq, device=device), requires_grad=True)
        
        # 添加到参数列表
        self.coef_k.append(coef_k)
        self.coef_v.append(coef_v)
        
    def select_pos(self, t, dim, n_frq=None, seed=777):
        # 如果没有指定 n_frq，使用默认 self.n_frq
        target_n_frq = n_frq if n_frq is not None else self.n_frq
        
        indices = torch.randperm(dim * dim, generator=torch.Generator().manual_seed(seed+t*10))[:target_n_frq]
        indices = torch.stack([indices // dim, indices % dim], dim=0)
        return indices

    # [新增] GCV 引导的核心：动态调整指定任务的容量
    def resize_task_capacity(self, task_id, new_n_frq):
        """
        根据 GCV 反馈，调整特定任务的频率分量数量。
        """
        if task_id >= len(self.coef_k):
            return # 还没初始化这个任务，忽略

        logging.info(f"Dynamic Resizing: Task {task_id} capacity {self.coef_k[task_id].shape[0]} -> {new_n_frq}")
        device = self.qkv.weight.device
        
        # 1. 重新生成对应数量的频率索引
        new_indices = self.select_pos(task_id, self.dim, n_frq=new_n_frq).to(device)
        self.indices[task_id] = new_indices
        
        # 2. 重新初始化参数 (Parameter)
        # 注意：这里是重置参数，因为结构变了。通常在任务开始训练前调用。
        self.coef_k[task_id] = nn.Parameter(torch.zeros(new_n_frq, device=device), requires_grad=True)
        self.coef_v[task_id] = nn.Parameter(torch.zeros(new_n_frq, device=device), requires_grad=True)

    # [新增] 频谱侦察与重置
    def probe_and_select_best_frequencies(self, task_id, gradient_matrix, k):
        """
        根据计算出的梯度矩阵 (Dim, Dim)，选择梯度最大的 k 个位置作为当前任务的 indices
        """
        device = self.qkv.weight.device
        
        # 1. 计算梯度显著性 (Saliency)
        # gradient_matrix: (Dim, Dim)
        saliency = gradient_matrix.abs()
        
        # 2. 选取 Top-K 索引
        # flatten 后取 topk
        flat_saliency = saliency.flatten()

        # [新增] 屏蔽旧任务已占用的频点 (Masking)
        if task_id > 0:
            # 遍历之前所有任务
            for t in range(task_id):
                prev_indices = self.indices[t] # Shape: (2, k_prev)
                if prev_indices is None: continue
                
                # 将二维坐标转换为扁平坐标: idx = row * dim + col
                flat_prev_indices = prev_indices[0] * self.dim + prev_indices[1]
                
                # 将这些位置的显著性设为负无穷，确保 topk 绝对不会选中它们
                flat_saliency[flat_prev_indices] = -float('inf')

        _, topk_indices_flat = torch.topk(flat_saliency, k)
        
        # 转回 (2, k) 坐标
        # row = idx // dim, col = idx % dim
        rows = topk_indices_flat // self.dim
        cols = topk_indices_flat % self.dim
        new_indices = torch.stack([rows, cols], dim=0).to(device)
        
        # 3. 更新 BiLoRA 的 indices
        self.indices[task_id] = new_indices
        
        # 4. 重置参数
        # 既然位置变了，参数必须重置为 0
        nn.init.zeros_(self.coef_k[task_id])
        nn.init.zeros_(self.coef_v[task_id])
        
        # 清除梯度，防止干扰后续训练
        if self.coef_k[task_id].grad is not None:
            self.coef_k[task_id].grad = None
        if self.coef_v[task_id].grad is not None:
            self.coef_v[task_id].grad = None
            
        return new_indices

    def get_delta_w_k(self, task, alpha=300):
        indices = self.indices[task]
        dim = self.bases[task].shape[1]
        # F = torch.zeros(dim, dim).to(self.qkv.weight.device)
        F = torch.zeros(self.dim, self.dim).to(self.qkv.weight.device)
        F[indices[0,:], indices[1,:]] =  self.coef_k[task]
        return torch.fft.ifft2(F, dim=(-2,-1)).real * alpha
        base = self.bases[task].to(self.qkv.weight.device)
        # base_trans = self.bases_trans[task].to(self.qkv.weight.device)
        return base @ F @ base.t()# + base_trans @ torch.diag(self.S_trans_k[task].weight) @ base_trans.t()
    
    def get_delta_w_v(self, task, alpha=300):
        indices = self.indices[task]
        dim = self.bases[task].shape[1]
        # F = torch.zeros(dim, dim).to(self.qkv.weight.device)
        F = torch.zeros(self.dim, self.dim).to(self.qkv.weight.device)
        F[indices[0,:], indices[1,:]] =  self.coef_v[task]
        return torch.fft.ifft2(F, dim=(-2,-1)).real * alpha
        base = self.bases[task].to(self.qkv.weight.device)
        # base_trans = self.bases_trans[task].to(self.qkv.weight.device)
        return base @ F @ base.t()# + base_trans @ torch.diag(self.S_trans_v[task].weight) @ base_trans.t()
    
    def get_pre_matrix(self, task):
        with torch.no_grad():
            weight_k = torch.stack([self.get_delta_w_k(t) for t in range(task)], dim=0).sum(dim=0)
            weight_v = torch.stack([self.get_delta_w_k(t) for t in range(task)], dim=0).sum(dim=0)
        return weight_k, weight_v

    def forward(self, x, task, register_hook=False, get_feat=False,get_cur_feat=False, get_cur_x=False, alpha=1.0):
        if get_feat:
            self.matrix = (self.matrix*self.n_matrix + torch.bmm(x.detach().permute(0, 2, 1), x.detach()).sum(dim=0).cpu())/(self.n_matrix + x.shape[0]*x.shape[1])
            self.n_matrix += x.shape[0]*x.shape[1]
        if get_cur_feat:
            # if self.new_cur_matrix is None:
            #     self.new_cur_matrix = x.mean(dim=0)
            # else:
            #     self.new_cur_matrix = (self.new_cur_matrix*self.n_new_cur_matrix + x.sum(dim=0))/(self.n_new_cur_matrix + x.shape[0])
            # self.n_new_cur_matrix += x.shape[0]

            # # 保存 fft_cur_matrix = FFT(x) 的统计信息
            # x_fft = x @ self.bases[task].to(x.device)
            # if self.fft_cur_matrix is None:
            #     self.fft_cur_matrix = x_fft.sum(dim=0)
            # else:
            #     self.fft_cur_matrix += x_fft.sum(dim=0)
            # self.n_fft_cur_matrix += x.shape[0]

            # 保存 fft_cur_matrix = FFT(x) 的统计信息
            with torch.no_grad():
                x_fft = torch.fft.fft2(x)
                if self.fft_cur_matrix is None:
                    self.fft_cur_matrix = x_fft.sum(dim=0)
                else:
                    self.fft_cur_matrix += x_fft.sum(dim=0)
                self.n_fft_cur_matrix += x.shape[0]

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        x0 = x.clone().detach()

        if self.MoE:
            gate_logits = self.gate(x)  # Shape: (batch_size, num_experts)

            mask = torch.zeros(self.n_tasks).to(self.qkv.weight.device)
            mask[:task] = 1
            gate_logits = gate_logits.masked_fill(mask == 0, float('-inf'))

            # Compute softmax over masked logits
            gate_values = F.softmax(gate_logits, dim=-1)  # Shape: (batch_size, num_experts)

            # Compute expert outputs
            expert_outputs = torch.stack([self.get_delta_w(t) for t in range(task+1)], dim=0).sum(dim=0)  # Shape: (batch_size, num_experts, expert_dim)

            # Weighted sum of expert outputs
            weighted_expert_output = torch.einsum('be,bed->bd', gate_values, expert_outputs)  # Shape: (batch_size, expert_dim)

        else:
        # insert lora   
            if get_cur_x:
                rate = 1
                self.cur_matrix = (self.cur_matrix * self.n_cur_matrix + torch.bmm(x0.detach().permute(0, 2, 1),
                                                                                x0.detach()).sum(dim=0).cpu()) / (
                                            rate * (self.n_cur_matrix + x0.shape[0] * x0.shape[1]))
                self.n_cur_matrix += x0.shape[0] * x0.shape[1]

                if task > 0:
                    weight_k_old = torch.stack([self.get_delta_w_k(t) for t in range(task)], dim=0).sum(dim=0)
                    weight_v_old = torch.stack([self.get_delta_w_v(t) for t in range(task)], dim=0).sum(dim=0)
                    k = k - alpha * F.linear(x, weight_k_old).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                    v = v - alpha * F.linear(x, weight_v_old).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            else:
                if task > -0.5:
                    weight_k = torch.stack([self.get_delta_w_k(t) for t in range(task+1)], dim=0).sum(dim=0)
                    weight_v = torch.stack([self.get_delta_w_v(t) for t in range(task+1)], dim=0).sum(dim=0)
                k = k + F.linear(x, weight_k).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
                v = v + F.linear(x, weight_v).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_tasks=10, r=64):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_LoRA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, n_tasks=n_tasks, r=r)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, task, register_hook=False, get_feat=False, get_cur_feat=False, get_cur_x=False, alpha=1):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), task, register_hook=register_hook, get_feat=get_feat, get_cur_feat=get_cur_feat, get_cur_x=get_cur_x, alpha=alpha)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
class Block_FFT(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_tasks=10, r=64):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, init_values, drop_path, act_layer, norm_layer, n_tasks, r)
        self.attn = Attention_LoRA_FFT(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, n_tasks=n_tasks, r=r)


class ParallelBlock(nn.Module):

    def __init__(
            self, dim, num_heads, num_parallel=2, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_parallel = num_parallel
        self.attns = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(num_parallel):
            self.attns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('attn', Attention_LoRA(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))
            self.ffns.append(nn.Sequential(OrderedDict([
                ('norm', norm_layer(dim)),
                ('mlp', Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)),
                ('ls', LayerScale(dim, init_values=init_values) if init_values else nn.Identity()),
                ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
            ])))

    def _forward_jit(self, x):
        x = x + torch.stack([attn(x) for attn in self.attns]).sum(dim=0)
        x = x + torch.stack([ffn(x) for ffn in self.ffns]).sum(dim=0)
        return x

    @torch.jit.ignore
    def _forward(self, x):
        x = x + sum(attn(x) for attn in self.attns)
        x = x + sum(ffn(x) for ffn in self.ffns)
        return x

    def forward(self, x):
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return self._forward_jit(x)
        else:
            return self._forward(x)


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, n_tasks=10, rank=64):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init: (str): weight init scheme
            init_values: (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_grow = nn.Parameter(torch.zeros(1, 5000, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_embed_grow = nn.Parameter(torch.zeros(1, num_patches + 1000, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,n_tasks=n_tasks,r=rank)
            for i in range(depth)])
        use_fc_norm = self.global_pool == 'avg'
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Representation layer. Used for original ViT models w/ in21k pretraining.
        self.representation_size = representation_size
        self.pre_logits = nn.Identity()
        if representation_size:
            self._reset_representation(representation_size)

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()
        self.out_dim = final_chs

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def _reset_representation(self, representation_size):
        self.representation_size = representation_size
        if self.representation_size:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(self.embed_dim, self.representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.pos_embed_grow, std=.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.cls_token_grow, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None, representation_size=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        if representation_size is not None:
            self._reset_representation(representation_size)
        final_chs = self.representation_size if self.representation_size else self.embed_dim
        self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_features_grow(self, x, class_num):
        x = self.patch_embed(x)
        # x = torch.cat((self.cls_token_grow[:, :class_num, :].expand(x.shape[0], -1, -1), self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        # x = self.pos_drop(x + self.pos_embed_grow[:, :self.patch_embed.num_patches+class_num, :])
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = torch.cat((self.cls_token_grow[:, :class_num*2, :].expand(x.shape[0], -1, -1), x), dim=1)

        # import pdb;pdb.set_trace()
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.pre_logits(x)
        return x if pre_logits else self.head(x)

    def forward(self, x, grow_flag=False, numcls=0):
        if not grow_flag:
            x = self.forward_features(x)
        else:
            x = self.forward_features_grow(x, numcls)

        if self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        return {
            'fmaps': [x],
            'features': x
        }


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_vit_moco(module: nn.Module, name: str = ''):
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    # pretrained_cfg = resolve_pretrained_cfg(variant, kwargs=kwargs)
    pretrained_cfg = resolve_pretrained_cfg(variant)
    default_num_classes = pretrained_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformer, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model


@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_384(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32)
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/32) at 384x384.
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_384(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base2_patch32_256(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32)
    # FIXME experiment
    """
    model_kwargs = dict(patch_size=32, embed_dim=896, depth=12, num_heads=14, **kwargs)
    model = _create_vision_transformer('vit_base2_patch32_256', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch8_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch8_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_384(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch14_224(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/14)
    """
    model_kwargs = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_huge_patch14_224(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_giant_patch14_224(pretrained=False, **kwargs):
    """ ViT-Giant model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_giant_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_gigantic_patch14_224(pretrained=False, **kwargs):
    """ ViT-Gigantic model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_kwargs = dict(patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_gigantic_patch14_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch8_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch8_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch32_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=32, embed_dim=1024, depth=24, num_heads=16, representation_size=1024, **kwargs)
    model = _create_vision_transformer('vit_large_patch32_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_large_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_vision_transformer('vit_large_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_sam(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
    """
    # NOTE original SAM weights release worked with representation_size=768
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_sam', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch32_224_sam(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/32) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
    """
    # NOTE original SAM weights release worked with representation_size=768
    model_kwargs = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch32_224_sam', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_224_dino(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224_dino', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch8_224_dino(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/8) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=8, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch8_224_dino', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_dino(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) /w DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_dino', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch8_224_dino(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/8) w/ DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch8_224_dino', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil_in21k(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil_in21k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_224_miil(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_36x1_224(pretrained=False, **kwargs):
    """ ViT-Base w/ LayerScale + 36 x 1 (36 block serial) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=36, num_heads=6, init_values=1e-5, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_36x1_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_small_patch16_18x2_224(pretrained=False, **kwargs):
    """ ViT-Small w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=384, depth=18, num_heads=6, init_values=1e-5, block_fn=ParallelBlock, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_18x2_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def vit_base_patch16_18x2_224(pretrained=False, **kwargs):
    """ ViT-Base w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=18, num_heads=12, init_values=1e-5, block_fn=ParallelBlock, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_18x2_224', pretrained=pretrained, **model_kwargs)
    return model
import torch
import torch.nn as nn
import copy

from models.vit_base import VisionTransformer, PatchEmbed,resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn, Block, Attention_LoRA_FFT
import torch.nn.functional as F

class Block_FFT(Block):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, n_tasks=10, r=64):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, init_values, drop_path, act_layer, norm_layer, n_tasks, r)
        self.attn = Attention_LoRA_FFT(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, n_tasks=n_tasks, r=r)

class ViT_lora_fft(VisionTransformer):
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='', init_values=None,
            embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block_FFT, n_tasks=10, rank=64):

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, n_tasks=n_tasks, rank=rank)


    def forward(self, x, task_id, register_blk=-1, get_feat=False, get_cur_feat=False, get_cur_x=False, alpha=1):
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        prompt_loss = torch.zeros((1,), requires_grad=True).to(x.device)
        for i, blk in enumerate(self.blocks):
            x = blk(x, task_id, register_blk==i, get_feat=get_feat, get_cur_feat=get_cur_feat, get_cur_x=get_cur_x, alpha=alpha)

        x = self.norm(x)
        
        return x, prompt_loss



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
        repr_size = None

    model = build_model_with_cfg(
        ViT_lora_fft, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model
# def _create_vision_transformer(variant, pretrained=False, **kwargs):
#     if kwargs.get('features_only', None):
#         raise RuntimeError('features_only not implemented for Vision Transformer models.')

#     pretrained_cfg = resolve_pretrained_cfg(variant)
#     default_num_classes = pretrained_cfg['num_classes']
#     num_classes = kwargs.get('num_classes', default_num_classes)
#     repr_size = kwargs.pop('representation_size', None)
#     if repr_size is not None and num_classes != default_num_classes:
#         repr_size = None

#     # 1. 强制设置 pretrained=False，先只初始化模型结构，不自动加载权重
#     model = build_model_with_cfg(
#         ViT_lora_fft, variant, pretrained=False, 
#         pretrained_cfg=pretrained_cfg,
#         representation_size=repr_size,
#         pretrained_filter_fn=checkpoint_filter_fn,
#         pretrained_custom_load='npz' in pretrained_cfg['url'],
#         **kwargs)

#     # 2. 如果需要预训练权重，手动调用 timm 的加载函数，并设置 strict=False
#     if pretrained:
#         from timm.models.helpers import load_pretrained
#         load_pretrained(
#             model,
#             pretrained_cfg=pretrained_cfg,
#             num_classes=num_classes,
#             filter_fn=checkpoint_filter_fn,
#             strict=False  # <--- 在这里设置非严格加载
#         )
#     return model
# def _create_vision_transformer(variant, pretrained=False, pretrained_path=None, **kwargs):
#     if kwargs.get('features_only', None):
#         raise RuntimeError('features_only not implemented for Vision Transformer models.')

#     pretrained_cfg = resolve_pretrained_cfg(variant)
#     default_num_classes = pretrained_cfg['num_classes']
#     num_classes = kwargs.get('num_classes', default_num_classes)
#     repr_size = kwargs.pop('representation_size', None)
#     if repr_size is not None and num_classes != default_num_classes:
#         repr_size = None

#     # 1. 初始化模型结构 (不自动从网络下载)
#     model = build_model_with_cfg(
#         ViT_lora_fft, variant, pretrained=False, 
#         pretrained_cfg=pretrained_cfg,
#         representation_size=repr_size,
#         pretrained_filter_fn=checkpoint_filter_fn,
#         pretrained_custom_load='npz' in pretrained_cfg['url'],
#         **kwargs)

#     # 2. 优先加载本地指定的 iBOT 权重路径
#     if pretrained_path is not None:
#         print(f"==> Loading local pretrained weights from: {pretrained_path}")
#         checkpoint = torch.load(pretrained_path, map_location='cpu')
        
#         # iBOT 的权重通常在 'state_dict' 或 'model' 键下面，需要根据你下载的文件结构调整
#         if 'model' in checkpoint:
#             state_dict = checkpoint['model']
#         elif 'state_dict' in checkpoint:
#             state_dict = checkpoint['state_dict']
#         else:
#             state_dict = checkpoint

#         # 使用 timm 原有的过滤函数处理（如 patch_embed 权重转换等）
#         state_dict = checkpoint_filter_fn(state_dict, model)
        
#         # 核心：使用 strict=False 加载，忽略 InfLoRA 的新参数
#         msg = model.load_state_dict(state_dict, strict=False)
#         print(f"==> Missing keys (these are InfLoRA params): {len(msg.missing_keys)}")
#         print(f"==> Unexpected keys: {msg.unexpected_keys}")

#     # 3. 如果没有本地路径但开启了 pretrained，则按原逻辑从网络下载
#     elif pretrained:
#         from timm.models.helpers import load_pretrained
#         load_pretrained(
#             model,
#             pretrained_cfg=pretrained_cfg,
#             num_classes=num_classes,
#             filter_fn=checkpoint_filter_fn,
#             strict=False 
#         )
#     return model


class SiNet(nn.Module):

    def __init__(self, args):
        super(SiNet, self).__init__()

        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, n_tasks=args["total_sessions"], rank=args["rank"])
        self.image_encoder =_create_vision_transformer('vit_base_patch16_224_in21k', pretrained=True, **model_kwargs)
        # self.image_encoder =_create_vision_transformer('vit_base_patch16_224_dino', pretrained=True, **model_kwargs)
        # ibot_path = "../fft_lora_cl/pretrained/ibot_vit_base_patch16_224.pth" 
        # self.image_encoder = _create_vision_transformer('vit_base_patch16_224_dino', pretrained=True, pretrained_path=ibot_path, **model_kwargs)
        # print(self.image_encoder)
        # exit()

        self.class_num = 1
        self.class_num = args["init_cls"]
        self.classifier_pool = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        self.classifier_pool_backup = nn.ModuleList([
            nn.Linear(args["embd_dim"], self.class_num, bias=True)
            for i in range(args["total_sessions"])
        ])

        # self.prompt_pool = CodaPrompt(args["embd_dim"], args["total_sessions"], args["prompt_param"])

        self.numtask = 0

    @property
    def feature_dim(self):
        return self.image_encoder.out_dim

    def extract_vector(self, image, task=None):
        if task == None:
            image_features, _ = self.image_encoder(image, self.numtask-1)
        else:
            image_features, _ = self.image_encoder(image, task)
        image_features = image_features[:,0,:]
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, image, get_feat=False, get_cur_feat=False, get_cur_x=False, alpha=1.0, fc_only=False):
        if fc_only:
            fc_outs = []
            for ti in range(self.numtask):
                fc_out = self.classifier_pool[ti](image)
                fc_outs.append(fc_out)
            return torch.cat(fc_outs, dim=1)

        logits = []
        image_features, prompt_loss = self.image_encoder(image, task_id=self.numtask-1, get_feat=get_feat, get_cur_feat=get_cur_feat, get_cur_x=get_cur_x, alpha=alpha)
        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)
        for prompts in [self.classifier_pool[self.numtask-1]]:
            logits.append(prompts(image_features))

        logits_all = []
        for prompts in self.classifier_pool_backup[:self.numtask]:
            logits_all.append(prompts(image_features))

        return {
            'logits': torch.cat(logits, dim=1),
            'features': image_features,
            'prompt_loss': prompt_loss,
            'logits_all': torch.cat(logits_all, dim=1)
        }

    def interface(self, image, task_id = None):
        image_features, _ = self.image_encoder(image, task_id=self.numtask-1 if task_id is None else task_id)

        image_features = image_features[:,0,:]
        image_features = image_features.view(image_features.size(0),-1)

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        return logits
    
    def interface1(self, image, task_ids):
        logits = []
        for index in range(len(task_ids)):
            image_features, _ = self.image_encoder(image[index:index+1], task_id=task_ids[index].item())
            image_features = image_features[:,0,:]
            image_features = image_features.view(image_features.size(0),-1)

            logits.append(self.classifier_pool_backup[task_ids[index].item()](image_features))

        logits = torch.cat(logits,0)
        return logits

    def interface2(self, image_features):

        logits = []
        for prompt in self.classifier_pool[:self.numtask]:
            logits.append(prompt(image_features))

        logits = torch.cat(logits,1)
        return logits

    def update_fc(self, nb_classes):
        self.numtask +=1

    def classifier_backup(self, task_id):
        self.classifier_pool_backup[task_id].load_state_dict(self.classifier_pool[task_id].state_dict())

    def classifier_recall(self):
        self.classifier_pool.load_state_dict(self.old_state_dict)

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class LearnableSparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity # e.g. 100 connections per row
        
        # 1. 初始化固定的索引 (Indices) - 冻结状态
        # 形状: [2, nnz], nnz = out_features * sparsity
        indices_list = []
        for i in range(out_features):
            # 每一行随机选 sparsity 个连接
            cols = torch.randperm(in_features)[:sparsity]
            rows = torch.full_like(cols, i)
            indices_list.append(torch.stack([rows, cols]))
        
        # [2, Total_Non_Zeros]
        self.indices = torch.cat(indices_list, dim=1).to(device)
        
        # 2. 初始化可学习的权重 (Values) - nn.Parameter
        # 使用高斯分布初始化
        self.values = nn.Parameter(torch.randn(self.indices.shape[1], device=device))
        
    def forward(self, x):
        """
        x: [Batch, In_Features] (Dense)
        Return: [Batch, Out_Features] (Dense)
        """
        # 实时构建稀疏矩阵，这样 PyTorch 的自动求导机制可以追踪到 self.values
        # 形状 [Out, In]
        w_sparse = torch.sparse_coo_tensor(
            self.indices, 
            self.values, 
            (self.out_features, self.in_features)
        )
        
        # 执行稀疏矩阵乘法: (Out, In) @ (In, Batch)^T -> (Out, Batch) -> Transpose -> (Batch, Out)
        # 注意：torch.sparse.mm 通常要求 (Sparse) @ (Dense)
        # x.t() 是 (In, Batch)
        out = torch.sparse.mm(w_sparse, x.t()).t()
        return out
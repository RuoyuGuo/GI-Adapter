# --------------------------------------------------------
# Adapted from  https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.models.layers import drop_path, to_2tuple, trunc_normal_
except:
    from timm.layers import drop_path, to_2tuple, trunc_normal_
    
from .transformer import PatchDropout
from .rope import VisionRotaryEmbedding, VisionRotaryEmbeddingFast
from ..dino_v2 import DinoVisionTransformer

if os.getenv('ENV_TYPE') == 'deepspeed':
    try:
        from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
    except:
        from torch.utils.checkpoint import checkpoint
else:
    from torch.utils.checkpoint import checkpoint

try:
    import xformers.ops as xops
except ImportError:
    xops = None
    print("Please 'pip install xformers'")

from mmseg.models.builder import BACKBONES

from .adapter_module import MVFuser
from ..adapter_collection import *

from .eva_vit_model import *

@BACKBONES.register_module()
class EVAVisionTransformerBASELINE(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, patch_dropout=0.,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, rope=False,
                 use_mean_pooling=True, init_scale=0.001, grad_checkpointing=False, xattn=False, postnorm=False,
                 pt_hw_seq_len=16, intp_freq=False, naiveswiglu=False, subln=False, out_indices=[], pretrained=None,
                start_indices_vlm=None, start_indices_vfm=None, adapter_type=None):
        super().__init__()
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.start_indices_vlm = start_indices_vlm
        self.start_indices_vfm = start_indices_vfm
        self.adapter_type = adapter_type
        
        self.image_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        
        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
                # patch_dropout=patch_dropout
            )
        else: 
            self.rope = None

        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                xattn=xattn, rope=self.rope, postnorm=postnorm, subln=subln, naiveswiglu=naiveswiglu)
            for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.grad_checkpointing = grad_checkpointing

        self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim*2, embed_dim*2, kernel_size=2, stride=2),
                nn.SyncBatchNorm(embed_dim*2),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim*2, embed_dim*2, kernel_size=2, stride=2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim*2, embed_dim*2, kernel_size=2, stride=2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dinov2 = DinoVisionTransformer(patch_size=14,
                        embed_dim=1024,
                        depth=24,
                        num_heads=16,
                        mlp_ratio=4,
                        img_size=512,
                        ffn_layer="mlp",
                        init_values=1e-05,
                        block_chunks=0,
                        qkv_bias=True,
                        proj_bias=True,
                        ffn_bias=True,)
        dinov2_state_dict = torch.load('pretrained/dinov2_vitl14_pretrain.pth')
        all_keys = list(dinov2_state_dict.keys())
        # interpolate position embedding
        if 'pos_embed' in dinov2_state_dict:
            pos_embed_checkpoint = dinov2_state_dict['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.dinov2.patch_embed.num_patches
            num_extra_tokens = self.dinov2.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                dinov2_state_dict['pos_embed'] = new_pos_embed

                patch_embed_proj = dinov2_state_dict['patch_embed.proj.weight']
                patch_size = self.dinov2.patch_embed.patch_size
                dinov2_state_dict['patch_embed.proj.weight'] = torch.nn.functional.interpolate(
                    patch_embed_proj.float(), size=patch_size, mode='bicubic', align_corners=False)
        self.dinov2.load_state_dict(dinov2_state_dict, strict=True)
        
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_cast_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)
    
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, return_all_features=False):
        x, _ = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.grad_checkpointing:
                x = checkpoint(blk, x, (rel_pos_bias,))
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(1))
            else:
                return x[:, 0]
        return x

    def forward(self, x, return_all_features=False):
        if return_all_features:
            return self.forward_features(x, return_all_features)
        x = self.forward_features(x)
        x = self.head(x)
        return x

    
    # custom functions
    def init_weights(self):
        pass

    def extract_feats(self, x, use_fpn=True, use_adapter=True):
        # Various Foundation Models use different normalization, convert inputs correspondingly
        IMG_MEAN = torch.tensor([ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]).view(1, 3, 1, 1).cuda()
        IMG_STD = torch.tensor([ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]).view(1, 3, 1, 1).cuda()
        original_x = x * IMG_STD + IMG_MEAN
        
        DINOV2_IMG_MEAN = torch.tensor([v * 255 for v in [0.485, 0.456, 0.406]]).view(1, 3, 1, 1).cuda()
        DINOV2_IMG_STD = torch.tensor([v * 255 for v in [0.229, 0.224, 0.225]]).view(1, 3, 1, 1).cuda()
        normalized_x = (original_x - DINOV2_IMG_MEAN) / DINOV2_IMG_STD        
        dinov2_x = self.dinov2.prepare_tokens_with_masks(normalized_x)

        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias)
            
            dinov2_x = self.dinov2.blocks[i](dinov2_x)
  
            if i in self.out_indices:
                xp = torch.cat([dinov2_x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous(), x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()], dim=1)
                features.append(xp.contiguous())
                
        if use_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])
            
        x = self.norm(x) + self.dinov2.norm(dinov2_x)
        
        if self.fc_norm is not None:
            x = self.fc_norm(x)
        x = self.head(x)
        
        global_embedding = x[:, :1]
        visual_embedding = x[:, 1:].permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        features.append([global_embedding, visual_embedding])

        return list(features)

    def get_all_features(self, x, use_fpn=True, use_adapter=True):
        # Various Foundation Models use different normalization, convert inputs correspondingly
        IMG_MEAN = torch.tensor([ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]).view(1, 3, 1, 1).cuda()
        IMG_STD = torch.tensor([ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]).view(1, 3, 1, 1).cuda()
        original_x = x * IMG_STD + IMG_MEAN
        
        DINOV2_IMG_MEAN = torch.tensor([v * 255 for v in [0.485, 0.456, 0.406]]).view(1, 3, 1, 1).cuda()
        DINOV2_IMG_STD = torch.tensor([v * 255 for v in [0.229, 0.224, 0.225]]).view(1, 3, 1, 1).cuda()
        normalized_x = (original_x - DINOV2_IMG_MEAN) / DINOV2_IMG_STD        
        dinov2_x = self.dinov2.prepare_tokens_with_masks(normalized_x)

        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        features = []

        analysis_feataure_vlm = dict()
        analysis_feataure_vfm = dict()
        
        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias)
            
            dinov2_x = self.dinov2.blocks[i](dinov2_x)

            analysis_feataure_vlm[i] = x[:,1:].contiguous()
            analysis_feataure_vfm[i] = dinov2_x[:,1:].contiguous()
            
            if i in self.out_indices:
                xp = torch.cat([dinov2_x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous(), x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()], dim=1)
                features.append(xp.contiguous())
                
        if use_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])
            
        x = self.norm(x) + self.dinov2.norm(dinov2_x)
        
        if self.fc_norm is not None:
            x = self.fc_norm(x)
        x = self.head(x)
        
        global_embedding = x[:, :1]
        visual_embedding = x[:, 1:].permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        features.append([global_embedding, visual_embedding])

        return analysis_feataure_vlm, analysis_feataure_vfm, _ ,_
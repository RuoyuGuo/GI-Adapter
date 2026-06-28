import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict
from mmseg.models.backbones import ResNet
from mmseg.models.builder import BACKBONES
from timm.models.layers import drop_path, trunc_normal_

from ..eva_clip.adapter_module import MVFuser
from ..dino_v2 import DinoVisionTransformer
from ..adapter_collection2 import *
from ..adapter_pruning import *
from .models import *
    
@BACKBONES.register_module()
class CLIPVisionTransformerGate(nn.Module):

    def __init__(self, 
                 input_resolution=224,
                 patch_size=32, 
                 width=768, 
                 layers=12, 
                 heads=12, 
                 output_dim=512, 
                 drop_path_rate=0.0, 
                 out_indices=[3, 5, 7, 11], 
                 pretrained=None, 
                 get_embeddings=False, 
                 ignore_last_attn=False, 

                 #from here, my params
                 adapter_type = None,
                 **kwargs):

        super().__init__()

        self.embed_dim = width
        self.output_dim = output_dim
        self.pretrained = pretrained
        self.patch_size = patch_size
        
        self.adapter_type    = adapter_type
            
        if isinstance(input_resolution, int):
            self.input_resolution = (input_resolution, input_resolution)
        elif isinstance(input_resolution, tuple):
            self.input_resolution = input_resolution       

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn((self.input_resolution[0] // patch_size) * (self.input_resolution[1] // patch_size) + 1, width))
        self.spatial_size = (self.input_resolution[0] // patch_size, self.input_resolution[1] // patch_size)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.out_indices = out_indices
        self.ignore_last_attn = ignore_last_attn

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))      

        self.fpn_dim = width + 1024
        self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(self.fpn_dim),
                nn.GELU(),
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)      
        
        # DINOv2-L
        self.dinov2 = DinoVisionTransformer(patch_size=16,
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
        
   
        self.vlm_adapter   = nn.Sequential(*[MSGateAdapter(borrow_flag=True) for i in range(24)]) 
        self.vfm_adapter   = nn.Sequential(*[MSGateAdapter(borrow_flag=False) for i in range(24)])
            
        # self.adapter = nn.Sequential(*[MVFuser(self.embed_dim, d_state=16) for i in range(layers)])
        # self.adapter_proj1 = nn.Sequential(*[nn.Linear(1024, self.embed_dim) for i in range(layers)])
        # self.adapter_proj2 = nn.Sequential(*[nn.Linear(self.embed_dim, 1024) for i in range(layers)])
        self.adapter_proj3 = nn.Linear(1024, self.embed_dim)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        print("backbone:", pretrained)
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    print(f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]
                    orig_size = int(state_dict["positional_embedding"][1:,].shape[0] ** 0.5)
                    spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, orig_size, orig_size, self.embed_dim).permute(0, 3, 1, 2), size=self.spatial_size, mode='bilinear')
                    spatial_pos = spatial_pos.reshape(self.embed_dim, self.spatial_size[0]*self.spatial_size[1]).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            if self.conv1.weight.shape != state_dict['conv1.weight'].shape:
                print(f'Resize the patch_embed shape from {state_dict["conv1.weight"].shape} to {self.conv1.weight.shape}')
                state_dict["conv1.weight"] = F.interpolate(state_dict["conv1.weight"], size=self.conv1.weight.shape[-2:], mode='bilinear')
                assert self.conv1.weight.shape == state_dict['conv1.weight'].shape
                
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer')

    def prepare_tokens_with_masks(self, x: torch.Tensor):
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size[0], self.spatial_size[1], C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)

        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        return x
    
    @staticmethod
    def convert_list_to_tensor(list_convert):
        if len(list_convert):
            result = torch.stack(list_convert, dim=1)
        else :
            result = None
        return result 
    
    def forward(self, x: torch.Tensor, use_adapter=True, train_loss=False):
        # Various Foundation Models use different normalization, convert inputs correspondingly
        IMG_MEAN = torch.tensor([ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]).view(1, 3, 1, 1).cuda()
        IMG_STD = torch.tensor([ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]).view(1, 3, 1, 1).cuda()
        original_x = x * IMG_STD + IMG_MEAN
        DINOV2_IMG_MEAN = torch.tensor([v * 255 for v in [0.485, 0.456, 0.406]]).view(1, 3, 1, 1).cuda()
        DINOV2_IMG_STD = torch.tensor([v * 255 for v in [0.229, 0.224, 0.225]]).view(1, 3, 1, 1).cuda()
        normalized_x = (original_x - DINOV2_IMG_MEAN) / DINOV2_IMG_STD        
        dinov2_x = self.dinov2.prepare_tokens_with_masks(normalized_x)
        
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size[0], self.spatial_size[1], C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)

        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        features = []

        vlm_o_feature_list = []
        vfm_o_feature_list = []
        vlm_feature_list = []
        vfm_feature_list = []

        
        for i, blk in enumerate(self.transformer.resblocks):
            if self.ignore_last_attn:
                mask = torch.empty(x.shape[0], x.shape[0])
                mask.fill_(float('-inf'))
                mask.fill_diagonal_(0)
                self.transformer.resblocks[-1].attn_mask = mask

            x = blk(x)
            dinov2_x = self.dinov2.blocks[i](dinov2_x)
            
            #use aadapter
            if use_adapter:
                x_delta       = self.vlm_adapter[i](x_self=x.permute(1,0,2)[:,1:], x_borrow=dinov2_x[:,1:])
                dinov2_delta  = self.vfm_adapter[i](x_self=dinov2_x[:,1:],         x_borrow=x.permute(1,0,2)[:,1:])
    
                
                x[1:]          = x[1:]        + x_delta.permute(1,0,2)
                dinov2_x[:,1:] = dinov2_x[:,1:] + dinov2_delta


            if i in self.out_indices:
                xp = torch.cat([dinov2_x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous(), x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()], dim=1)
                features.append(xp.contiguous())
        
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        if self.get_embeddings:
            x = x.permute(1, 0, 2) + self.adapter_proj3(dinov2_x)
            x = self.ln_post(x)
            x = x @ self.proj
            
            global_embedding = x[:, :1]
            visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2) # B C H W

            features.append([global_embedding, visual_embedding])
        

        return tuple(features)
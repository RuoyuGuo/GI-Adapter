# coding=utf-8
# Copyright 2024 Google AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Siglip model."""

import math
import warnings
from typing import Any, Optional, Tuple, Union
from packaging import version
from torch import Tensor, nn

import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.init import _calculate_fan_in_and_fan_out

from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    torch_int,
)
from transformers.activations import PytorchGELUTanh
from mmseg.models.builder import BACKBONES

from ..eva_clip.adapter_module import MVFuser
from ..dino_v2 import DinoVisionTransformer

from .modeling_siglip import *

if is_flash_attn_2_available():
    from ...modeling_flash_attention_utils import _flash_attention_forward


logger = logging.get_logger(__name__)

@BACKBONES.register_module()
class SiglipVisionTransformerBASELINE(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12, num_channels=3,
        image_size=224, patch_size=16, hidden_act=PytorchGELUTanh(), layer_norm_eps=1e-6, attention_dropout=0.0, 
        out_indices=[7, 11, 15, 23], pretrained=None, output_attentions=False, output_hidden_states=False, return_dict=True, **kwargs,):
        super().__init__()
        self.image_size = image_size
        self.embed_dim = hidden_size
        self.hidden_size = hidden_size
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.num_hidden_layers = num_hidden_layers
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict
        self.spatial_size = (self.image_size // patch_size, self.image_size // patch_size)
        self.embeddings = SiglipVisionEmbeddings(hidden_size, image_size, patch_size, num_channels)
        self.encoder = SiglipEncoder(hidden_size, hidden_act, num_attention_heads, intermediate_size, attention_dropout, layer_norm_eps, num_hidden_layers)
        self.post_layernorm = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)
        self.use_head = True 
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(hidden_size, num_attention_heads, layer_norm_eps, hidden_act, intermediate_size)

        self.fpn_dim = self.embed_dim + 1024
        self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(self.fpn_dim),
                nn.GELU(),
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)   

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
        self.adapter_proj3 = nn.Linear(1024, self.embed_dim)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        print("backbone:", pretrained)
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained, map_location='cpu')
            self.logit_scale = checkpoint['logit_scale']
            self.logit_bias = checkpoint['logit_bias']

            state_dict = {}
            
            for k in checkpoint.keys():
                if k.startswith('vision_model.'):
                    new_k = k.replace('vision_model.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'embeddings.position_embedding.weight' in state_dict.keys():
                if self.embeddings.position_embedding.weight.shape != state_dict['embeddings.position_embedding.weight'].shape:
                    print(f'Resize the pos_embed shape from {state_dict["embeddings.position_embedding.weight"].shape} to {self.embeddings.position_embedding.weight.shape}')
                    orig_size = int(state_dict["embeddings.position_embedding.weight"].shape[0] ** 0.5)
                    spatial_pos = F.interpolate(state_dict["embeddings.position_embedding.weight"].reshape(1, orig_size, orig_size, 1024).permute(0, 3, 1, 2), size=self.spatial_size, mode='bilinear')
                    spatial_pos = spatial_pos.reshape(1024, self.spatial_size[0]*self.spatial_size[1]).permute(1, 0)
                    position_embedding = spatial_pos
                    state_dict['embeddings.position_embedding.weight'] = position_embedding
                    assert self.embeddings.position_embedding.weight.shape == state_dict['embeddings.position_embedding.weight'].shape

            if self.embeddings.patch_embedding.weight.shape != state_dict['embeddings.patch_embedding.weight'].shape:
                print(f'Resize the patch_embed shape from {state_dict["embeddings.patch_embedding.weight"].shape} to {self.embeddings.patch_embedding.weight.shape}')
                state_dict["embeddings.patch_embedding.weight"] = F.interpolate(state_dict["embeddings.patch_embedding.weight"], size=self.embeddings.patch_embedding.weight.shape[-2:], mode='bilinear')
                assert self.embeddings.patch_embedding.weight.shape == state_dict['embeddings.patch_embedding.weight'].shape
                
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer')


    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings.patch_embedding

    def forward(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        use_adapter = True
    ):

        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.return_dict

        # Various Foundation Models use different normalization, convert inputs correspondingly
        IMG_MEAN = torch.tensor([ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]).view(1, 3, 1, 1).cuda()
        IMG_STD = torch.tensor([ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]).view(1, 3, 1, 1).cuda()
        original_pixel_values = pixel_values * IMG_STD + IMG_MEAN
        DINOV2_IMG_MEAN = torch.tensor([v * 255 for v in [0.485, 0.456, 0.406]]).view(1, 3, 1, 1).cuda()
        DINOV2_IMG_STD = torch.tensor([v * 255 for v in [0.229, 0.224, 0.225]]).view(1, 3, 1, 1).cuda()
        normalized_pixel_values = (original_pixel_values - DINOV2_IMG_MEAN) / DINOV2_IMG_STD
        dinov2_x = self.dinov2.prepare_tokens_with_masks(normalized_pixel_values)

        SIGLIP_IMG_MEAN = torch.tensor([v * 255 for v in [0.5, 0.5, 0.5]]).view(1, 3, 1, 1).cuda()
        SIGLIP_IMG_STD = torch.tensor([v * 255 for v in [0.5, 0.5, 0.5]]).view(1, 3, 1, 1).cuda()
        normalized_pixel_values = (original_pixel_values - SIGLIP_IMG_MEAN) / SIGLIP_IMG_STD
        
        hidden_states = self.embeddings(normalized_pixel_values, interpolate_pos_encoding=interpolate_pos_encoding) #[b_size, num_tokens, fea_dim]

        features = []
        for i, encoder_layer in enumerate(self.encoder.layers):
            
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=None,
                output_attentions=output_attentions,
            )[0]
            
            dinov2_x = self.dinov2.blocks[i](dinov2_x)
            
            
            if i in self.out_indices:
                xp = torch.cat([dinov2_x[:, 1:, :].permute(0, 2, 1).reshape(dinov2_x.shape[0], -1, self.spatial_size[0], self.spatial_size[1]).contiguous(), hidden_states.permute(0, 2, 1).reshape(hidden_states.shape[0], -1, self.spatial_size[0], self.spatial_size[1]).contiguous()], dim=1)
                features.append(xp.contiguous())
            
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        last_hidden_state = self.post_layernorm(hidden_states) + self.dinov2.norm(dinov2_x[:, 1:, :])

        pooler_output = self.head(last_hidden_state) if self.use_head else None

        global_embedding = pooler_output[0]
        visual_embedding = pooler_output[1].reshape(last_hidden_state.shape[0], self.spatial_size[0], self.spatial_size[1], -1).permute(0, 3, 1, 2) # B C H W
        
        features.append([global_embedding, visual_embedding])
        
        return tuple(features)

    def get_all_features(
        self,
        pixel_values,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        use_adapter = True
    ):

        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.return_dict

        # Various Foundation Models use different normalization, convert inputs correspondingly
        IMG_MEAN = torch.tensor([ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]).view(1, 3, 1, 1).cuda()
        IMG_STD = torch.tensor([ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]).view(1, 3, 1, 1).cuda()
        original_pixel_values = pixel_values * IMG_STD + IMG_MEAN
        DINOV2_IMG_MEAN = torch.tensor([v * 255 for v in [0.485, 0.456, 0.406]]).view(1, 3, 1, 1).cuda()
        DINOV2_IMG_STD = torch.tensor([v * 255 for v in [0.229, 0.224, 0.225]]).view(1, 3, 1, 1).cuda()
        normalized_pixel_values = (original_pixel_values - DINOV2_IMG_MEAN) / DINOV2_IMG_STD
        dinov2_x = self.dinov2.prepare_tokens_with_masks(normalized_pixel_values)

        SIGLIP_IMG_MEAN = torch.tensor([v * 255 for v in [0.5, 0.5, 0.5]]).view(1, 3, 1, 1).cuda()
        SIGLIP_IMG_STD = torch.tensor([v * 255 for v in [0.5, 0.5, 0.5]]).view(1, 3, 1, 1).cuda()
        normalized_pixel_values = (original_pixel_values - SIGLIP_IMG_MEAN) / SIGLIP_IMG_STD
        
        hidden_states = self.embeddings(normalized_pixel_values, interpolate_pos_encoding=interpolate_pos_encoding) #[b_size, num_tokens, fea_dim]

        features = []        
        vlm_features = dict()
        vfm_features = dict()
        for i, encoder_layer in enumerate(self.encoder.layers):
            
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=None,
                output_attentions=output_attentions,
            )[0]
            
            dinov2_x = self.dinov2.blocks[i](dinov2_x)
            
            vlm_features[i] = hidden_states.contiguous()
            vfm_features[i] = dinov2_x.contiguous()
            
            if i in self.out_indices:
                xp = torch.cat([dinov2_x[:, 1:, :].permute(0, 2, 1).reshape(dinov2_x.shape[0], -1, self.spatial_size[0], self.spatial_size[1]).contiguous(), hidden_states.permute(0, 2, 1).reshape(hidden_states.shape[0], -1, self.spatial_size[0], self.spatial_size[1]).contiguous()], dim=1)
                features.append(xp.contiguous())
            
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        last_hidden_state = self.post_layernorm(hidden_states) + self.dinov2.norm(dinov2_x[:, 1:, :])

        pooler_output = self.head(last_hidden_state) if self.use_head else None

        global_embedding = pooler_output[0]
        visual_embedding = pooler_output[1].reshape(last_hidden_state.shape[0], self.spatial_size[0], self.spatial_size[1], -1).permute(0, 3, 1, 2) # B C H W
        
        features.append([global_embedding, visual_embedding])
        
        return vlm_features, vfm_features
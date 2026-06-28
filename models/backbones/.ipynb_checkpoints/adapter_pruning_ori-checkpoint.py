import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
# from einops import rearrange
import einops

from .pruning.adapter import DeltaBlock, Compensator
from .pruning.mask import TokenSelect


class WrapAdapterVFM(nn.Module): 
    '''
    in:  B, T, C
    out: B, T, C
    '''
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0):
        super().__init__()
        self.spatial_adapter = Compensator(dim=dim, bottleneck_dim=bottleneck_dim)
        self.linear_adapter = DeltaBlock(dim=dim, bottleneck_dim=bottleneck_dim)
        self.mask_gen = TokenSelect(dim=dim, mask_dim=1)
        self.scale = scale
        self.dim = dim
        
    def forward(self, x: Tensor, blk: nn.Module) -> torch.Tensor:
        #original blk operation
        def attn_residual_func(x: Tensor) -> Tensor:
            return blk.ls1(blk.attn(blk.norm1(x)))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return blk.ls2(blk.mlp(blk.norm2(x)))
        
        #att no changes
        x = x + attn_residual_func(x)
        
        #adjust ffn layer with PEFT operation\
        
        if self.training: 
            policy_token = x
            
            #generate token mask
            token_mask, token_logits = self.mask_gen(policy_token)
            
            #normal ffn and apply token_mask\
            #mlp_x is the ffn outputs
            mlp_x = ffn_residual_func(x)
            if token_mask is not None:
                mlp_x = token_mask * mlp_x
            
            #linear adapter and spatial adapter
            adpt_x = self.linear_adapter(x)
            spt_x = self.spatial_adapter(x)
            
            #adding
            adpt_x[:, 1:, :] = adpt_x[:, 1:, :] + spt_x
            adpt_x = adpt_x * self.scale
            
            #
            x = x + mlp_x + adpt_x

        else:
            policy_token = x
            token_mask, token_logits = self.mask_gen(policy_token)
            
            #linear adapter and spatial adapter
            adpt_x = self.linear_adapter(x)
            spt_x = self.spatial_adapter(x)
            
            #adding
            adpt_x[:, 1:, :] = adpt_x[:, 1:, :] + spt_x
            adpt_x = adpt_x * self.scale
            
            #for ffn layer, pruning
            if token_mask is not None:
                idx = torch.nonzero(token_mask.detach(), as_tuple=True)[1] # (K, )
                idx = idx.unsqueeze(0) # (1, K, )
                idx = einops.repeat(idx, 'b k -> b k d', d=self.dim)
                mlp_x = torch.gather(x, dim=1, index=idx)
                mlp_x = ffn_residual_func(mlp_x)
                x = x + adpt_x.scatter_add(dim=1, index=idx, src=mlp_x)      
            else:
                mlp_x = ffn_residual_func(x)
                x = x + mlp_x + adpt_x
            
        return x, dict(sub_token_select=token_mask, token_logits=token_logits)
    
    
class WrapAdapterCLIP(nn.Module): 
    '''
    in:  T, B, C
    out: T, B, C
    permute when only we need it to change
    '''
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0):
        super().__init__()
        self.spatial_adapter = Compensator(dim=dim, bottleneck_dim=bottleneck_dim)
        self.linear_adapter = DeltaBlock(dim=dim, bottleneck_dim=bottleneck_dim)
        self.mask_gen = TokenSelect(dim=dim, mask_dim=1)
        self.scale = scale
        self.dim = dim
    
    def clip_op(self, x):
        return x.permute(1,0,2)
    
    def forward(self, x: Tensor, blk: nn.Module) -> torch.Tensor:
        #att no changes
        x = x + blk.drop_path(blk.attention(blk.ln_1(x)))
        
        #adjust ffn layer with PEFT operation
        if self.training:
            policy_token = x
            
            #generate token mask, adjust to B, T, C for clip
            token_mask, token_logits = self.mask_gen(policy_token.permute(1,0,2)) 
            
            #normal ffn and apply token_mask
            #mlp_x is the ffn outputs
            mlp_x = blk.drop_path(blk.mlp(blk.ln_2(x)))
            
            # if token_mask is not None:
            #mask output is B, T, C, need to adjust to T, B, C
            mlp_x = token_mask.permute(1,0,2) * mlp_x 
            
            #linear adapter and spatial adapter
            #use clip op function
            adpt_x = self.linear_adapter(x.permute(1,0,2))
            spt_x = self.spatial_adapter(x.permute(1,0,2))
            
            #adding
            adpt_x[:, 1:, :] = adpt_x[:, 1:, :] + spt_x
            adpt_x = adpt_x * self.scale
            
            #
            x = x + mlp_x + adpt_x.permute(1,0,2)

        else:
            policy_token = x
            token_mask, token_logits = self.mask_gen(policy_token.permute(1,0,2))
            
            #linear adapter and spatial adapter
            adpt_x = self.linear_adapter(x.permute(1,0,2))
            spt_x = self.spatial_adapter(x.permute(1,0,2))
            
            #adding
            adpt_x[:, 1:, :] = adpt_x[:, 1:, :] + spt_x
            adpt_x = adpt_x * self.scale
            
            #for ffn layer, pruning
            if token_mask is not None:
                idx = torch.nonzero(token_mask.detach(), as_tuple=True)[1] # (K, )
                idx = idx.unsqueeze(0) # (1, K, )
                idx = einops.repeat(idx, 'b k -> b k d', d=self.dim)
                
                #x is T, B, C, trans to B, T, C
                mlp_x = torch.gather(x.permute(1,0,2), dim=1, index=idx)

                #ori lln here, trans X back to T,B,C
                mlp_x = blk.drop_path(blk.mlp(blk.ln_2(mlp_x.permute(1,0,2))))
                x = x + adpt_x.scatter_add(dim=1, index=idx, src=mlp_x.permute(1,0,2)).permute(1,0,2)      

            else:
                #ori lln here
                mlp_x = blk.drop_path(blk.mlp(blk.ln_2(x)))
                x = x + mlp_x + adpt_x.permute(1,0,2)
                
        return x, dict(sub_token_select=token_mask, token_logits=token_logits)

class WrapATT(nn.Module): 
    '''
    in:  B, T, C
    out: B, T, C
    '''
    
    def __init__(self, model_name):
        super().__init__()      
        self.model_name = model_name
        
    def forward(self, x, blk):
        if self.model_name == 'dinov2':
            return blk.ls1(blk.attn(blk.norm1(x)))
        
        elif self.model_name == 'clip':
            return blk.drop_path(blk.attention(blk.ln_1(x)))
        
        else:
            assert False, "Unkonw name"
        
class WrapFFN(nn.Module):
    '''
    in:  B, T, C
    out: B, T, C
    '''
    
    def __init__(self, model_name):
        super().__init__()      
        self.model_name = model_name
        
    def forward(self, x, blk):
        if self.model_name == 'dinov2':
            return blk.ls2(blk.mlp(blk.norm2(x)))
        
        elif self.model_name == 'clip':
            return blk.drop_path(blk.mlp(blk.ln_2(x)))   
        
        else:
            assert False, "Unkonw name"

            
    
class TR(nn.Module):
    def __init__(self, model_name, dim=1024, bottleneck_dim=64, scale=0.1):
        super().__init__()
        self.spatial_adapter = Compensator(dim=dim, bottleneck_dim=bottleneck_dim)
        self.linear_adapter = DeltaBlock(dim=dim, bottleneck_dim=bottleneck_dim)
        self.mask_gen = TokenSelect(dim=dim*2, mask_dim=1)
        self.scale = nn.Parameter(torch.tensor(scale))
        self.dim = dim 
        
        self.model_name = model_name

    def forward(self, x, x_other):
        if self.model_name == 'clip':
            x = x.permute(1,0,2)
        
        #att layer
        policy_token = x

        #generate token mask
        token_mask, token_logits = self.mask_gen(torch.cat([policy_token, x_other], dim=-1))

        #linear adapter and spatial adapter
        adpt_x = self.linear_adapter(x)
        spt_x  = self.spatial_adapter(x)

        #adding
        adpt_x[:, 1:, :] = adpt_x[:, 1:, :] + spt_x
        adpt_x = adpt_x * self.scale
        
        
        if self.model_name == 'clip':       
            return adpt_x.permute(1,0,2), token_mask.permute(1,0,2), dict(sub_token_select=token_mask, token_logits=token_logits)
        
        else:       
            return adpt_x, token_mask, dict(sub_token_select=token_mask, token_logits=token_logits)
        
class TRr(nn.Module):
    def __init__(self, model_name, dim=1024, bottleneck_dim=64, scale=0.1):
        super().__init__()
        # self.spatial_adapter = Compensator(dim=dim, bottleneck_dim=bottleneck_dim)
        self.linear_adapter = DeltaBlock(dim=dim, bottleneck_dim=bottleneck_dim)
        self.mask_gen = TokenSelect(dim=dim*2, mask_dim=1)
        self.scale = nn.Parameter(torch.tensor(scale))
        self.dim = dim 
        
        self.model_name = model_name

    def forward(self, x, x_other):
        if self.model_name == 'clip':
            x = x.permute(1,0,2)
        
        #att layer
        policy_token = x

        #generate token mask
        token_mask, token_logits = self.mask_gen(torch.cat([policy_token, x_other], dim=-1))

        #linear adapter and spatial adapter
        adpt_x = self.linear_adapter(x)
        # spt_x  = self.spatial_adapter(x)

        #adding
        adpt_x[:, 1:, :] = adpt_x[:, 1:, :] + spt_x
        adpt_x = (adpt_x * self.scale * (1-token_mask)) * (1-token_logits)
        
        
        if self.model_name == 'clip':       
            return adpt_x.permute(1,0,2), token_mask.permute(1,0,2), token_logits, dict(sub_token_select=token_mask, token_logits=token_logits)
        
        else:       
            return adpt_x, token_mask, token_logits. dict(sub_token_select=token_mask, token_logits=token_logits)
            
        
        #Inference, to do...


class AR(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0, borrow_flag=False):
        '''
        borrow_flag: 
            True: borrow from another model
            False: self evolve
        '''
        super().__init__()
        # hr module
        self.adapter = DeltaBlock(dim, bottleneck_dim)
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1),
            nn.Sigmoid(),
        )
        self.scale = scale
        self.borrow_flag = borrow_flag
            

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.weight[0].bias)
            
            #zero init
            nn.init.zeros_(self.weight[2].weight)
            nn.init.zeros_(self.weight[2].bias)

    def forward(self, x_self, x_borrow):
        # Transform VFM via LoRA-style adapter
        if self.borrow_flag is True:
            #borrowing
            delta = self.adapter(x_borrow)
        else:
            #self evolving
            delta = self.adapter(x_self)
            
        #learning weight
        x_cat = torch.cat([x_self, x_borrow], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)      

        return self.scale * w * delta, w.detach()


class ARr(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0, borrow_flag=False):
        '''
        borrow_flag: 
            True: borrow from another model
            False: self evolve
        '''
        super().__init__()
        # hr module
        self.adapter = DeltaBlock(dim, bottleneck_dim)
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1),
            nn.Sigmoid(),
        )
        self.scale = scale
        self.borrow_flag = borrow_flag
            

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.weight[0].bias)
            
            #zero init
            nn.init.zeros_(self.weight[2].weight)
            nn.init.zeros_(self.weight[2].bias)

    def forward(self, x_self, x_borrow):
        # Transform VFM via LoRA-style adapter
        if self.borrow_flag is True:
            #borrowing
            delta = self.adapter(x_borrow)
        else:
            #self evolving
            delta = self.adapter(x_self)
            
        #learning weight
        x_cat = torch.cat([x_self, x_borrow], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)      

        return self.scale * w * delta, w.detach()
        
    `
class TRDual(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, VLM='clip', scale_TR=0.1):
        super().__init__()
        self.TR_VLM = TR(model_name=VLM, dim=dim, bottleneck_dim=bottleneck_dim, scale=scale_TR)
        self.TR_VFM = TR(model_name='dinov2', dim=dim, bottleneck_dim=bottleneck_dim, scale=scale_TR) 
        
        self.att_layer_vlm = WrapATT(VLM) 
        self.att_layer_vfm = WrapATT('dinov2')
        
        self.ffn_layer_vlm = WrapFFN(VLM) 
        self.ffn_layer_vfm = WrapFFN('dinov2')
        
        
    def forward(self, x_vlm, x_vfm, blk_vlm, blk_vfm):
        
        #forward att layer
        x_vlm = x_vlm + self.att_layer_vlm(x_vlm, blk_vlm)
        x_vfm = x_vfm + self.att_layer_vfm(x_vfm, blk_vfm)
        
        #generate mask, and adapted output
        apt_vlm, mask_vlm, dict_vlm = self.TR_VLM(x_vlm, x_other=x_vfm)
        apt_vfm, mask_vfm, dict_vfm = self.TR_VFM(x_vfm, x_other=x_vlm.permute(1,0,2))
        
        #applying on the ffn and forward
        mlp_vlm = self.ffn_layer_vlm(x_vlm, blk_vlm)
        mlp_vfm = self.ffn_layer_vfm(x_vfm, blk_vfm)
        
        #apply mask and adapted output
        x_vlm = x_vlm + mlp_vlm * mask_vlm + apt_vlm
        x_vfm = x_vfm + mlp_vfm * mask_vfm + apt_vfm
        
        return x_vlm, x_vfm, dict_vlm, dict_vfm
    
    
class TRrDual(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, VLM='clip', scale_TR=0.1):
        super().__init__()
        self.TR_VLM = TRr(model_name=VLM, dim=dim, bottleneck_dim=bottleneck_dim, scale=scale_TR)
        self.TR_VFM = TRr(model_name='dinov2', dim=dim, bottleneck_dim=bottleneck_dim, scale=scale_TR) 
        
        self.att_layer_vlm = WrapATT(VLM) 
        self.att_layer_vfm = WrapATT('dinov2')
        
        self.ffn_layer_vlm = WrapFFN(VLM) 
        self.ffn_layer_vfm = WrapFFN('dinov2')
        
        
    def forward(self, x_vlm, x_vfm, blk_vlm, blk_vfm):
        
        #forward att layer
        x_vlm = x_vlm + self.att_layer_vlm(x_vlm, blk_vlm)
        x_vfm = x_vfm + self.att_layer_vfm(x_vfm, blk_vfm)
        
        #generate mask, and adapted output
        apt_vlm, mask_vlm, dict_vlm = self.TR_VLM(x_vlm, x_other=x_vfm)
        apt_vfm, mask_vfm, dict_vfm = self.TR_VFM(x_vfm, x_other=x_vlm.permute(1,0,2))
        
        #applying on the ffn and forward
        mlp_vlm = self.ffn_layer_vlm(x_vlm, blk_vlm)
        mlp_vfm = self.ffn_layer_vfm(x_vfm, blk_vfm)
        
        #apply mask and adapted output
        x_vlm = x_vlm + mlp_vlm * mask_vlm + apt_vlm
        x_vfm = x_vfm + mlp_vfm * mask_vfm + apt_vfm
        
        return x_vlm, x_vfm, dict_vlm, dict_vfm

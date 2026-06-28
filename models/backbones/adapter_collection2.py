import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0, act_layer='sigmoid', borrow_flag=False):
        '''
        borrow_flag: 
            True: borrow from another model
            False: self evolve
        '''
        super().__init__()
        # hr module
        self.adapter = DeltaBlock(dim, bottleneck_dim)
        self.proj = nn.Linear(dim, dim)
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1),
        )
        self.scale = scale
        self.borrow_flag = borrow_flag
        
        if act_layer == 'identity':
            self.act = nn.Identity()
        elif act_layer == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_layer == 'gelu':
            self.act = nn.GELU()
        else:
            assert False, f"Not implement"
            

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)
            
            #zero init
            nn.init.kaiming_uniform_(self.weight[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.weight[0].bias)
            
            nn.init.zeros_(self.weight[-1].weight)
            nn.init.zeros_(self.weight[-1].bias)

    def forward(self, x_self, x_borrow):
        # Transform VFM via LoRA-style adapter
        if self.borrow_flag is True:
            #borrowing
            delta = self.adapter(x_borrow)
        else:
            #self evolving
            delta = self.adapter(x_self)
            
        #learning weight
        x_cat = torch.cat([x_self, self.proj(x_borrow)], dim=-1)  # [B, T, 2C]
        w = self.act(self.weight(x_cat))         

        return self.scale * w * delta


class DynAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0, act_layer='sigmoid', borrow_flag=False):
        '''
        borrow_flag: 
            True: borrow from another model
            False: self evolve
        '''
        super().__init__()
        # hr module
        self.adapter_self = DeltaBlock(dim, bottleneck_dim)
        self.adapter_borrow = DeltaBlock(dim, bottleneck_dim)
        
        self.proj = nn.Linear(dim, dim)
        
        self.weight_self = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1),
        )

        self.weight_borrow = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1),
        )
        self.scale = scale
        self.borrow_flag = borrow_flag
        
        if act_layer == 'identity':
            self.act = nn.Identity()
        elif act_layer == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_layer == 'gelu':
            self.act = nn.GELU()
        else:
            assert False, f"Not implement"
            

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)
            
            #zero init
            nn.init.zeros_(self.weight_self[-1].weight)
            nn.init.zeros_(self.weight_self[-1].bias)
            
            nn.init.zeros_(self.weight_borrow[-1].weight)
            nn.init.zeros_(self.weight_borrow[-1].bias)

    def forward(self, x_self, x_borrow):
        delta_self    = self.adapter_self(x_self)
        delta_borrow  = self.adapter_borrow(x_borrow)
            
        #learning weight
        x_cat = torch.cat([x_self, self.proj(x_borrow)], dim=-1)  # [B, T, 2C]
        
        w_borrow = self.act(self.weight_borrow(x_cat))  
        w_self = self.act(self.weight_self(x_cat))       

        return self.scale * w_self * delta_self + self.scale * w_borrow * delta_borrow, w_self, w_borrow

    def getfea(self, x_self, x_borrow):
        delta_self    = self.adapter_self(x_self)
        delta_borrow  = self.adapter_borrow(x_borrow)
            
        #learning weight
        x_cat = torch.cat([x_self, self.proj(x_borrow)], dim=-1)  # [B, T, 2C]
        
        w_borrow = self.act(self.weight_borrow(x_cat))  
        w_self = self.act(self.weight_self(x_cat))       

        return self.scale * w_self * delta_self + self.scale * w_borrow * delta_borrow, w_self*delta_self, w_borrow*delta_borrow


class PrunAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0, act_layer='sigmoid'):
        '''
        borrow_flag: 
            True: borrow from another model
            False: self evolve
        '''
        super().__init__()
        # hr module
        self.adapter_self = DeltaBlock(dim, bottleneck_dim)
        self.adapter_borrow = DeltaBlock(dim, bottleneck_dim)
        
        self.proj = nn.Linear(dim, dim)
        
        self.weight_self = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1),
        )

        self.weight_borrow = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1),
        )
        self.scale = scale
        
        if act_layer == 'identity':
            self.act = nn.Identity()
        elif act_layer == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_layer == 'gelu':
            self.act = nn.GELU()
        else:
            assert False, f"Not implement"

        self.register_buffer("act_self", torch.tensor(True))
        self.register_buffer("act_borrow", torch.tensor(True))

        self.running_mean    = {'self': None,
                                'borrow': None,}
        
        self.running_entropy = {'self': None,
                               'borrow': None,}
            

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)
            
            #zero init
            nn.init.zeros_(self.weight_self[-1].weight)
            nn.init.zeros_(self.weight_self[-1].bias)
            
            nn.init.zeros_(self.weight_borrow[-1].weight)
            nn.init.zeros_(self.weight_borrow[-1].bias)

    
    def update_stats(self, w_self, w_borrow):
        """
        Update the running mean and entropy for self and borrow gating weights.
        """

        def compute_mean_and_entropy(w):
            with torch.no_grad():
                w = w.detach().squeeze(-1)  # [B, T]
            
                mean = w.mean().item()
            
                eps = 1e-6
                entropy = -(w * (w + eps).log() + (1 - w) * (1 - w + eps).log())
                entropy = entropy.mean().item() / math.log(2)
        
            return mean, entropy

        if self.act_self.item():
            mean, entropy = compute_mean_and_entropy(w_self)
            self.running_mean['self'] = mean
            self.running_entropy['self'] = entropy

        if self.act_borrow.item():
            mean, entropy = compute_mean_and_entropy(w_borrow)
            self.running_mean['borrow'] = mean
            self.running_entropy['borrow'] = entropy
            

    def forward(self, x_self, x_borrow):
        #learning weight
        if self.act_self.item() or self.act_borrow.item():
            x_cat = torch.cat([x_self, self.proj(x_borrow)], dim=-1)  # [B, T, 2C]

        #learnning delta
        if self.act_self.item():
            delta_self  = self.adapter_self(x_self)
            w_self = self.act(self.weight_self(x_cat))       
        else:
            delta_self = 0
            w_self = 0
            
        if self.act_borrow.item():
            delta_borrow  = self.adapter_borrow(x_borrow)
            w_borrow = self.act(self.weight_borrow(x_cat))  
        else:
            delta_borrow = 0
            w_borrow = 0

        self.update_stats(w_self, w_borrow)

        return self.scale * w_self * delta_self + self.scale * w_borrow * delta_borrow, w_self, w_borrow

class VLMAdapterMoe(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=1):
        super().__init__()
        
        # Linear projection for gating input
        self.proj = nn.Linear(dim, dim)
        
        # LoRA-style MLP adapter (non-zero init to allow learning)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else lambda x: x
        self.up = nn.Linear(bottleneck_dim, dim)

        # Gating MLP from [VFM_proj, VLM]
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1)
        )

        self.scale = scale

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)

            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)

            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.bias)
            
            #zero init
            nn.init.zeros_(self.weight[-1].weight)
            nn.init.zeros_(self.weight[-1].bias)

    def forward(self, x_vfm, x_vlm, w_Moe=1):
        # Transform VFM via LoRA-style adapter
        delta = self.down(x_vfm)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up(delta)

        # Compute token-wise gate
        x_cat = torch.cat([self.proj(x_vfm), x_vlm], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                # [B, T, 1]

        # Residual injection into VLM
        return x_vlm + self.scale * w * delta * w_Moe

    def getweights(self, x_vfm, x_vlm, w_Moe=1):
        # Transform VFM via LoRA-style adapter
        delta = self.down(x_vfm)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up(delta)

        # Compute token-wise gate
        x_cat = torch.cat([self.proj(x_vfm), x_vlm], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                # [B, T, 1]

        # Residual injection into VLM
        return x_vlm + self.scale * w * delta * w_Moe, w

class VFMAdapterMoe(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=0.1):
        super().__init__()
        
        # Projection for cross-model gating (VLM feature)
        self.proj = nn.Linear(dim, dim)

        # LoRA-style MLP adapter for VFM
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else nn.Identity()
        self.up = nn.Linear(bottleneck_dim, dim)

        # Token-wise gating from [x_vfm, projected x_vlm]
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1)  # token-wise scalar gate
        )

        self.scale = scale  # global residual scale (optional)

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)

            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)

            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.bias)

            # zero init
            nn.init.zeros_(self.weight[-1].weight)
            nn.init.zeros_(self.weight[-1].bias)

    def forward(self, x_vfm, x_vlm, w_Moe=1):
        # Adapt VFM features (delta)
        delta = self.down(x_vfm)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up(delta)

        # Compute token-wise gating weight using [x_vfm, projected x_vlm]
        x_cat = torch.cat([x_vfm, self.proj(x_vlm)], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                    # [B, T, 1]

        # Inject back into VFM (residual)
        return x_vfm + self.scale * w * delta * w_Moe

    def getweights(self, x_vfm, x_vlm, w_Moe=1):
        # Adapt VFM features (delta)
        delta = self.down(x_vfm)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up(delta)

        # Compute token-wise gating weight using [x_vfm, projected x_vlm]
        x_cat = torch.cat([x_vfm, self.proj(x_vlm)], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                    # [B, T, 1]

        # Inject back into VFM (residual)
        return x_vfm + self.scale * w * delta * w_Moe, w


class VLMAdapterWeight(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=1):
        super().__init__()
        
        # Linear projection for gating input
        self.proj = nn.Linear(dim, dim)
        
        # LoRA-style MLP adapter (non-zero init to allow learning)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else lambda x: x
        self.up = nn.Linear(bottleneck_dim, dim)

        # Gating MLP from [VFM_proj, VLM]
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1)
        )

        self.scale = scale

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)

            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)

            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.bias)
            
            #zero init
            nn.init.zeros_(self.weight[-1].weight)
            nn.init.zeros_(self.weight[-1].bias)

    def forward(self, x_vfm, x_vlm, use_sigmoid=True):
        # Transform VFM via LoRA-style adapter
        delta = self.down(x_vfm)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up(delta)

        # Compute token-wise gate
        x_cat = torch.cat([self.proj(x_vfm), x_vlm], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                # [B, T, 1]

        # Residual injection into VLM
        return x_vlm, self.scale * w * delta

    def getweights(self, x_vfm, x_vlm):
        # Transform VFM via LoRA-style adapter
        delta = self.down(x_vfm)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up(delta)

        # Compute token-wise gate
        x_cat = torch.cat([self.proj(x_vfm), x_vlm], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                # [B, T, 1]

        # Residual injection into VLM
        return x_vlm + self.scale * w * delta, w

class VFMAdapterWeight(nn.Module):
    # def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=0.1):
    def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=0.1):
        super().__init__()
        
        # Projection for cross-model gating (VLM feature)
        self.proj = nn.Linear(dim, dim)

        # LoRA-style MLP adapter for VFM
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else nn.Identity()
        self.up = nn.Linear(bottleneck_dim, dim)

        # Token-wise gating from [x_vfm, projected x_vlm]
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1)  # token-wise scalar gate
        )

        self.scale = scale  # global residual scale (optional)

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)

            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)

            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.bias)

            # zero init
            nn.init.zeros_(self.weight[-1].weight)
            nn.init.zeros_(self.weight[-1].bias)

    def forward(self, x_vfm, x_vlm, use_sigmoid=True):
        # Adapt VFM features (delta)
        delta = self.down(x_vfm)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up(delta)

        # Compute token-wise gating weight using [x_vfm, projected x_vlm]
        x_cat = torch.cat([x_vfm, self.proj(x_vlm)], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)          

        # Inject back into VFM (residual)
        return x_vfm, self.scale * w * delta

    def getweights(self, x_vfm, x_vlm):
        # Adapt VFM features (delta)
        delta = self.down(x_vfm)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up(delta)

        # Compute token-wise gating weight using [x_vfm, projected x_vlm]
        x_cat = torch.cat([x_vfm, self.proj(x_vlm)], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                    # [B, T, 1]

        # Inject back into VFM (residual)
        return x_vfm + self.scale * w * delta, w


class VLMAdapterLightweight(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=1):
        super().__init__()
        
        # Linear projection for gating input
        self.proj = nn.Linear(dim, dim)
        
        # LoRA-style MLP adapter (non-zero init to allow learning)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else lambda x: x
        self.up = nn.Linear(bottleneck_dim, dim)

        # Gating MLP from [VFM_proj, VLM]
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1)
        )

        self.scale = scale

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)

            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)

            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.bias)
            
            #zero init
            nn.init.zeros_(self.weight[-1].weight)
            nn.init.zeros_(self.weight[-1].bias)

    def forward(self, x_vfm, x_vlm, use_sigmoid=True):
        # Transform VFM via LoRA-style adapter
        delta = self.down(x_vfm)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up(delta)

        # Compute token-wise gate
        x_cat = torch.cat([self.proj(x_vfm), x_vlm], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                # [B, T, 1]
        if use_sigmoid:
            w = torch.sigmoid(w)

        # Residual injection into VLM
        return x_vlm + self.scale * w * delta

    def getweights(self, x_vfm, x_vlm):
        # Transform VFM via LoRA-style adapter
        delta = self.down(x_vfm)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up(delta)

        # Compute token-wise gate
        x_cat = torch.cat([self.proj(x_vfm), x_vlm], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                # [B, T, 1]

        # Residual injection into VLM
        return x_vlm + self.scale * w * delta, w, self.scale * w * delta

class DeltaBlock(nn.Module):
    def __init__(self, dim, bottleneck_dim, dropout_rate=0.1):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else lambda x: x
        self.up = nn.Linear(bottleneck_dim, dim)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)

            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.bias)
    
    def forward(self, x):
        delta = self.down(x)
        delta = self.act(delta)
        delta = self.dropout(delta)
        delta = self.up(delta)

        return delta
        
class AdapterParral(nn.Module):
    def __init__(self, dim=1024, bottleneck_lr=8, bottleneck_hr=64, self_scale=0.1, borrow_scale=1, self_evolve=False):
        super().__init__()
        # lr module
        self.lr_adapter = DeltaBlock(dim, bottleneck_lr)
        
        # hr module
        self.proj = nn.Linear(dim, dim)
        self.hr_adapter = DeltaBlock(dim, bottleneck_hr)
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_hr * 2),
            nn.GELU(),
            nn.Linear(bottleneck_hr * 2, 1)
        )

        self.borrow_scale = borrow_scale
        self.self_scale = self_scale
        self.self_evolve = self_evolve

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)

            #zero init
            nn.init.zeros_(self.weight[-1].weight)
            nn.init.zeros_(self.weight[-1].bias)

    def freeze_lr(self):
        for param in self.lr_adapter.parameters():
            param.requires_grad = False
        
    def freeze_hr(self):
        for param in self.hr_adapter.parameters():
            param.requires_grad = False
        for param in self.proj.parameters():
            param.requires_grad = False
        for param in self.weight.parameters():
            param.requires_grad = False

    def unfreeze_lr(self):
        for param in self.lr_adapter.parameters():
            param.requires_grad = True        

    def unfreeze_hr(self):
        for param in self.hr_adapter.parameters():
            param.requires_grad = True
        for param in self.proj.parameters():
            param.requires_grad = True
        for param in self.weight.parameters():
            param.requires_grad = True

    def forward(self, x_self, x_borrow, use_stage2):
        # Transform VFM via LoRA-style adapter
        delta_self = self.lr_adapter(x_self)
        if use_stage2:
            if self.self_evolve:
                delta_borrow = self.hr_adapter(x_self)
            else:
                delta_borrow = self.hr_adapter(delta_borrow)
            x_cat = torch.cat([x_self, self.proj(x_borrow)], dim=-1)  # [B, T, 2C]
            w = self.weight(x_cat)         

            return x_self + self.self_scale * delta_self + self.borrow_scale * w * delta_borrow
        
        return x_self + self.self_scale * delta_self

class AdapterSeq1(nn.Module):
    def __init__(self, dim=1024, bottleneck_lr=8, bottleneck_hr=64, self_scale=0.1, borrow_scale=1):
        super().__init__()
        
        # lr module
        self.lr_adapter = DeltaBlock(dim, bottleneck_lr)
        self.self_scale = self_scale

    def freeze_lr(self):
        for param in self.lr_adapter.parameters():
            param.requires_grad = False

    def unfreeze_lr(self):
        for param in self.lr_adapter.parameters():
            param.requires_grad = True        

    def forward(self, x_self, holder=None):
        # Transform VFM via LoRA-style adapter
        delta_self = self.lr_adapter(x_self)
        return x_self + self.self_scale * delta_self


class AdapterSeq2(nn.Module):
    def __init__(self, dim=1024, bottleneck_lr=8, bottleneck_hr=64, self_scale=0.1, borrow_scale=1, self_evolve=False):
        super().__init__()
        # hr module
        self.hr_adapter = DeltaBlock(dim, bottleneck_hr)
        self.proj = nn.Linear(dim, dim)
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_hr * 2),
            nn.GELU(),
            nn.Linear(bottleneck_hr * 2, 1)
        )
        self.borrow_scale = borrow_scale
        self.self_evolve = self_evolve

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)

            #zero init
            nn.init.zeros_(self.weight[-1].weight)
            nn.init.zeros_(self.weight[-1].bias)

    def freeze_hr(self):
        for param in self.hr_adapter.parameters():
            param.requires_grad = False
        for param in self.proj.parameters():
            param.requires_grad = False
        for param in self.weight.parameters():
            param.requires_grad = False    

    def unfreeze_hr(self):
        for param in self.hr_adapter.parameters():
            param.requires_grad = True
        for param in self.proj.parameters():
            param.requires_grad = True
        for param in self.weight.parameters():
            param.requires_grad = True

    def forward(self, x_self, x_borrow):
        # Transform VFM via LoRA-style adapter
        if self.self_evolve:
            delta_borrow = self.hr_adapter(x_self)
        else:
            delta_borrow = self.hr_adapter(x_borrow)
        x_cat = torch.cat([x_self, self.proj(x_borrow)], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)         

        return x_self + self.borrow_scale * w * delta_borrow
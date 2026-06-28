import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
# from einops import rearrange
import einops

from .pruning.adapter import DeltaBlock, Compensator
from .pruning.mask import *

class SelfAttAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up   = nn.Linear(bottleneck_dim, dim)
        self.att = nn.MultiheadAttention(bottleneck_dim, 4, batch_first=True)
        
        
        with torch.no_grad():
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        
    def forward(self, x):
        x = self.down(x)
        x, _ = self.att(x, x, x)
        x = self.up(x)
        
        return x
    
class CrossAttAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64):
        super().__init__()
        self.down1 = nn.Linear(dim, bottleneck_dim)
        self.down2 = nn.Linear(dim, bottleneck_dim)
        self.up   = nn.Linear(bottleneck_dim, dim)
        self.att = nn.MultiheadAttention(bottleneck_dim, 4,batch_first=True)
        
        
        with torch.no_grad():
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        
    def forward(self, x, y):
        x = self.down1(x)
        y = self.down2(y)
        x, _ = self.att(x, y, y)
        x = self.up(x)
        
        return x

class ExpAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0, borrow_flag=False):
        '''
        borrow_flag: 
            True: borrow from another model
            False: self evolve
        '''
        super().__init__()
        # hr module
        self.adapter = DeltaBlock(dim, bottleneck_dim)
        self.scale = scale
        self.borrow_flag = borrow_flag
            

    def forward(self, x_self, x_borrow):
        # Transform VFM via LoRA-style adapter
        if self.borrow_flag is True:
            #borrowing
            delta = self.adapter(x_borrow)
        else:
            #self evolving
            delta = self.adapter(x_self)     

        return self.scale * delta

    
class GateAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0, borrow_flag=False):
        '''
        borrow_flag: 
            True: borrow from another model
            False: self evolve
        '''
        super().__init__()
        # hr module
        self.adapter = DeltaBlock(dim, bottleneck_dim)
        self.gate_layer = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1),
            nn.Sigmoid(),
        )
        self.scale = scale
        self.borrow_flag = borrow_flag
            

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.gate_layer[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.gate_layer[0].bias)
            
            #zero init
            nn.init.zeros_(self.gate_layer[2].weight)
            nn.init.zeros_(self.gate_layer[2].bias)

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

        #w [B, T, 1]
        w = self.gate_layer(x_cat)      

        return self.scale * w * delta

    def getfea(self, x_self, x_borrow):
        # Transform VFM via LoRA-style adapter
        if self.borrow_flag is True:
            #borrowing
            delta = self.adapter(x_borrow)
        else:
            #self evolving
            delta = self.adapter(x_self)
            
        #learning weight
        x_cat = torch.cat([x_self, x_borrow], dim=-1)  # [B, T, 2C]

        #w [B, T, 1]
        w = self.gate_layer(x_cat)      

        return self.scale * w * delta, w

class MSOP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=3//2, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=5//2, groups=channels)
        self.fuse  = nn.Conv2d(channels*3, channels, kernel_size=1)
    
    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x)
        x3 = self.conv2(x)

        x_cat = torch.cat([x1,x2,x3], dim=1)
        out = self.fuse(x_cat)
        
        return out

class MSGate(nn.Module):
    def __init__(self, dim, bottleneck_dim=64, scales=(1,3,5)):
        super().__init__()
        self.reduce = nn.Linear(dim * 2, bottleneck_dim * 2)
        self.msconv = MSOP(bottleneck_dim * 2)
        self.out = nn.Sequential(
            nn.Linear(bottleneck_dim*2, 1),
            nn.Sigmoid(),
        )

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.reduce.weight, a=math.sqrt(5))
            nn.init.zeros_(self.reduce.bias)
            #zero init
            nn.init.zeros_(self.out[0].weight)
            nn.init.zeros_(self.out[0].bias)
        
    
    def forward(self, x_cat):
        B, T, _ = x_cat.shape
        H = W = int(T ** 0.5)  
        feat = self.reduce(x_cat)           # [B,T,b]
        feat_2d = feat.transpose(1, 2).reshape(B, -1, H, W)

        multi = self.msconv(feat_2d)
        multi = multi.flatten(2).transpose(1, 2)  # [B,T,b]

        w = self.out(multi)  # [B,T,1]
        return w


class MSGateAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0, borrow_flag=False):
        '''
        borrow_flag: 
            True: borrow from another model
            False: self evolve
        '''
        super().__init__()
        # hr module
        self.adapter = DeltaBlock(dim, bottleneck_dim)
        self.gate_layer = MSGate(dim, bottleneck_dim)
        self.scale = scale
        self.borrow_flag = borrow_flag

    def forward(self, x_self, x_borrow):
    # def forward(self, x):
        # x_self = x
        # x_borrow = x
        # Transform VFM via LoRA-style adapter
        if self.borrow_flag is True:
            #borrowing
            delta = self.adapter(x_borrow)
        else:
            #self evolving
            delta = self.adapter(x_self)
            
        #learning weight
        x_cat = torch.cat([x_self, x_borrow], dim=-1)  # [B, T, 2C]

        #w [B, T, 1]
        w = self.gate_layer(x_cat)      

        return self.scale * w * delta

    def getfea(self, x_self, x_borrow):
        # Transform VFM via LoRA-style adapter
        if self.borrow_flag is True:
            #borrowing
            delta = self.adapter(x_borrow)
        else:
            #self evolving
            delta = self.adapter(x_self)
            
        #learning weight
        x_cat = torch.cat([x_self, x_borrow], dim=-1)  # [B, T, 2C]

        #w [B, T, 1]
        w = self.gate_layer(x_cat)      

        return self.scale * w * delta, w

class SingleGateAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0, borrow_flag=False):
        '''
        borrow_flag: 
            True: borrow from another model
            False: self evolve
        '''
        super().__init__()
        # hr module
        self.adapter = DeltaBlock(dim, bottleneck_dim)
        self.gate_layer = MSGate(dim//2, bottleneck_dim)
        self.scale = scale
        self.borrow_flag = borrow_flag

    def forward(self, x_self, x_borrow):
    # def forward(self, x):
        # x_self = x
        # x_borrow = x
        # Transform VFM via LoRA-style adapter
        if self.borrow_flag is True:
            #borrowing
            delta = self.adapter(x_borrow)
        else:
            #self evolving
            delta = self.adapter(x_self)
            
        #learning weight
        # x_cat = torch.cat([x_self, x_borrow], dim=-1)  # [B, T, 2C]

        #w [B, T, 1]
        w = self.gate_layer(x_self)      

        return self.scale * w * delta


class DynGateAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0, 
                 warmup_iters=2500, policy="mean_std", log_stats=True):
        '''
        borrow_flag: 
            True: borrow from another model
            False: self evolve

        policy: "entropy", "mean_std"
        '''
        super().__init__()
        # hr module
        self.adapter_self   = DeltaBlock(dim, bottleneck_dim)
        self.adapter_borrow = DeltaBlock(dim, bottleneck_dim)
        
        self.gate_layer_self = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1),
            nn.Sigmoid(),
        )

        self.gate_layer_borrow = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1),
            nn.Sigmoid(),
        )

        self.scale = scale

        # Initialization
        with torch.no_grad():
            for gate in [self.gate_layer_self, self.gate_layer_borrow]:
                nn.init.kaiming_uniform_(gate[0].weight, a=math.sqrt(5))
                nn.init.zeros_(gate[0].bias)
                nn.init.zeros_(gate[2].weight)
                nn.init.zeros_(gate[2].bias)

        # --- stats buffers (persist in state_dict, move with .to(device)) ---self.log_stats = True                 
        # turn off after freeze
        
        # --- accumulators (float32) ---
        self.register_buffer("w_sum_self",   torch.zeros((), dtype=torch.float32))
        self.register_buffer("w_sq_self",    torch.zeros((), dtype=torch.float32))
        self.register_buffer("H_sum_self",   torch.zeros((), dtype=torch.float32))

        self.register_buffer("w_sum_borrow", torch.zeros((), dtype=torch.float32))
        self.register_buffer("w_sq_borrow",  torch.zeros((), dtype=torch.float32))
        self.register_buffer("H_sum_borrow", torch.zeros((), dtype=torch.float32))

        # --- counters (int64) ---
        self.register_buffer("count",     torch.zeros((), dtype=torch.long))
        self.register_buffer("iter_ctr",  torch.zeros((), dtype=torch.long))

        # --- config/state as buffers (so they’re saved with state_dict) ---
        self.log_stats = log_stats
        self.warmup_iters = warmup_iters
        self.policy = policy

        self.register_buffer("act_self", torch.tensor(True, dtype=torch.bool))
        self.register_buffer("act_borrow", torch.tensor(True, dtype=torch.bool))

    @torch.no_grad()
    def binary_entropy(self, w, eps=1e-8):
        w = w.clamp(eps, 1 - eps)
        return -(w * w.log() + (1 - w) * (1 - w).log())
        
    @torch.no_grad()
    def _update_stats(self, w_self, w_borrow):
        # w_*: [B, T, 1] or [B, T]
        ws = w_self.reshape(-1).detach()
        wb = w_borrow.reshape(-1).detach()
        
        Hs = self.binary_entropy(ws)
        Hb = self.binary_entropy(wb)

        self.w_sum_self   += ws.sum()
        self.w_sq_self    += (ws * ws).sum()
        self.H_sum_self   += Hs.sum()

        self.w_sum_borrow += wb.sum()
        self.w_sq_borrow  += (wb * wb).sum()
        self.H_sum_borrow += Hb.sum()

        self.count += ws.numel()


    @torch.no_grad()
    def _means_stds(self):
        # return dict with mean w, std w, mean entropy for both branches
        def pack(sum_w, sum_sq, sum_H, cnt):
            if cnt.item() == 0:
                return dict(mean_w=0.0, std_w=0.0, mean_H=0.0, n=0)
            mean_w = (sum_w / cnt).item()
            var_w  = (sum_sq / cnt - (sum_w / cnt) ** 2).clamp(min=0).item()
            mean_H = (sum_H / cnt).item()
            return dict(mean_w=mean_w, std_w=var_w ** 0.5, mean_H=mean_H, n=int(cnt.item()))
        stats_s = pack(self.w_sum_self,   self.w_sq_self,   self.H_sum_self,   self.count)
        stats_b = pack(self.w_sum_borrow, self.w_sq_borrow, self.H_sum_borrow, self.count)
        return stats_s, stats_b

    @torch.no_grad()
    def reset_stats(self):
        self.w_sum_self.zero_();   self.w_sq_self.zero_();   self.H_sum_self.zero_();   
        self.w_sum_borrow.zero_(); self.w_sq_borrow.zero_(); self.H_sum_borrow.zero_(); 
        self.count.zero_(); self.iter_ctr.zero_()

    
    @torch.no_grad()
    def finalize(self, margin=0.05, beta=0.0):
        """
        Decide one winner per layer and hard-wire branch_mode.
        policy="entropy": choose lower mean entropy; tie-break by mean_w, then std_w.
        policy="mean_std": choose higher mean_w if > margin; else lower std_w; then lower mean_H.
        beta>0 enables Score = mean_w - beta * mean_H (optional).
        """
        stats_s, stats_b = self._means_stds()

        def choose_by_entropy(sb, bb):
            print("#####################")
            print("Drop use entropy")
            
            # primary: mean entropy
            if bb["mean_H"] < sb["mean_H"]:
                self.act_self.fill_(False)
                self.act_borrow.fill_(True)
            else:
                self.act_self.fill_(True)
                self.act_borrow.fill_(False)

        # def choose_by_mean_then_std(sb, bb):
        #     if (bb["mean_w"] - sb["mean_w"]) > margin: return "borrow"
        #     if (sb["mean_w"] - bb["mean_w"]) > margin: return "self"
        #     # close → stability
        #     if bb["std_w"] < sb["std_w"]:
                # self.act_self.fill_(False)
                # self.act_borrow.fill_(True)
        #     else:
                # self.act_self.fill_(True)
                # self.act_borrow.fill_(False)

        def choose_by_mean_then_std(sb, bb, lam=1.0):
            print("#####################")
            print("Drop use mean")
            
            s_self = sb["mean_w"] - lam * sb["std_w"]
            s_borr = bb["mean_w"] - lam * bb["std_w"]
            if s_borr > s_self:
                self.act_self.fill_(False)
                self.act_borrow.fill_(True)
            else:
                self.act_self.fill_(True)
                self.act_borrow.fill_(False)

        
        if self.policy == "entropy":
            self.branch_mode = choose_by_entropy(stats_s, stats_b)
        else:
            self.branch_mode = choose_by_mean_then_std(stats_s, stats_b)

        # stop logging once we’ve chosen
        self.log_stats = False
    
    def forward(self, x_self, x_borrow):
        #learning weight
        x_cat = torch.cat([x_self, x_borrow], dim=-1)  # [B, T, 2C]

        # mixing per current mode
        if self.act_self.item():
            delta_self   = self.adapter_self(x_self)
            w_self   = self.gate_layer_self(x_cat)
        else:
            delta_self = 0
            w_self = 0

        if self.act_borrow.item():
            delta_borrow = self.adapter_borrow(x_borrow)
            w_borrow = self.gate_layer_borrow(x_cat)
        else:
            delta_borrow = 0
            w_borrow = 0
            
 
        # warm-up logging
        if self.training and self.log_stats:
            self._update_stats(w_self, w_borrow)
            self.iter_ctr += 1
            
            # optional: auto-finalize right here
            if self.iter_ctr.item() >= self.warmup_iters and self.act_self.item() and self.act_borrow.item():
                self.finalize()  # uses accumulated stats

        return self.scale * (w_self * delta_self + w_borrow * delta_borrow)

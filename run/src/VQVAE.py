

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import RelaxedOneHotCategorical, Normal, Categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.manifold import TSNE
import os
import seaborn as sns
from timm.scheduler import TanhLRScheduler
from torchsummary import summary

def mish(x):
    return x * torch.tanh(F.softplus(x))

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)
    

class Upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channel, channel, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv1d(channel, in_channel, kernel_size=3, padding=1)

    def forward(self, inp):
        x = self.conv_1(inp)
        x = mish(x)
        x = self.conv_2(x)
        x = x + inp
        return mish(x)
    


class Decoder(nn.Module):
    def __init__(
        self, in_feat_dim, out_feat_dim, hidden_dim=128, num_res_blocks=0,
    ):
        super().__init__()
        self.out_feat_dim = out_feat_dim 
        blocks = [nn.Conv1d(in_feat_dim, hidden_dim, kernel_size=3, padding=1),nn.BatchNorm1d(hidden_dim), Mish()]
        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))
            blocks.append(nn.BatchNorm1d(hidden_dim))

        blocks.extend([
                Upsample(),
                nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim // 2),
                Mish(),
                nn.Conv1d(hidden_dim // 2, out_feat_dim, kernel_size=3, padding=1),
        ])
 
        blocks.append(nn.Tanh())       
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        x = x.float()
        return self.blocks(x)
    


class Encoder(nn.Module):

    def __init__(self, in_feat_dim, codebook_dim, hidden_dim=128, num_res_blocks=0):
        super().__init__()
        blocks = [
            nn.Conv1d(in_feat_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1),
            Mish(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            Mish(),
        ]

        for _ in range(num_res_blocks):
            blocks.append(ResBlock(hidden_dim, hidden_dim // 2))
            blocks.append(nn.BatchNorm1d(hidden_dim))
            
        blocks.append(nn.Conv1d(hidden_dim, codebook_dim, kernel_size=1))
        blocks.append(nn.BatchNorm1d(codebook_dim))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.float()
        return self.blocks(x)
    

class VQCodebook(nn.Module):
    def __init__(self,codebook_dim = 16, codebook_slots = 32,temperature = 0.5):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.codebook_slots = codebook_slots
        self.codebook = nn.Embedding(self.codebook_slots,self.codebook_dim)
        self.temperature = temperature
        self.log_slots_const = np.log(self.codebook_slots)

    def ze_to_zq(self,ze,soft=True):
        bs, feat_dim, w = ze.shape
        assert feat_dim == self.codebook_dim
        ze = ze.permute(0, 2, 1).contiguous()
        
        z_e_flat = ze.view(bs * w, feat_dim)
        codebook = self.codebook.weight
        codebook_sqr = torch.sum(codebook ** 2, dim=1)
        z_e_flat_sqr = torch.sum(z_e_flat ** 2, dim=1, keepdim=True)
        distances = torch.addmm(
            codebook_sqr + z_e_flat_sqr, z_e_flat, codebook.t(), alpha=-2.0, beta=1.0
        )
        if soft is True:
            dist = RelaxedOneHotCategorical(self.temperature, logits=-distances)
            soft_onehot = dist.rsample()
            hard_indices = torch.argmax(soft_onehot, dim=1).view(bs, w)
            z_q = (soft_onehot @ codebook).view(bs, w, feat_dim)
            
            # entropy loss
            KL = dist.probs * (dist.probs.add(1e-9).log() + self.log_slots_const)
            KL = KL.view(bs, w, self.codebook_slots).sum(dim=(1,2)).mean()
            
            # probability-weighted commitment loss    
            commit_loss = (dist.probs.view(bs, w, self.codebook_slots) * distances.view(bs, w, self.codebook_slots)).sum(dim=(1,2)).mean()
        else:
            with torch.no_grad():
                dist = Categorical(logits=-distances)
                hard_indices = dist.sample().view(bs, w)
                hard_onehot = (
                    F.one_hot(hard_indices, num_classes=self.codebook_slots)
                    .type_as(codebook)
                    .view(bs * w , self.codebook_slots)
                )
                z_q = (hard_onehot @ codebook).view(bs, w, feat_dim)
                
                # entropy loss
                KL = dist.probs * (dist.probs.add(1e-9).log() + np.log(self.codebook_slots))
                KL = KL.view(bs, w, self.codebook_slots).sum(dim=(1,2)).mean()

                commit_loss = 0.0

        z_q = z_q.permute(0, 2, 1)

        return z_q, hard_indices, KL, commit_loss

    def lookup(self, ids: torch.Tensor):
        codebook = self.codebook.weight
        #print( F.embedding(ids, codebook).shape)
        return F.embedding(ids, codebook).permute(0, 2, 1)

    def quantize(self, z_e, soft=False):
        with torch.no_grad():
            z_q, indices, _, _ = self.ze_to_zq(z_e, soft=soft)
        return z_q, indices

    def quantize_indices(self, z_e, soft=False):
        with torch.no_grad():
            _, indices, _, _ = self.ze_to_zq(z_e, soft=soft)
        return indices

    def forward(self, z_e, soft=True):
        z_q, indices, kl, commit_loss = self.ze_to_zq(z_e, soft)
        return z_q, indices, kl, commit_loss



class VQVAE(pl.LightningModule):

    def __init__(self,in_feat_dim=128,out_feat_dim=256,hidden_dim=128,
                 num_res_blocks=0,
                 codebook_dim=16,  
                codebook_slots= 32,    
                epochs = 10,
                KL_coeff = 0.001,
                CL_coeff = 0.0001):
        super().__init__()
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.hidden_dim = hidden_dim
        self.num_res_blocks = num_res_blocks
        self.codebook_dim = codebook_dim
        self.codebook_slots = codebook_slots
        self.KL_coeff = KL_coeff
        self.CL_coeff = CL_coeff
        self.Encoder = Encoder(in_feat_dim = self.in_feat_dim, codebook_dim = self.codebook_dim, hidden_dim=self.hidden_dim, num_res_blocks=self.num_res_blocks)
        self.Decoder = Decoder(in_feat_dim = self.codebook_dim, out_feat_dim = self.in_feat_dim, hidden_dim=self.hidden_dim, num_res_blocks=self.num_res_blocks)
        self.codebook = VQCodebook(codebook_dim = self.codebook_dim , codebook_slots = self.codebook_slots)
        self.codebook.codebook.weight.data.normal_()

    def forward(self,x):
        ze = self.Encoder(x)
        zq,indices,KL_loss,commit_Loss = self.codebook.ze_to_zq(ze,soft=True)
        x_hat = self.Decoder(zq) 
        return x_hat,ze,zq,indices,KL_loss,commit_Loss
        
    def on_train_start(self):
        self.code_count = torch.zeros(self.codebook.codebook_slots, device=self.device, dtype=torch.float64)
        self.codebook_resets = 0
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-4)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x,_ = batch
        x = x.float()
        x_hat,ze,zq,indices,KL_loss,commit_Loss = self(x)
        
        dims = np.prod(x_hat.shape[1:]) 
        recon_loss = F.mse_loss(x,x_hat,reduction='none').sum(dim=(1,2)).mean()
        loss = recon_loss/dims + self.KL_coeff * KL_loss/dims + self.CL_coeff * commit_Loss/dims
        
        indices_onehot = F.one_hot(indices, num_classes=self.codebook.codebook_slots).float()
        self.code_count = self.code_count + indices_onehot.sum(dim=(0, 1))
        
        if batch_idx > 0 and batch_idx % 25 == 0:
            self.reset_least_used_codeword()

        #self.log("recon_loss", recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss",loss,on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return recon_loss
        
    def decode(self, z_q):
        with torch.no_grad():
            recon = self.Decoder(z_q)
        return recon
        
    def encode(self, x):
        with torch.no_grad():
            z_e = self.Encoder(x)
        return z_e
        
    def quantize(self, z_e):
        z_q, _ = self.codebook.quantize(z_e)
        return z_q
        
    def reconstruct(self, x):
        return self.decode(self.quantize(self.encode(x)))
    
        
    @torch.no_grad()
    def reset_least_used_codeword(self):
        max_count, most_used_code = torch.max(self.code_count, dim=0)
        frac_usage = self.code_count / max_count
        z_q_most_used = self.codebook.lookup(most_used_code.view(1, 1)).squeeze()
        min_frac_usage, min_used_code = torch.min(frac_usage, dim=0)
        reset_factor = (1*(self.codebook_resets+1))
        
        if min_frac_usage < 0.03:
            #print(f'reset code {min_used_code} and codebook resetcount is {self.codebook_resets+1}')
            moved_code = z_q_most_used + torch.randn_like(z_q_most_used) / reset_factor
            self.codebook.codebook.weight.data[min_used_code] = moved_code
        self.code_count = torch.zeros_like(self.code_count, device=self.device)
        self.codebook_resets += 1    


if __name__ == "__main__":
    

    input_feat_dim=2
    enc_hidden_dim=16
    dec_hidden_dim=32
    num_res_blocks=2
    codebook_dim=128 
    codebook_slots= 64
    
    VQVAE_model = VQVAE(
        in_feat_dim=input_feat_dim,
        hidden_dim=enc_hidden_dim,
        out_feat_dim=dec_hidden_dim,
        num_res_blocks=num_res_blocks,
        codebook_dim=codebook_dim, 
        codebook_slots= codebook_slots
    ).to('cuda:0')
    print(summary(VQVAE_model,(2,128),16,device='cuda'))
    print(VQVAE_model)
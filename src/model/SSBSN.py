
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
from . import regist_model
from .SSBSNl import SSBSNl

@regist_model
class SSBSN(nn.Module):
    def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16, 
                    bsn='SSBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9, mode='ss', f_scale=2, ss_exp_factor=1.):
        super().__init__()

        # network hyper-parameters
        self.pd_a    = pd_a
        self.pd_b    = pd_b
        self.pd_pad  = pd_pad
        self.R3      = R3
        self.R3_T    = R3_T
        self.R3_p    = R3_p
        self.mode    = mode
        self.f_scale = f_scale
        
        # define network
        if bsn == 'SSBSNl':
            self.bsn = SSBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module, mode, f_scale, ss_exp_factor)
        else:
            raise NotImplementedError('bsn %s is not implemented'%bsn)

    def forward(self, img, pd=None):
        # default pd factor is training factor (a)
        if pd is None: pd = self.pd_a

        # do PD
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            if self.pd_pad >0:
                p = self.pd_pad
                pd_img = F.pad(img, (p,p,p,p))
            else:
                pd_img=img
        
        # forward blind-spot network
        pd_img_denoised = self.bsn(pd_img)

        # do inverse PD
        if pd > 1:
            img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        else:
            if self.pd_pad >0:
                p = self.pd_pad
                img_pd_bsn = pd_img_denoised[:,:,p:-p,p:-p]
            else:
                img_pd_bsn = pd_img_denoised        

        return img_pd_bsn

    def denoise(self, x):
        '''
        Denoising process for inference.
        '''
        b,c,h,w = x.shape

        # pad images for PD process
        if h % self.pd_b != 0:
            x = F.pad(x, (0, 0, 0, self.pd_b - h%self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            x = F.pad(x, (0, self.pd_b - w%self.pd_b, 0, 0), mode='constant', value=0)

        # forward PD-BSN process with inference pd factor
        img_pd_bsn = self.forward(img=x, pd=self.pd_b)

        # Random Replacing Refinement
        if not self.R3:
            ''' Directly return the result (w/o R3) '''
            return img_pd_bsn[:,:,:h,:w]
        else:
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = x[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p,p,p,p), mode='reflect')
                if self.pd_pad == 0:
                    denoised[..., t] = self.bsn(tmp_input)
                else:
                    denoised[..., t] = self.bsn(tmp_input)[:,:,p:-p,p:-p]

            return torch.mean(denoised, dim=-1)
            
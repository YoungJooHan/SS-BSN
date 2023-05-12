import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SSBlockNaive(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)
        self.conv1_act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=stride, dilation=stride)
        self.conv2_act = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def _get_ff(self, x):
        x = self.conv1_act(self.conv1(x))
        x = self.conv2_act(self.conv2(x))
        x = self.conv3(x)
        return x

    def forward(self, x):
        return x + self._get_ff(x)

class SSBlock(SSBlockNaive):
    def __init__(self, stride, in_ch, f_scale=4, ss_exp_factor=1.):
        super().__init__(stride, in_ch)
        self.embed_size = int(in_ch * ss_exp_factor)
        
        self.wqk = nn.Parameter(torch.zeros(size=(in_ch, self.embed_size)))
        self.wqk.requires_grad = True
        nn.init.xavier_uniform_(self.wqk.data, gain=1.414)             

        self.stride = stride
        self.f_scale = f_scale

    def _pixel_unshuffle(self, x, c, f):
        x = rearrange(x, 'b c h w -> b 1 (c h) w')
        x = F.pixel_unshuffle(x, f)
        x = rearrange(x, 'b k (c h) w -> b (k c) h w', c=c)
        return x

    def _pixel_shuffle(self, x, c, f):
        x = rearrange(x, 'b (f c) h w -> b f (c h) w', f=f**2, c=c)
        x = F.pixel_shuffle(x, f)
        x = rearrange(x, 'b f (c h) w -> b (f c) h w', f=1, c=c)
        return x

    def _pad_for_shuffle(self, x, f):
        _,_,h,w = x.shape
        pad_h = 0
        pad_w = 0
        if h % f != 0:
            pad_h = f - h%f
            x = F.pad(x, (0, 0, 0, pad_h), mode='constant', value=0)
        if w % f != 0:
            pad_w = f - w%f
            x = F.pad(x, (0, pad_w, 0, 0), mode='constant', value=0)
        return x, pad_h, pad_w 

    def _get_attention(self, x, f):
        _,c,_,_ = x.shape 
        xx = F.layer_norm(x, x.shape[-3:]) # layer normalization

        xx, ph, pw = self._pad_for_shuffle(xx, f)
        xx = self._pixel_unshuffle(xx, c, f)
        xx = rearrange(xx, 'b (f c) h w -> (b f) c h w', c=c, f=f**2)

        v, ph, pw = self._pad_for_shuffle(x, f)
        v = self._pixel_unshuffle(v, c, f)
        v = rearrange(v, 'b (f c) h w -> (b f) c h w', c=c, f=f**2)

        b,_,sh,sw = xx.shape

        # embed by wqk
        qk = rearrange(xx, 'b c h w -> (b h w) c')
        v = rearrange(v, 'b c h w -> (b h w) c')
        qk = torch.mm(qk, self.wqk)

        # process per image
        qk = rearrange(qk, '(b h w) k -> b (h w) k', b=b, h=sh, w=sw)
        v = rearrange(v, '(b h w) k -> b (h w) k', b=b, h=sh, w=sw)

        # get cosine similarity
        qk_norm = torch.linalg.norm(qk, dim=-1).unsqueeze(dim=-1) + 1e-8
        qk = qk / qk_norm

        attn = torch.bmm(qk, torch.transpose(qk, 1, 2)) 
        attn += 1.
        attn /= self.embed_size ** 0.5
        attn = F.softmax(attn, dim=-1) 
    
        out = torch.bmm(attn, v)
        out = rearrange(out, 'b (h w) e -> b e h w', b=b, h=sh, w=sw)
        out = rearrange(out, '(b f) c h w -> b (f c) h w', f=f**2,c=c)
        out = self._pixel_shuffle(out, c, f)
        if ph > 0:
            out =  out[:,:,:-ph,:]
        if pw > 0:
            out = out[:,:,:,:-pw]
        return out

    def forward(self, x):
        f = self.f_scale * self.stride
        return x + self._get_ff(x + self._get_attention(x, f))
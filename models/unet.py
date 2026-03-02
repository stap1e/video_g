import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class SpaceTimeFactorizedAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        
        # Spatial Attention
        # Merge batch and frames: (b*f, c, h, w)
        x_in = rearrange(x, 'b c f h w -> (b f) c h w')
        
        qkv = self.to_qkv(x_in).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)

        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        out = rearrange(out, 'b h (x y) c -> b (h c) x y', x=h, y=w)
        out = self.to_out(out)
        
        # Restore shape
        out = rearrange(out, '(b f) c h w -> b c f h w', b=b)
        
        return out + x

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, max_time_steps=100):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim)
        
        # Relative position bias
        self.max_time_steps = max_time_steps
        self.rel_pos_emb = nn.Parameter(torch.randn(2 * max_time_steps - 1, heads))

    def forward(self, x):
        b, c, f, h, w = x.shape
        
        # Temporal Attention
        # Group spatial dimensions: (b*h*w, f, c)
        x_in = rearrange(x, 'b c f h w -> (b h w) f c')
        
        qkv = self.to_qkv(x_in).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b f (h c) -> b h f c', h=self.heads), qkv)

        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        
        # Relative position bias
        # Distance matrix
        indices = torch.arange(f, device=x.device)
        distance = indices[None, :] - indices[:, None]
        distance = distance + self.max_time_steps - 1
        distance = distance.clamp(0, 2 * self.max_time_steps - 2)
        
        rel_bias = self.rel_pos_emb[distance].permute(2, 0, 1) # (heads, f, f)
        sim = sim + rel_bias[None, ...]

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        out = rearrange(out, 'b h f c -> b f (h c)')
        out = self.to_out(out)
        
        out = rearrange(out, '(b h w) f c -> b c f h w', b=b, h=h, w=w)
        
        return out + x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Conv3d(dim, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
            nn.Conv3d(dim_out, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        )

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            h = h * (scale + 1) + shift

        h = self.block2(h)

        return h + self.res_conv(x)

class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose3d(dim, dim, kernel_size=(1, 4, 4), stride=(1, 2, 2), padding=(0, 1, 1))
    def forward(self, x):
        return self.conv(x)

class Unet3D(nn.Module):
    def __init__(self, dim, dim_mults=(1, 2, 4, 8), channels=3, out_dim=None, time_steps=16):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else channels
        self.time_steps = time_steps

        # Determine dimensions
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Initial convolution
        self.conv_in = nn.Conv3d(channels, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        # Downsampling
        for ind, (d_in, d_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            
            self.downs.append(nn.ModuleList([
                ResnetBlock(d_in, d_out, time_emb_dim=dim * 4),
                ResnetBlock(d_out, d_out, time_emb_dim=dim * 4),
                SpaceTimeFactorizedAttention(d_out),
                TemporalAttention(d_out, max_time_steps=time_steps),
                Downsample(d_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim * 4)
        self.mid_attn_spatial = SpaceTimeFactorizedAttention(mid_dim)
        self.mid_attn_temporal = TemporalAttention(mid_dim, max_time_steps=time_steps)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim * 4)

        # Upsampling
        for ind, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)
            # d_in: current larger dim (from mid or up), d_out: next smaller dim
            # Skip connection is d_in. Input to block is d_in (from prev up) + d_in (skip) = 2*d_in?
            # Wait. 
            # downs: (64->128), (128->256), (256->512)
            # h saves: 128, 256, 512.
            # mid: 512.
            # up 1: need 512->256. Skip is 512.
            # Input to up 1 is 512 (from mid). Concat with skip (512) = 1024.
            # So in_channels = d_in * 2. Out channels = d_out.
            # Correct logic:
            # We are iterating reversed in_out: (256, 512), (128, 256), (64, 128)
            # d_in=256, d_out=512. This is confusing.
            # Let's rename.
            # in_out entries are (small, big).
            # reversed entries are (small, big).
            # We want to go big -> small.
            
            # d_small = d_in, d_big = d_out.
            # Input to block is d_big (from lower layer) + d_big (skip connection) = 2 * d_big.
            # Output of block is d_small.
            
            d_small, d_big = d_in, d_out
            
            self.ups.append(nn.ModuleList([
                ResnetBlock(d_big * 2, d_small, time_emb_dim=dim * 4),
                ResnetBlock(d_small, d_small, time_emb_dim=dim * 4),
                SpaceTimeFactorizedAttention(d_small),
                TemporalAttention(d_small, max_time_steps=time_steps),
                Upsample(d_small) if not is_last else nn.Identity()
            ]))

        self.final_res_block = ResnetBlock(dim, dim, time_emb_dim=dim * 4)
        self.final_conv = nn.Conv3d(dim, self.out_dim, 1)

    def forward(self, x, time, cond=None):
        # x: (b, c, f, h, w)
        t = self.time_pos_emb(time)
        t = self.mlp(t)

        x = self.conv_in(x)
        h = []

        # Down
        for block1, block2, attn_s, attn_t, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn_s(x)
            x = attn_t(x)
            h.append(x) # Save for skip
            x = downsample(x)

        # Mid
        x = self.mid_block1(x, t)
        x = self.mid_attn_spatial(x)
        x = self.mid_attn_temporal(x)
        x = self.mid_block2(x, t)

        # Up
        for block1, block2, attn_s, attn_t, upsample in self.ups:
            skip = h.pop()
            x = torch.cat((x, skip), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn_s(x)
            x = attn_t(x)
            x = upsample(x)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

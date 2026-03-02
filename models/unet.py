import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math, sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from models.utils import init_weights

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

class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs
    
class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.params = params
        self.is_batchnorm = self.params['is_batchnorm']
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        
        self.conv1 = UnetConv3(self.in_chns, self.ft_chns[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(self.ft_chns[0], self.ft_chns[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(self.ft_chns[1], self.ft_chns[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(self.ft_chns[2], self.ft_chns[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(self.ft_chns[3], self.ft_chns[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        
        center = self.center(maxpool4)
        center = self.dropout(center)
        
        return [conv1, conv2, conv3, conv4, center]

class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2, deep=False):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        self.cat_feature = torch.cat([outputs1, outputs2], 1)
        if deep:
            return self.conv(self.cat_feature), self.cat_feature
        return self.conv(self.cat_feature)
    
class Decoder_wtcls(nn.Module):
    def __init__(self, params, in_channels=3, is_batchnorm=True):
        super(Decoder_wtcls, self).__init__()
        # self.is_batchnorm = is_batchnorm 
        self.params = params
        self.is_batchnorm = self.params['is_batchnorm']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        # upsampling
        self.up_concat4 = UnetUp3_CT(self.ft_chns[4], self.ft_chns[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(self.ft_chns[3], self.ft_chns[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(self.ft_chns[2], self.ft_chns[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(self.ft_chns[1], self.ft_chns[0], is_batchnorm)

        # final conv (without any concat)
        self.dropout = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
                
    def forward(self, features, deep=False, consist=False):
        conv1, conv2, conv3, conv4, center = features[:]
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2, deep=deep)
        if deep:
            up1, feature_deep = up1[0], up1[1]
        up1 = self.dropout(up1)
        if deep and not consist:
            return up1, feature_deep
        elif deep and consist:
            return center, up1, feature_deep
        elif consist:
            return center, up1
        return up1
    
class unet_3D(nn.Module):
    def __init__(self, in_chns, class_num, deep=False):
        super(unet_3D, self).__init__()
        params = {'in_chns': in_chns,
                  'is_batchnorm': True,
                  'feature_scale': 4,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.channels = in_chns
        self.deep = deep
        self.params = params
        self.encoder = Encoder(self.params)
        self.decoder = Decoder_wtcls(self.params)
        self.classifier = nn.Conv3d(self.params['feature_chns'][0], class_num, 1)

    def forward(self, inputs, t=None, comp_drop=None, feature_need=False, consist=False):
        features = self.encoder(inputs)
        if comp_drop != None:
            for i in range(0, len(features)):
                features[i] = features[i] * comp_drop[i].unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
            out = self.decoder(features)
            return out
        feature_final = self.decoder(features, deep=self.deep, consist=consist)
        if not self.deep and consist:
            center, feature_final = feature_final[0], feature_final[1]
        if self.deep and not consist:
            feature_final, deep_feature = feature_final[0], feature_final[1]
        elif self.deep and consist:
            center, feature_final, deep_feature = feature_final[0], feature_final[1], feature_final[2]
        # Add final convolution to match input channels (3)
        final_conv = nn.Conv3d(feature_final.size(1), 3, kernel_size=1).to(feature_final.device)
        output = final_conv(feature_final)
        return output

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p

    def lock_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

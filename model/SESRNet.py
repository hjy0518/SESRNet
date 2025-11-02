import numpy as np

import torch
import torch.nn as nn
from model.smt import smt_t
import torch.nn.functional as F
import os
from einops import rearrange
from mamba_ssm import Mamba
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.cbr = conv1x1_bn_relu(dim,dim)
        self.lnorm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand  # Block expansion factor
        )

    def forward(self, x):
        x = self.cbr(x)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_flat = self.lnorm(x_flat)
        x_mamba = self.mamba(x_flat)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        return out


class AttenMFFT(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, ):
        super(AttenMFFT, self).__init__()
        self.num_heads = num_heads

        self.qkv1conv_1 = MambaLayer(dim)
        self.qkv1conv_3 = MambaLayer(dim)
        self.qkv1conv_5 = MambaLayer(dim)

        self.qm = MambaLayer(dim)
        self.km = MambaLayer(dim)
        self.vm = MambaLayer(dim)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperatured = nn.Parameter(torch.ones(num_heads, 1, 1))


        self.project_out = conv1x1(dim * 2, dim)


    def forward(self, x,d):
        b, c, h, w = x.shape
        q_s = self.qkv1conv_5(x)
        k_s = self.qkv1conv_3(x)
        v_s = self.qkv1conv_1(x)
        q_s = torch.fft.fft2(q_s.float())
        k_s = torch.fft.fft2(k_s.float())
        v_s = torch.fft.fft2(v_s.float())
        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = custom_complex_normalization(attn_s, dim=-1)
        attn_s = torch.abs(torch.fft.ifft2(attn_s))


        dq_s = self.qm(d)
        dk_s = self.km(d)
        dv_s = self.vm(d)
        dq_s = rearrange(dq_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        dk_s = rearrange(dk_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        dv_s = rearrange(dv_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        dq_s = torch.nn.functional.normalize(dq_s, dim=-1)
        dk_s = torch.nn.functional.normalize(dk_s, dim=-1)
        dattn_s = (dq_s @ dk_s.transpose(-2, -1)) * self.temperatured
        dattn_s = torch.softmax(dattn_s, dim=-1)

        dattn_s = torch.fft.fft2(dattn_s.float())

        outd = torch.abs(torch.fft.ifft2(dattn_s @ v_s))
        outr = attn_s @ dv_s

        outd = rearrange(outd, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # outd0 = rearrange(outd0, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        outr = rearrange(outr, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # outr0 = rearrange(outr0, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(torch.cat((outr, outd), 1))

        return out


class MambaAT(nn.Module):
    def __init__(self, dim, num_heads=2, bias=False, ):
        super(MambaAT, self).__init__()
        self.num_heads = num_heads

        self.qkv1conv_1 = MambaLayer(dim)
        self.qkv1conv_3 = MambaLayer(dim)
        self.qkv1conv_5 = MambaLayer(dim)


        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, bias=True),
            nn.Sigmoid())

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b, c, h, w = x.shape
        q_s = self.qkv1conv_5(x)
        k_s = self.qkv1conv_3(x)
        v_s = self.qkv1conv_1(x)

        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = torch.softmax(attn_s, dim=-1)


        outr = torch.abs(attn_s @ v_s)
        outr = rearrange(outr, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        return outr


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class SESRNet(nn.Module):
    def __init__(self):
        super(SESRNet, self).__init__()

        self.rgb = smt_t()
        self.Dec = Decode(64,64,64,64)
        self.f1 = FM(64)
        self.f2 = FM(128)
        self.f3 = FM(256)
        self.f4 = FM(512)
        self.fs = [self.f1,self.f2,self.f3,self.f4]
        self.sig = nn.Sigmoid()
    def forward(self, x):

        fuses = []
        B = x.shape[0]

        for j in range(4):
            patch_embed = getattr(self.rgb, f"patch_embed{j + 1}")
            block = getattr(self.rgb, f"block{j + 1}")
            norm = getattr(self.rgb, f"norm{j + 1}")
            x, H, W = patch_embed(x)
            for i, blk in enumerate(block):
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x,fuse = self.fs[j](x)
            fuses.append(fuse)


        pred1 = self.Dec(fuses[0], fuses[1], fuses[2], fuses[3], 352)


        return pred1,self.sig(pred1)
    def load_pre(self, pre_model):
        self.rgb.load_state_dict(torch.load(pre_model)["model"],strict=False)
        print(f"RGB loading pre_model ${pre_model}")


class FM(nn.Module):
    def __init__(self, dim):
        super(FM, self).__init__()
        self.att = Attention(dim)
        self.br = AttenMFFT(dim)
        self.conv2 = nn.Conv2d(dim, 64, kernel_size=1)
    def forward(self, r):
        out = self.att(r)
        out = self.br(out,r)
        out_r = out + r
        out = self.conv2(out)
        return out_r,out


class ME(nn.Module):
    def __init__(self, in1,in2,in3=None,in4=None):
        super(ME, self).__init__()
        if in3 !=None and in4!=None:
            self.fm = BasicConv2d(in1 + in2 + in3 + in4, in1,kernel_size=3,padding=1)
        elif in3 !=None and in4==None:
            self.fm = BasicConv2d(in1 + in2 + in3, in1, kernel_size=3, padding=1)
        else:
            self.fm = BasicConv2d(in1 + in2 + in1, in1,kernel_size=3,padding=1)
        self.att = Attention(in1)

    def forward(self, in1, in2=None, in3=None,in4=None):
        if in3 != None and in4!=None:
            in2 = F.interpolate(in2, size=in1.size()[2:],mode='bilinear')
            in3 = F.interpolate(in3, size=in1.size()[2:], mode='bilinear')
            in4 = F.interpolate(in4, size=in1.size()[2:], mode='bilinear')
            x = torch.cat((in1, in2, in3,in4), 1)
            out = self.fm(x)
        elif in3 != None and in4==None:
            in2 = F.interpolate(in2, size=in1.size()[2:],mode='bilinear')
            in3 = F.interpolate(in3, size=in1.size()[2:], mode='bilinear')
            x = torch.cat((in1, in2, in3), 1)
            out = self.fm(x)
        else:
            in2 = F.interpolate(in2, size=in1.size()[2:],mode='bilinear')
            x = torch.cat((in1, in2, in1), 1)
            out = self.fm(x)
        out = self.att(out)
        return out


class Attention(nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.atten_H = H(self.dim, self.dim)
        self.atten_W = W(self.dim, self.dim)
        self.sa = SpatialAttention()
        self.sa_conv = MambaLayer(1)
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x1 = self.atten_W(x)
        x2 = self.atten_H(x)

        out = x * x1 * x2
        sa = self.sa_conv(self.sa(out))
        sa = self.sig(sa)
        out = out.mul(sa)

        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class H(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(H, self).__init__()
        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        mip = max(8, inp // reduction)
        self.hm = MambaLayer(inp)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, rgb):
        x = rgb
        n, c, h, w = x.size()
        y= self.pool_h(x)
        y = self.hm(y)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        a_h = self.conv_h(y).sigmoid()

        return a_h

class W(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(W, self).__init__()
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))
        mip = max(8, inp // reduction)
        self.hm = MambaLayer(inp)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, rgb):
        x = rgb
        n, c, h, w = x.size()
        y = self.pool_w(x)
        y = self.hm(y)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        a_w = self.conv_w(y).sigmoid()


        return a_w


class Decode(nn.Module):
    def __init__(self, in1,in2,in3,in4):
        super(Decode, self).__init__()
        self.dim = in1
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up4 = nn.Sequential(
            nn.Conv2d(in_channels=in4*2, out_channels=in3, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(in3),
            nn.GELU(),
            self.upsample2
        )
        self.conv_up3 = nn.Sequential(
            nn.Conv2d(in_channels=in3*2, out_channels=in2, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(in2),
            nn.GELU(),
            self.upsample2
        )
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(in_channels=in2*2, out_channels=in1, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(in1),
            nn.GELU(),
            self.upsample2
        )
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(in_channels=in1*2, out_channels=in1, kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(in1),
            nn.GELU(),
            self.upsample2
        )

        self.upb4 = Block(in3)
        self.upb3 = Block(in2)
        self.upb2 = Block(in1)
        self.upb1 = Block(in1)


        self.p_1 = nn.Sequential(
            nn.Conv2d(in_channels=in1, out_channels=in1//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in1//2),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=in1//2, out_channels=1, kernel_size=3, padding=1, bias=True),
        )

        self.fm1 = ME(self.dim,self.dim,self.dim,self.dim)
        self.fm2 = ME(self.dim,self.dim,self.dim)
        self.fm3 = ME(self.dim,self.dim)

    def forward(self,x1,x2,x3,x4,s):


        br3 = self.fm3(x3,x4)
        br2 = self.fm2(x2, x3, x4)
        br1 = self.fm1(x1, x2, x3,x4)
        up4 = self.upb4(self.conv_up4(torch.cat((x4,x4),1)))
        up3 = self.upb3(self.conv_up3(torch.cat((br3,up4),1)))
        up2 = self.upb2(self.conv_up2(torch.cat((br2,up3),1)))
        up1 = self.upb1(self.conv_up1(torch.cat((br1,up2),1)))
        pred1 = self.p_1(up1)

        return pred1





class CropLayer(nn.Module):
    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        mip = min(8, in_planes // ratio)
        self.avg_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, mip, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(mip, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out = self.sigmoid(max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class sa_layer(nn.Module):
    def __init__(self, channel, groups=4):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = nn.Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = nn.Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)

        return out

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim // 4
        self.conv1 = nn.Conv2d(self.dim, self.dim, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(self.dim, self.dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.dim, self.dim, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(self.dim, self.dim, kernel_size=7, stride=1, padding=3)


        self.at1 = MambaAT(self.dim)
        self.at2 = MambaAT(self.dim)
        self.at3 = MambaAT(self.dim)
        self.at4 = MambaAT(self.dim)

        self.conv = BasicConv2d(dim*2,dim,kernel_size=3,padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        x1 = self.at1(self.conv1(x1))
        x2 = self.at2(self.conv2(x2))
        x3 = self.at3(self.conv3(x3))
        x4 = self.at4(self.conv4(x4))

        out = self.conv(torch.cat((x,x1,x2,x3,x4),1))
        return out


class DWConv_3(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)  # depthwise conv
        self.conv_end = nn.Conv2d(dim*2,dim,kernel_size=3,padding=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.conv_end(torch.cat((shortcut,x),1))
        return x


if __name__ == '__main__':
    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count

    # 模型设置
    model =  SESRNet().cuda()
    input = torch.randn(1, 3, 352, 352).cuda()

    # 计算FLOPs和参数
    flops = FlopCountAnalysis(model, input)
    params = parameter_count(model)

    # 打印结果
    print('=' * 50)
    print('Model: SESRNet')
    print(f'Input Size: {input.shape}')
    print(f'Parameters: {params[""]/1000000:.2f}M')
    print(f'FLOPs: {flops.total() / 1e9:.2f}G')
    print('=' * 50)
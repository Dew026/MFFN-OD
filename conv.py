# Ultralytics YOLO ðŸš€, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from typing import List
from torch import Tensor
# from ultralytics.nn.extra_modules.block import C2f_Faster
# from ultralytics.nn.modules.conv import InversePolarTransform, PolarTransform, Conv, AMMFAM

__all__ = ('Conv', 'Conv2', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv', 
           'PolarTransform', 'InversePolarTransform', 'OffsetNet', 'AMMFAM',
           'PatchEmbed', 'BasicStage', 'PatchMerging',
           'AttentionIteract', 'APartial_conv',
           'mixConv', 'splitConv', 'Identity')


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        
        # print(self.act(self.bn(self.conv(x))).shape)
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""

        # if x[1].shape[2] == x[1].shape[3]:
        #     return torch.cat((x[1], x[1]), self.d)
        # elif x[0].shape!=x[1].shape:
        #     if x[1].shape[2] != 640:
        #         x[1] = F.interpolate(x[1], size=(640, 640), mode='bilinear', align_corners=True)
        #         x[0] = F.interpolate(x[0], size=(1280, 320), mode='bilinear', align_corners=True)

        #     x = [x[0], torch.reshape(x[1], x[0].shape)]
        #     return torch.cat(x, self.d)
        # else:
        return torch.cat(x, self.d)

####################################################
class PolarTransform(nn.Module):
    def __init__(self, c1, c2, k1, k2, ratio, tmp, zoom):
        super(PolarTransform, self).__init__()
        self.k1 = k1*0.5
        self.k2 = k2*0.5
        self.ratio = ratio # 图像的h/w
        self.tmp = tmp # tmp=1保留w, tmp=0保留h
        self.zoom = zoom #控制由圆环图像缩放的大小

    def forward(self, image):
        # 极坐标变换
        polar_images = self.polar_transform(image)
        
        return polar_images

    def polar_transform(self, image):
        if isinstance(image, list) != True:
            # 获取图像尺寸
            bs, c, h, w = image.size()
            
            h, w = int(h*self.zoom), int(w*self.zoom)
    
            #
            self.center = [h//2, w//2]
            
            if self.tmp == 0:
                self.max_radius = int(self.k1*h)
                self.min_radius = int(self.k2*h)
                
                self.aim_height = int(h)
                self.aim_width = int(h/self.ratio)
                
            elif self.tmp == 1:
                self.max_radius = int(self.k1*w)
                self.min_radius = int(self.k2*w)
                
                self.aim_width = int(w)
                self.aim_height = int(w*self.ratio)
            
            # 构建极坐标网格
            theta = torch.linspace(0, 2 * torch.pi, self.aim_height).view(-1, 1).expand(-1, self.aim_width).to(image.device)
            radius = torch.linspace(0, self.max_radius-self.min_radius, self.aim_width).view(1, -1).expand(self.aim_height, -1).to(image.device)
            # 极坐标变换
            x = (radius+self.min_radius) * torch.cos(theta)/self.center[0]
            y = (radius+self.min_radius) * torch.sin(theta)/self.center[1]
            
            grid = torch.stack([x, y], dim=2).view(1, self.aim_height, self.aim_width, 2).to(image.device)
            grid = grid.repeat(bs,1,1,1)
            
            polar_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
            
        elif isinstance(image, list):
            
            # 获取图像尺寸
            offset = image[1]
            
            img = image[0]
            bs, c, h, w = img.size()
            
            h, w = int(h*self.zoom), int(w*self.zoom)
    
            #
            self.center = [h//2, w//2]
            
            if self.tmp == 0:
                self.max_radius = int(self.k1*h)
                self.min_radius = int(self.k2*h)
                
                self.aim_height = int(h)
                self.aim_width = int(h/self.ratio)
                
            elif self.tmp == 1:
                self.max_radius = int(self.k1*w)
                self.min_radius = int(self.k2*w)
                
                self.aim_width = int(w)
                self.aim_height = int(w*self.ratio)
            
            # 构建极坐标网格
            theta = torch.linspace(0, 2 * torch.pi, self.aim_height).view(-1, 1).expand(-1, self.aim_width).to(img.device)
            radius = torch.linspace(0, self.max_radius-self.min_radius, self.aim_width).view(1, -1).expand(self.aim_height, -1).to(img.device)
            # 极坐标变换
            x = (radius+self.min_radius) * torch.cos(theta)/self.center[0]
            y = (radius+self.min_radius) * torch.sin(theta)/self.center[1]
            
            grid = torch.stack([x, y], dim=2).view(1, self.aim_height, self.aim_width, 2).to(img.device)
            
            # print(grid.repeat(bs,1,1,1).shape, offset.shape)
            
            
            if grid.repeat(bs,1,1,1).shape!=offset.shape:
                grid_repeat = grid.repeat(bs,1,1,1)
                grid_repeat = F.interpolate(grid_repeat, scale_factor=(offset.shape[-2]/grid_repeat.shape[-2], 1), mode='bilinear')
                grid = grid_repeat + offset
            
            else:
                grid = grid.repeat(bs,1,1,1) + offset
            
            polar_image = F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)
            
        
        return polar_image

class InversePolarTransform(nn.Module):
    def __init__(self, c1, c2, k1, k2, ratio, tmp, zoom):
        super(InversePolarTransform, self).__init__()
        self.k1 = k1*0.5
        self.k2 = k2*0.5
        self.ratio = ratio # 图像的h/w
        self.tmp = tmp # tmp=1保留w, tmp=0保留h
        self.zoom = zoom
        

    def forward(self, polar_image):
        # 逆极坐标变换
        cartesian_images = self.inverse_polar_transform(polar_image)
        return cartesian_images

    def inverse_polar_transform(self, polar_image):
        # 获取图像尺寸
        if isinstance(polar_image, list) != True:
            bs, c, h, w = polar_image.size()
            
            h, w = int(h*self.zoom), int(w*self.zoom)
            #
            if self.tmp == 0:
                self.center = [h//2, h//2]
                
                self.max_radius = int(self.k1*h)
                self.min_radius = int(self.k2*h)
                
                self.aim_height = int(h)
                self.aim_width = self.aim_height
                
            elif self.tmp == 1:
                self.center = [w//2, w//2]
                
                self.max_radius = int(self.k1*w)
                self.min_radius = int(self.k2*w)
                
                self.aim_width = int(w)
                self.aim_height = self.aim_width
                
            # 构建笛卡尔坐标网格
            x = torch.linspace(-1, 1, self.aim_height).view(1, -1).expand(self.aim_height, -1).to(polar_image.device)
            y = torch.linspace(-1, 1, self.aim_height).view(-1, 1).expand(-1, self.aim_height).to(polar_image.device)
            
            x = x*self.center[0]
            y = y*self.center[1]
            # 将笛卡尔坐标转换为极坐标
            theta = (torch.atan2(y, x)-torch.pi)/(torch.pi)  # 计算角度
            
            theta[theta<-1] = theta[theta<-1]+2
            
            radius = ((torch.sqrt(x**2 + y**2)-self.min_radius)-(self.max_radius-self.min_radius)/2)/((self.max_radius-self.min_radius)/2) # 计算半径
            
            grid = torch.stack([radius, theta], dim=2).view(1, self.aim_height, self.aim_width, 2).to(polar_image.device)
            grid = grid.repeat(bs,1,1,1)
            cartesian_image = F.grid_sample(polar_image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        elif isinstance(polar_image, list):
            offset = polar_image[1]
            p_img = polar_image[0]
            
            bs, c, h, w = p_img.size()
            
            h, w = int(h*self.zoom), int(w*self.zoom)
            #
            if self.tmp == 0:
                self.center = [h//2, h//2]
                
                self.max_radius = int(self.k1*h)
                self.min_radius = int(self.k2*h)
                
                self.aim_height = int(h)
                self.aim_width = self.aim_height
                
            elif self.tmp == 1:
                self.center = [w//2, w//2]
                
                self.max_radius = int(self.k1*w)
                self.min_radius = int(self.k2*w)
                
                self.aim_width = int(w)
                self.aim_height = self.aim_width
                
            # 构建笛卡尔坐标网格
            x = torch.linspace(-1, 1, self.aim_height).view(1, -1).expand(self.aim_height, -1).to(p_img.device)
            y = torch.linspace(-1, 1, self.aim_height).view(-1, 1).expand(-1, self.aim_height).to(p_img.device)
            
            x = x*self.center[0]
            y = y*self.center[1]
            # 将笛卡尔坐标转换为极坐标
            theta = (torch.atan2(y, x)-torch.pi)/(torch.pi)  # 计算角度
            
            theta[theta<-1] = theta[theta<-1]+2
            
            radius = ((torch.sqrt(x**2 + y**2)-self.min_radius)-(self.max_radius-self.min_radius)/2)/((self.max_radius-self.min_radius)/2) # 计算半径
            
            grid = torch.stack([radius, theta], dim=2).view(1, self.aim_height, self.aim_width, 2).to(p_img.device)
            grid = grid.repeat(bs,1,1,1) + offset
            cartesian_image = F.grid_sample(p_img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            
     

        return cartesian_image
    
class OffsetNet(nn.Module):
    def __init__(self, c1, c2):
        super(OffsetNet, self).__init__()
        
        self.offconv_r = Conv(c1, c1, k=1, s=1)
        self.offconv_c = Conv(c2, c1, k=1, s=1)
        
        self.combat_x = Conv(c1, 1, k=3, s=1)
        self.combat_y = Conv(c1, 1, k=3, s=1)
        
        self.tanh = nn.Tanh()
        
    def forward(self, r, c):
        
        r_1 = self.offconv_r(r)
        c_1 = self.offconv_c(c)
            
        
        sub = r_1-c_1
        
        offset_x = self.combat_x(sub)
        offset_y = self.combat_y(sub)

        
        merge = torch.cat([offset_x, offset_y], dim=1)
        
        return self.tanh(merge).permute(0,2,3,1)/(offset_x.shape[2]/2)*5
    
class AMMFAM(nn.Module):
    def __init__(self, c1, c2, k1, k2, ratio, tmp, zoom):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.ratio = ratio # 图像的h/w
        self.tmp = tmp # tmp=1保留w, tmp=0保留h
        self.zoom = zoom
        
        self.offsetnet = OffsetNet(c1[0], c1[1])
        
        self.polar = PolarTransform(c1[1], c2, k1, k2, ratio, tmp, zoom)
        self.apolar = PolarTransform(c1[1], c2, k1, k2, ratio, tmp, zoom)
        
        self.conv = Conv(c1[0]+c1[1], c2, k=1, s=1)
        
    def forward(self, x):
        
        r = x[0]
        c = x[1]
        
        if (r.shape[2]==r.shape[3]):
            r_c = c
            
            merge = torch.cat([r, c], dim=1)
            
            out = self.conv(merge)
        else:
            r_c = self.polar(c)
            
            if r.shape!=r_c.shape:
                r_c = F.interpolate(r_c, scale_factor=(1, r.shape[-1]/r_c.shape[-1]), mode='bilinear')
            
            offset = self.offsetnet(r, r_c)
    
            c_out = self.apolar([c, offset])
            
            merge = torch.cat([r, c_out], dim=1)
            
            out = self.conv(merge)
        
        return out
    
####################################################
class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False, dilation=1)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 dim,
                 embed_dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 layer_scale_init_value
                 ):

        super().__init__()
        
        drop_path = [-1]*depth
        pconv_fw_type = 'split_cat'

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.GELU,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, in_chans, embed_dim, patch_size, patch_stride,  norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer == "BN":
            self.norm = nn.BatchNorm2d(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchMerging(nn.Module):

    def __init__(self, dim, embed_dim, patch_size2, patch_stride2, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, 2 * dim, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer == "BN":
            self.norm = nn.BatchNorm2d(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x
    
####################################################

class APartial_conv(nn.Module):

    def __init__(self, dim, n_div):
        super().__init__()
        
        assert n_div>=4 and n_div<=dim, "n_div must >= 4 and <=dim"

        self.dim_conv3 = dim // n_div * 4
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3x1 = nn.Conv2d(self.dim_conv3//4, 
                                       self.dim_conv3//4, 
                                       kernel_size=3, 
                                       stride=1, 
                                       padding=1,
                                       padding_mode='circular',
                                       bias=False, 
                                       dilation=1)
        
        self.partial_conv3x4 = nn.Conv2d(self.dim_conv3//4, 
                                       self.dim_conv3//4, 
                                       kernel_size=3, 
                                       stride=1, 
                                       padding=4,
                                       padding_mode='circular',
                                       bias=False, 
                                       dilation=4)
        
        self.partial_conv3x7 = nn.Conv2d(self.dim_conv3//4, 
                                       self.dim_conv3//4, 
                                       kernel_size=3, 
                                       stride=1, 
                                       padding=7,
                                       padding_mode='circular',
                                       bias=False, 
                                       dilation=7)
        
        self.partial_conv3x10 = nn.Conv2d(self.dim_conv3//4, 
                                       self.dim_conv3//4, 
                                       kernel_size=3, 
                                       stride=1, 
                                       padding=10,
                                       padding_mode='circular',
                                       bias=False, 
                                       dilation=10)

        self.forward = self.forward_split_cat
        
        self.norm_layer = nn.BatchNorm2d(dim)
        self.act_layer = nn.GELU()

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        
        x1_1, x1_2, x1_3, x1_4 = torch.split(x1, [self.dim_conv3//4]*4, dim=1)
        
        x1_1, x1_2, x1_3, x1_4 = self.partial_conv3x10(x1_1), self.partial_conv3x7(x1_2), self.partial_conv3x4(x1_3), self.partial_conv3x1(x1_4)
        
        x = torch.cat((x1_1, x1_2, x1_3, x1_4, x2), 1)

        return self.act_layer(self.norm_layer(x))

    
class AttentionIteract(nn.Module):
    def __init__(self, c1, c2):
        super(AttentionIteract, self).__init__()
        
        self.polar = PolarTransform(c1, c2, 1, 0, 1, 1, 1)
        self.ipolar = InversePolarTransform(c1, c2, 1, 0, 1, 1, 2)
        self.ap1 = APartial_conv(c1[0], 6)
        self.ap2 = APartial_conv(c1[1], 6)
        self.conv1 = Conv(2*c1[0]+c1[1], c1[0])
        self.conv2 = Conv(2*c1[1]+c1[0], c1[1])
        
        self.attention = MLCA(c1[1]+c1[0])
        
    def forward(self, x):
        r = x[0]
        c = x[1]
        
        r_c = self.polar(c)
        
        ap_r = self.ap1(r)
        ap_c = self.ap2(r_c)
        
        merge = self.attention(torch.cat([ap_r, ap_c], dim=1))
        
        r_bench = self.conv1(torch.cat([r, merge], dim=1))
        
        # c_bench = self.conv2(torch.cat([c, self.ipolar(merge)], dim=1))
        
        return r_bench

####################################################
class MLCA(nn.Module):
    def __init__(self, in_size, local_size=5, gamma = 2, b = 1,local_weight=0.5):
        super(MLCA, self).__init__()

        # ECA 计算方法
        self.local_size=local_size
        self.gamma = gamma
        self.b = b
        t = int(abs(math.log(in_size, 2) + self.b) / self.gamma)   # eca  gamma=2
        k = t if t % 2 else t + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.conv_local = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.local_weight=local_weight

        self.local_arv_pool = nn.AdaptiveAvgPool2d(local_size)
        self.global_arv_pool=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        local_arv=self.local_arv_pool(x)
        global_arv=self.global_arv_pool(local_arv)

        b,c,m,n = x.shape
        b_local, c_local, m_local, n_local = local_arv.shape

        # (b,c,local_size,local_size) -> (b,c,local_size*local_size)-> (b,local_size*local_size,c)-> (b,1,local_size*local_size*c)
        temp_local= local_arv.view(b, c_local, -1).transpose(-1, -2).reshape(b, 1, -1)
        temp_global = global_arv.view(b, c, -1).transpose(-1, -2)

        y_local = self.conv_local(temp_local)
        y_global = self.conv(temp_global)


        # (b,c,local_size,local_size) <- (b,c,local_size*local_size)<-(b,local_size*local_size,c) <- (b,1,local_size*local_size*c)
        y_local_transpose=y_local.reshape(b, self.local_size * self.local_size,c).transpose(-1,-2).view(b,c, self.local_size , self.local_size)
        y_global_transpose = y_global.view(b, -1).transpose(-1, -2).unsqueeze(-1)

        # 反池化
        att_local = y_local_transpose.sigmoid()
        att_global = F.adaptive_avg_pool2d(y_global_transpose.sigmoid(),[self.local_size, self.local_size])
        att_all = F.adaptive_avg_pool2d(att_global*(1-self.local_weight)+(att_local*self.local_weight), [m, n])

        x=x * att_all
        return x

class mixConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        assert isinstance(c1, int) and c1%2 == 0 and c2%2 == 0
        self.Conv = Conv(c1//2, c2//2, k, s, p, g, d, act)
    def forward(self, x):
        c1 = x.shape[1]
        split_cir = torch.split(x, c1//2, dim=1)[0]
        split_rec = torch.split(x, c1//2, dim=1)[1]

        mix_x = torch.cat((self.Conv(split_cir), self.Conv(split_rec)), dim=1)
        return mix_x
    
class splitConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, tmp=0):
        super().__init__()
        assert isinstance(c1, int) and c1%2 == 0 and c2%2 == 0
        self.Conv = Conv(c1//2, c2, k, s)
        self.tmp = tmp
    def forward(self, x):
        c1 = x.shape[1]
        if self.tmp == 0:
            split = torch.split(x, c1//2, dim=1)[0]
            return self.Conv(split)
        elif self.tmp == 1:
            split = torch.split(x, c1//2, dim=1)[1]
            split = torch.reshape(split, (split.shape[0], split.shape[1], int(2*split.shape[2]), int(split.shape[3]/2)))
            return self.Conv(split)

class Identity(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
    def forward(self, x):
        return x

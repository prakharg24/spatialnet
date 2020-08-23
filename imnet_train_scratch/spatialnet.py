"""senet in pytorch



[1] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu

    Squeeze-and-Excitation Networks
    https://arxiv.org/abs/1709.01507
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

older_channels = [32, 16, 24, 32, 64, 96, 160, 320]
older_repeats = [1, 2, 3, 4, 3, 3, 1]

original_channels = [32, 16, 24, 40, 80, 112, 192, 320]
original_repeats = [1, 2, 2, 3, 3, 4, 1]

depth_coefficient = [1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.1, 3.6]
width_coefficient = [1.0, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]


def round_filters(filters, multiplier):
  orig_f = filters
  divisor = 8
  min_depth = filters

  filters *= multiplier
  new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_filters < 0.9 * filters:
    new_filters += divisor
  return int(new_filters)


def round_repeats(repeats, multiplier):
  return int(math.ceil(multiplier * repeats))


# print("Original channels")
# print(original_channels)
# print("Original repeats")
# print(original_repeats)
# print("New channels")
# print(new_channels)
# print("New repeats")
# print(new_repeats)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class OriginalSEFixed(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100, r=24, dimension=8, kernel=3):
        super().__init__()

        self.t = t
        self.residual_pointwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            Swish()
        )

        self.residual_depthwise = nn.Sequential(
            nn.Conv2d(in_channels * t, in_channels * t, kernel, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            Swish()
        )

        self.residual2 = nn.Sequential(
            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels * t, max(1, in_channels * t // r)),
            Swish(),
            nn.Linear(max(1, in_channels * t // r), in_channels * t),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        # Create shortcut connection
        shortcut = self.shortcut(x)

        # Exapnsion of channels followed by depthwise convolutions
        if(self.t!=1):
            x = self.residual_pointwise(x)
        residual = self.residual_depthwise(x)

        # Channel wise Squeeze and Excitation block\
        channel_start = residual
        squeeze = self.squeeze(channel_start)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)
        excitation = residual * excitation.expand_as(residual)

        residual2 = self.residual2(excitation)

        if self.stride == 1 and self.in_channels == self.out_channels:
            x = residual2 + shortcut
        else:
            x = residual2

        return x

class OriginalSESpatialBlockFixed(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100, r=24, dimension=8, kernel=3):
        super().__init__()

        self.t = t
        self.dimension = min(dimension, 8)
        self.residual_pointwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            Swish()
        )

        self.residual_depthwise = nn.Sequential(
            nn.Conv2d(in_channels * t, in_channels * t, kernel, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            Swish()
        )

        self.residual2 = nn.Sequential(
            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.channel_start = nn.Linear(dimension*dimension, 1, bias=False)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels * t, max(1, in_channels * t // r)),
            Swish(),
            nn.Linear(max(1, in_channels * t // r), in_channels * t),
            nn.Sigmoid()
        )

        self.spatial_start = nn.Conv2d(in_channels * t, 1, 1)
        self.spatial_squeeze = nn.AdaptiveAvgPool3d((1,self.dimension,self.dimension))
        self.spatial_excitation = nn.Sequential(
            nn.Linear(self.dimension * self.dimension, self.dimension * self.dimension * 2),
            Swish(),
            nn.Linear(self.dimension * self.dimension * 2, self.dimension * self.dimension),
            nn.Sigmoid()
        )

        self.shortcut = nn.Sequential()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        # Create shortcut connection
        shortcut = self.shortcut(x)

        # Exapnsion of channels followed by depthwise convolutions
        if(self.t!=1):
            x = self.residual_pointwise(x)
        residual = self.residual_depthwise(x)

        # Channel wise Squeeze and Excitation block\
        # print(residual.size())
        channel_start = residual
        channel_start = channel_start.view(residual.size(0), residual.size(1), -1)
        channel_start = self.channel_start(channel_start)
        channel_start = channel_start.view(residual.size(0), residual.size(1), 1, 1)
        squeeze = self.squeeze(channel_start)
        squeeze = squeeze.view(squeeze.size(0), -1)
        excitation = self.excitation(squeeze)
        excitation = excitation.view(residual.size(0), residual.size(1), 1, 1)
        excitation = residual * excitation.expand_as(residual)

        # Spatial Squeeze and excitation
        spatial_start = self.spatial_start(residual)
        spatial_squeeze = self.spatial_squeeze(spatial_start)
        spatial_squeeze = spatial_squeeze.view(spatial_squeeze.size(0), -1)
        spatial_excitation = self.spatial_excitation(spatial_squeeze)
        spatial_excitation = spatial_excitation.view(residual.size(0), 1, self.dimension, self.dimension)
        spatial_excitation = F.interpolate(spatial_excitation, (residual.size(2), residual.size(3)))
        spatial_excitation = residual * spatial_excitation.expand_as(residual)

        # combine_excitation = excitation + residual
        # combine_excitation = excitation + spatial_excitation + residual
        combine_excitation = nn.ReLU6()(excitation + spatial_excitation + residual)
        # combine_excitation = nn.ReLU6()((excitation + spatial_excitation + residual)/3)
        # combine_excitation = nn.ReLU6()(torch.max(excitation, spatial_excitation))
        # combine_excitation = nn.ReLU6()(torch.cat((excitation, spatial_excitation, residual), 1))
        # combine_excitation = excitation + spatial_excitation
        # combine_excitation = excitation
        # Final channel reduction layer
        residual2 = self.residual2(combine_excitation)

        if self.stride == 1 and self.in_channels == self.out_channels:
            x = residual2 + shortcut
        else:
            x = residual2

        return x

class SpatialNetUpscaled(nn.Module):

    def __init__(self, class_num=1000, block_func=OriginalSESpatialBlockFixed, level=0):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, stride=2, padding=1),
            nn.BatchNorm2d(32),
            Swish()
        )

        self.block_func = block_func

        start_dimension = 113

        new_channels = [round_filters(f, width_coefficient[level]) for f in original_channels]
        new_repeats = [round_repeats(f, depth_coefficient[level]) for f in original_repeats]

        # self.stage1 = self.block_func(32, 16, 1, 1, dimension=start_dimension)
        self.stage1 = self._make_stage(new_repeats[0], new_channels[0], new_channels[1], 1, 1, dimension=start_dimension)
        self.stage2 = self._make_stage(new_repeats[1], new_channels[1], new_channels[2], 2, 6, dimension=(start_dimension+1)//2)
        self.stage3 = self._make_stage(new_repeats[2], new_channels[2], new_channels[3], 2, 6, dimension=(start_dimension+3)//4)
        self.stage4 = self._make_stage(new_repeats[3], new_channels[3], new_channels[4], 2, 6, dimension=(start_dimension+7)//8)
        self.stage5 = self._make_stage(new_repeats[4], new_channels[4], new_channels[5], 1, 6, dimension=(start_dimension+7)//8)
        self.stage6 = self._make_stage(new_repeats[5], new_channels[5], new_channels[6], 2, 6, dimension=(start_dimension+15)//16)
        # self.stage7 = self.block_func(160, 320, 1, 6, dimension=(start_dimension+7)//8)
        self.stage7 = self._make_stage(new_repeats[6], new_channels[6], new_channels[7], 1, 6, dimension=(start_dimension+15)//16)

        self.conv1 = nn.Sequential(
            nn.Conv2d(new_channels[7], 1280, 1),
            nn.BatchNorm2d(1280),
            Swish()
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t, dimension=8):

        layers = []
        layers.append(self.block_func(in_channels, out_channels, stride, t, dimension=dimension))

        # if(stride!=1):
        #     dimension = (dimension + 1) // 2
        while repeat - 1:
            layers.append(self.block_func(out_channels, out_channels, 1, t, dimension=dimension))
            repeat -= 1

        return nn.Sequential(*layers)

def spatialnet_original(level=0):
    return SpatialNetUpscaled(block_func=OriginalSESpatialBlockFixed, level=level)

def efficientnet_original(level=0):
    return SpatialNetUpscaled(block_func=OriginalSEFixed, level=level)

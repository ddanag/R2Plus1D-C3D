import math

import torch.nn as nn
from torch.nn.modules.utils import _triple

from brevitas.nn import QuantConv2d, QuantLinear, QuantReLU, QuantAvgPool2d
from brevitas.quant import Uint8ActPerTensorFloatMaxInit, Int8ActPerTensorFloatMinMaxInit
from brevitas.quant import IntBias, Int8WeightPerTensorFloat
from brevitas.core.restrict_val import RestrictValueType

from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas_modules.quant_conv3d import QuantConv3d
from brevitas_modules.quant_avg_pool3d import TruncAdaptiveAvgPool3d


class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
    """
    Common per-tensor weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None


class CommonIntWeightPerChannelQuant(CommonIntWeightPerTensorQuant):
    """
    Common per-channel weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_per_output_channel = True


class CommonIntActQuant(Int8ActPerTensorFloatMinMaxInit):
    """
    Common signed act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    min_val = -10.0
    max_val = 10.0
    restrict_scaling_type = RestrictValueType.LOG_FP


class CommonUintActQuant(Uint8ActPerTensorFloatMaxInit):
    """
    Common unsigned act quantizer with bit-width set to None so that it's forced to be specified by
    each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
    max_val = 6.0
    restrict_scaling_type = RestrictValueType.LOG_FP

FIRST_LAYER_BIT_WIDTH = 8

class SpatioTemporalConv(nn.Module):
    r"""Applies a factored 3D convolution over an input signal composed of several input
    planes with distinct spatial and time axes, by performing a 2D convolution over the
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time
    axis to produce the final output.
    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, weight_bit_width, act_bit_width, stride=1, padding=0, bias=False, first_conv=False):
        super(SpatioTemporalConv, self).__init__()

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        spatial_stride = (1, stride[1], stride[2])
        spatial_padding = (0, padding[1], padding[2])

        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_stride = (stride[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)

        if first_conv:
            intermed_channels = 45
        else:
            intermed_channels = int(math.floor(
                (kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) / (
                        kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        self.spatial_conv = QuantConv3d(in_channels = in_channels, out_channels = intermed_channels, kernel_size = spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias, weight_quant=CommonIntWeightPerChannelQuant, weight_bit_width=weight_bit_width)
        
        self.bn1 = nn.BatchNorm3d(intermed_channels)

        self.temporal_conv = QuantConv3d(in_channels = intermed_channels, out_channels = out_channels, kernel_size = temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias, weight_quant=CommonIntWeightPerChannelQuant, weight_bit_width=weight_bit_width)
        #pytorch conv 3d: in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
        #bias=True, padding_mode='zeros', device=None, dtype=None

        #brevitas quant 3d: in_channels, out_channels, kernel_size, stride,
        # padding, dilation, groups, bias, padding_type, 
        # weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
        # bias_quant: Optional[BiasQuantType] = None, input_quant: Optional[ActQuantType] = None,
        # output_quant: Optional[ActQuantType] = None, return_quant_tensor: bool = False,
        

        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = QuantReLU(  act_quant=CommonUintActQuant,
                                bit_width=act_bit_width,
                                per_channel_broadcastable_shape=(1, 1280, 1, 1),
                                scaling_per_channel=False,
                                return_quant_tensor=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.spatial_conv(x)))
        x = self.relu(self.bn2(self.temporal_conv(x)))
        return x


class ResBlock(nn.Module):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        Args:
            in_channels (int): Number of channels in the input tensor
            out_channels (int): Number of channels in the output produced by the block
            kernel_size (int or tuple): Size of the convolving kernels
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size,  weight_bit_width, act_bit_width, downsample=False):
        super(ResBlock, self).__init__()

        self.downsample = downsample
        padding = kernel_size // 2

        if self.downsample:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2, weight_bit_width = weight_bit_width, act_bit_width = act_bit_width)
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2, weight_bit_width = weight_bit_width, act_bit_width = act_bit_width)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, weight_bit_width = weight_bit_width, act_bit_width = act_bit_width)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding, weight_bit_width = weight_bit_width, act_bit_width = act_bit_width)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.relu = QuantReLU()

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.relu(x + res)


class ResLayer(nn.Module):
    r"""Forms a single layer of the ResNet network, with a number of repeating
    blocks of same output size stacked on top of each other
        Args:
            in_channels (int): Number of channels in the input tensor
            out_channels (int): Number of channels in the output produced by the layer
            kernel_size (int or tuple): Size of the convolving kernels
            layer_size (int): Number of blocks to be stacked to form the layer
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, weight_bit_width, act_bit_width, downsample=False):

        super(ResLayer, self).__init__()

        # implement the first block
        self.block1 = ResBlock(in_channels, out_channels, kernel_size, downsample=downsample, weight_bit_width = weight_bit_width, act_bit_width = act_bit_width)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [ResBlock(out_channels, out_channels, kernel_size, weight_bit_width = weight_bit_width, act_bit_width = act_bit_width)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class FeatureLayer(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in
    each layer set by layer_sizes, and by performing a global average pool at the end producing a
    512-dimensional vector for each element in the batch.
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
    """

    def __init__(self, layer_sizes, weight_bit_width, act_bit_width, input_channel=3):
        super(FeatureLayer, self).__init__()

        self.conv1 = SpatioTemporalConv(input_channel, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                                        first_conv=True, weight_bit_width = weight_bit_width, act_bit_width = act_bit_width)
        self.conv2 = ResLayer(64, 64, 3, layer_sizes[0], weight_bit_width = weight_bit_width, act_bit_width = act_bit_width)
        self.conv3 = ResLayer(64, 128, 3, layer_sizes[1], downsample=True, weight_bit_width = weight_bit_width, act_bit_width = act_bit_width)
        self.conv4 = ResLayer(128, 256, 3, layer_sizes[2], downsample=True, weight_bit_width = weight_bit_width, act_bit_width = act_bit_width)
        self.conv5 = ResLayer(256, 512, 3, layer_sizes[3], downsample=True, weight_bit_width = weight_bit_width, act_bit_width = act_bit_width)
        # global average pooling of the output
        self.pool = TruncAdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)

        return x.view(-1, 512)


class QuantR2Plus1D(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers,
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch,
    and passing them through a Linear layer.
        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
        """

    def __init__(self, num_classes, layer_sizes, weight_bit_width, act_bit_width, input_channel=3):
        super(QuantR2Plus1D, self).__init__()

        self.feature = FeatureLayer(layer_sizes, input_channel = input_channel, weight_bit_width = weight_bit_width, act_bit_width = act_bit_width)
        self.fc = QuantLinear(  512, 
                                num_classes, 
                                bias=True,
                                bias_quant=IntBias,
                                weight_quant=CommonIntWeightPerTensorQuant,
                                weight_bit_width=FIRST_LAYER_BIT_WIDTH)

        self.__init_weight()

    def forward(self, x):
        x = self.feature(x)
        logits = self.fc(x)

        return logits

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, QuantConv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

import torch.nn as nn
import torch.nn.functional as F
import torchyield as ty

from torch import Tensor
from typing import Optional

def basic_block(
        *,
        in_channels: int,
        mid_channels: Optional[int] = None,
        out_channels: int,
        stride: int = 1,
):
    if mid_channels is None:
        mid_channels = out_channels

    yield from ty.conv3_bn_relu_layer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
    )
    yield from ty.conv3_bn_layer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
    )

def bottleneck_block(
        *,
        in_channels: int,
        mid_channels: Optional[int] = None,
        out_channels: int,
        bottleneck_factor: Optional[int] = None,
        stride: int = 1,
):
    if mid_channels is None:
        if bottleneck_factor is None:
            raise TypeError("must specify `mid_channels` or `bottleneck_factor`")
        mid_channels = in_channels // bottleneck_factor

    yield from ty.conv3_bn_relu_layer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            padding=0,
    )
    yield from ty.conv3_bn_relu_layer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
    )
    yield from ty.conv3_bn_layer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
    )

class ResBlock(nn.Module):

    def __init__(
            self, *,
            in_channels: int,
            out_channels: int,
            conv_factory: ty.LayerFactory = basic_block,
            conv_stride: int = 1,
            skip_stride: int = 1,
            pool_factory: Optional[ty.LayerFactory] = None,
            pool_size: int = 1,
            pool_before_conv: bool = False,
            **conv_kwargs,
    ):
        super().__init__()

        self.conv = ty.Layers(
                conv_factory(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=conv_stride,
                    **conv_kwargs,
                ),
        )

        if in_channels == out_channels and skip_stride == 1:
            self.skip = None
        else:
            self.skip = ty.Layers(
                    ty.conv3_bn_layer(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=skip_stride,
                    )
            )

        if (pool_factory is None) or (pool_size <= 1):
            self.pool = None
        else:
            self.pool = ty.Layers(pool_factory(pool_size))

        self.pool_before_conv = pool_before_conv

    def forward(self, x: Tensor) -> Tensor:
        if (self.pool is not None) and self.pool_before_conv:
            x = self.pool(x)

        y = self.conv(x)

        if (self.pool is not None) and (not self.pool_before_conv):
            x = self.pool(x)

        if self.skip is not None:
            x = self.skip(x)

        y = F.relu(x + y)

        return y

def resnet_layer(
        *,
        in_channels: list[int],
        out_channels: list[int],
        conv_factory: ty.LayerFactory = basic_block,
        conv_stride: int = 1,
        skip_stride: int = 1,
        pool_factory: Optional[ty.LayerFactory] = None,
        pool_size: int = 1,
        pool_before_conv: bool = False,
        block_repeats: int = 1,
        initial_conv_size: int = 3,
        final_conv_size: int = 0,
        **conv_kwargs,
):
    # Always have an initial convolution with no padding, so we don't add zeros 
    # to the data that aren't really there.
    yield from ty.conv3_bn_relu_layer(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=initial_conv_size,
    )

    if final_conv_size > 0:
        i = slice(1, -1)
    else:
        i = slice(1, None)

    yield from ty.make_layers(
            resnet_blocks,
            in_channels=in_channels[i],
            out_channels=out_channels[i],
            conv_factory=conv_factory,
            conv_stride=conv_stride,
            skip_stride=skip_stride,
            pool_factory=pool_factory,
            pool_size=pool_size,
            pool_before_conv=pool_before_conv,
            block_repeats=block_repeats,
            **conv_kwargs,
    )

    # The last convolution isn't necessary.  It's useful in equivariant 
    # networks, because it's an equivariant way to reduce the size of all the 
    # spatial dimensions to 1, which has to be done before linear layers can be 
    # used.  When equivariance isn't a concern, it's also an option to just 
    # flatten any remaining spatial dimensions.
    if final_conv_size:
        yield from ty.conv3_bn_relu_layer(
                in_channels=in_channels[-1],
                out_channels=out_channels[-1],
                kernel_size=final_conv_size,
        )

def resnet_blocks(
        *,
        in_channels: int,
        out_channels: int,
        conv_factory: ty.LayerFactory = basic_block,
        conv_stride: int = 1,
        skip_stride: int = 1,
        pool_factory: Optional[ty.LayerFactory] = None,
        pool_size: int = 1,
        pool_before_conv: bool = False,
        block_repeats: int = 1,
        **conv_kwargs,
):
    yield ResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            conv_factory=conv_factory,
            conv_stride=conv_stride,
            skip_stride=skip_stride,
            pool_factory=pool_factory,
            pool_size=pool_size,
            pool_before_conv=pool_before_conv,
            **conv_kwargs,
    )

    for i in range(block_repeats - 1):
        yield ResBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                conv_factory=conv_factory,
                **conv_kwargs,
        )


def get_resnet(
        *,
        in_channels: int,
        hidden_channels: list[int],
        **hparams,
):
    return ty.Layers(
            resnet_layer(
                **ty.channels([in_channels, *hidden_channels]),
                **hparams,
            ),
            verbose=True,
    )

def get_default_resnet(**hparams):
    hparams = get_default_resnet_hparams() | hparams
    return get_resnet(**hparams)

def get_default_resnet_hparams():
    return dict(
            hidden_channels=[128, 64, 256, 512, 256, 1024],
            pool_factory=nn.MaxPool3d,
            pool_size=2,
            pool_before_conv=True,
            final_conv_size=0,
    )
    

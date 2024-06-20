import torch
import torch.nn as nn

from atom3d_menagerie.models.resnet import resnet_layer, bottleneck_block
from torch.optim import Adam
from torchyield import Layers, channels
from torchtest import assert_vars_change

def test_resnet_stride():
    # Not a perfect fit.  A convolution with stride=2 on an input with an even 
    # dimension results in an output with exactly half that dimension, but the 
    # filters only require padding on one side.
    resnet = Layers(
            resnet_layer(
                **channels([1, 2, 4, 8, 16]),
                conv_stride=2,
                skip_stride=2,
                block_repeats=2,
            ),
            nn.Flatten(),
            nn.Linear(16, 1),
    )
    x = torch.randn(2, 1, 10, 10, 10)
    y = torch.randn(2, 1)

    assert_vars_change(
            model=resnet,
            loss_fn=nn.MSELoss(),
            optim=Adam(resnet.parameters()),
            batch=(x, y),
            device='cpu',
    )

def test_resnet_pool_final_conv():
    resnet = Layers(
            resnet_layer(
                **channels([1, 2, 4, 8, 16]),
                pool_factory=nn.MaxPool3d,
                pool_size=2,
                pool_before_conv=True,
                block_repeats=2,
                final_conv_size=3,
            ),
            nn.Flatten(),
            nn.Linear(16, 1),
    )
    x = torch.randn(2, 1, 14, 14, 14)
    y = torch.randn(2, 1)

    assert_vars_change(
            model=resnet,
            loss_fn=nn.MSELoss(),
            optim=Adam(resnet.parameters()),
            batch=(x, y),
            device='cpu',
    )

def test_resnet_bottleneck():
    resnet = Layers(
            resnet_layer(
                **channels([1, 2, 4, 8, 16]),
                bottleneck_factor=2,
                conv_factory=bottleneck_block,
                conv_stride=2,
                skip_stride=2,
                block_repeats=2,
            ),
            nn.Flatten(),
            nn.Linear(16, 1),
    )
    x = torch.randn(2, 1, 10, 10, 10)
    y = torch.randn(2, 1)

    assert_vars_change(
            model=resnet,
            loss_fn=nn.MSELoss(),
            optim=Adam(resnet.parameters()),
            batch=(x, y),
            device='cpu',
    )

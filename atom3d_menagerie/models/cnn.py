import numpy as np
import torch.nn as nn

class Cnn(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        spatial_size: int,
        conv_drop_rate: float,
        fc_drop_rate: float,
        conv_filters: list[int],
        conv_kernel_size: int,
        max_pool_positions: list[bool],
        max_pool_sizes: list[int],
        max_pool_strides: list[int],
        fc_units: list[int],
        batch_norm: bool,
        dropout: bool,
    ):
        super().__init__()

        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm3d(in_channels))

        # Convs
        for i in range(len(conv_filters)):
            layers.extend(
                [
                    nn.Conv3d(
                        in_channels,
                        conv_filters[i],
                        kernel_size=conv_kernel_size,
                        bias=True,
                    ),
                    nn.ReLU(),
                ]
            )
            spatial_size -= conv_kernel_size - 1
            if max_pool_positions[i]:
                layers.append(nn.MaxPool3d(max_pool_sizes[i], max_pool_strides[i]))
                spatial_size = int(
                    np.floor(
                        (spatial_size - (max_pool_sizes[i] - 1) - 1)
                        / max_pool_strides[i]
                        + 1
                    )
                )
            if batch_norm:
                layers.append(nn.BatchNorm3d(conv_filters[i]))
            if dropout:
                layers.append(nn.Dropout(conv_drop_rate))
            in_channels = conv_filters[i]

        layers.append(nn.Flatten())
        in_features = in_channels * (spatial_size**3)

        # FC layers
        for units in fc_units:
            layers.extend([nn.Linear(in_features, units), nn.ReLU()])
            if batch_norm:
                layers.append(nn.BatchNorm3d(units))
            if dropout:
                layers.append(nn.Dropout(fc_drop_rate))
            in_features = units

        # Final FC layer
        layers.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def get_default_cnn(in_channels, spatial_size):
    return Cnn(
            in_channels=in_channels,
            spatial_size=spatial_size,
            **get_default_cnn_hparams(),
    )

def get_default_cnn_hparams():
    num_conv = 4
    return dict(
            conv_drop_rate=0.1,
            fc_drop_rate=0.25,
            conv_filters=[32 * (2**n) for n in range(num_conv)],
            conv_kernel_size=3,
            max_pool_positions=[False, True] * int((num_conv+1)/2),
            max_pool_sizes=[2] * num_conv,
            max_pool_strides=[2] * num_conv,
            fc_units=[512],
            batch_norm=False,
            dropout=True,
    )

if __name__ == '__main__':
    from torchinfo import summary

    summary(get_default_cnn(5, 21), input_size=(1, 5, 21, 21, 21))

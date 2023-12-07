import torch
import torch.nn as nn

from atompaint.nonlinearities import add_gates
from atompaint.pooling import FourierExtremePool3D, FourierAvgPool3D
from escnn.nn import (
        FieldType, FourierFieldType, GeometricTensor,
        R3Conv, IIDBatchNorm3d, FourierPointwise, GatedNonLinearity1,
        PointwiseAvgPoolAntialiased3D,
)
from itertools import cycle
from more_itertools import pairwise

from typing import TypeAlias, Callable, Iterable
from atompaint.type_hints import LayerFactory, ModuleFactory, Grid

InvariantFactory: TypeAlias = Callable[[FieldType, int], Iterable[nn.Module]]
LazyLinearFactory: TypeAlias = Callable[[int], Iterable[nn.Module]]
AdaptivePoolFactory: TypeAlias = Callable[[tuple[int, int, int]], nn.Module]

class EquivariantCnn(nn.Module):

    def __init__(
            self,
            *,
            field_types: Iterable[FieldType],
            conv_factory: LayerFactory,
            pool_factory: ModuleFactory,
            pool_toggles: list[bool],
            invariant_factory: InvariantFactory,
            mlp_channels: list[int],
            mlp_factory: LazyLinearFactory,
    ):
        super().__init__()

        field_types = list(field_types)

        def iter_layers():
            layer_params = zip(pairwise(field_types), cycle(pool_toggles))

            for (in_type, out_type), has_pool in layer_params:
                yield from conv_factory(in_type, out_type)
                if has_pool:
                    yield pool_factory(out_type)

            # After the invariance step, the output must be 1x1x1, otherwise 
            # flattening won't be equivariant.  That means that the 
            # tensor-extraction step will have to happen in the invariant 
            # factory, which means that I'll need a module for it.

            yield from invariant_factory(out_type)

            for out_channels in mlp_channels:
                yield from mlp_factory(out_channels)

            yield nn.Linear(out_channels, 1)

        self.layers = nn.Sequential(*iter_layers())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = GeometricTensor(x, self.layers[0].in_type)
        return self.layers(x)

class UnwrapTensor(nn.Module):

    def forward(self, x: GeometricTensor) -> torch.Tensor:
        return x.tensor

class Require1x1x1(nn.Module):

    def forward(self, x):
        if isinstance(x, GeometricTensor):
            y = x.tensor
        else:
            y = x

        assert len(y.shape) == 5
        assert y.shape[2:] == (1, 1, 1)

        return x


def conv_bn_gated(
        in_type: FieldType,
        out_type: FieldType,
        padding: int = 0,
):
    gate_type = add_gates(out_type)
    yield R3Conv(in_type, gate_type, kernel_size=3, padding=padding, bias=False)
    yield IIDBatchNorm3d(gate_type)
    yield GatedNonLinearity1(gate_type)

def conv_bn_fourier(
        in_type: FieldType,
        out_type: FourierFieldType,
        ift_grid: Grid,
        padding: int = 0,
        function='p_elu',
):
    yield R3Conv(in_type, out_type, kernel_size=3, padding=padding, bias=False)
    yield IIDBatchNorm3d(out_type)
    yield FourierPointwise(out_type, ift_grid, function=function)

def pool_conv(in_type):
    return R3Conv(
            in_type,
            in_type,
            kernel_size=3,
            stride=2,
            padding=1,
    )

def pool_fourier_extreme(in_type, ift_grid):
    return FourierExtremePool3D(
            in_type,
            grid=ift_grid,
            kernel_size=2,
    )

def pool_fourier_avg(in_type, ift_grid):
    return FourierAvgPool3D(
            in_type,
            grid=ift_grid,
            stride=2,
    )

def pool_avg(in_type):
    return PointwiseAvgPoolAntialiased3D(
            in_type,
            sigma=0.6,
            stride=2,
    )

def invariant_conv(in_type, out_channels, kernel_size, **kwargs):
    out_type = FieldType(
            in_type.gspace,
            out_channels * [in_type.gspace.trivial_repr],
    )
    yield R3Conv(in_type, out_type, kernel_size, **kwargs)
    yield UnwrapTensor()
    yield Require1x1x1()
    yield nn.Flatten()

def invariant_fourier(
        in_type: FourierFieldType,
        *,
        ift_grid: Grid,
        function='p_elu',
        pool: AdaptivePoolFactory,
):
    # This only works if the input size is 1x1x1, or if everything is 
    # average/max-pooled right afterward.  Otherwise, the tensor after this 
    # operation will have all the same values, but in different positions, and 
    # subsequent steps will break equivariance.
    yield Require1x1x1()

    out_type = FourierFieldType(
            in_type.gspace, 
            channels=in_type.channels,
            bl_irreps=in_type.gspace.fibergroup.bl_irreps(0),
            subgroup_id=in_type.subgroup_id,
    )
    yield FourierPointwise(
            in_type=in_type,
            out_type=out_type,
            grid=ift_grid,
            function='p_elu',
    )
    yield UnwrapTensor()
    yield nn.Flatten()

def invariant_fourier_pool(
        in_type: FourierFieldType,
        *,
        ift_grid: Grid,
        function='p_elu',
        pool: nn.Module,
):
    out_type = FourierFieldType(
            in_type.gspace, 
            channels=in_type.channels,
            bl_irreps=in_type.gspace.fibergroup.bl_irreps(0),
            subgroup_id=in_type.subgroup_id,
    )
    yield FourierPointwise(
            in_type=in_type,
            out_type=out_type,
            grid=ift_grid,
            function='p_elu',
    )
    yield UnwrapTensor()
    yield pool
    yield Require1x1x1()
    yield nn.Flatten()

def linear_relu_dropout(out_channels, drop_rate):
    yield nn.LazyLinear(out_channels)
    yield nn.ReLU()
    yield nn.Dropout(drop_rate)

def linear_relu_bn(out_channels):
    yield nn.LazyLinear(out_channels)
    yield nn.ReLU()
    yield nn.BatchNorm1d(out_channels)

def linear_bn_relu(out_channels):
    yield nn.LazyLinear(out_channels)
    yield nn.BatchNorm1d(out_channels)
    yield nn.ReLU()


if __name__ == '__main__':
    from atompaint.encoders.layers import (
            make_trivial_field_type, make_fourier_field_types,
    )
    from escnn.gspaces import rot3dOnR3
    from torchinfo import summary
    from functools import partial

    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    so2_z = False, -1
    grid = so3.grid('thomson_cube', N=4)
    grid_s2 = so3.sphere_grid('thomson_cube', N=4)
    L = 2

    field_types = [
            make_trivial_field_type(gspace, 5),
            *make_fourier_field_types(
                gspace,
                channels=[1, 2, 4],
                max_frequencies=L,
            ),
            *make_fourier_field_types(
                gspace,
                channels=[28],
                max_frequencies=L,
                subgroup_id=so2_z,
            ),
    ]
    
    cnn = EquivariantCnn(
            field_types=field_types,
            conv_factory=conv_bn_gated,
            #pool_factory=pool_conv,
            pool_factory=partial(
                pool_fourier_extreme,
                ift_grid=grid,
            ),
            #pool_factory=pool_avg,
            pool_toggles=[False, True],
            # invariant_factory=partial(
            #     invariant_conv,
            #     out_channels=256,
            #     kernel_size=3,
            # ),
            invariant_factory=partial(
                invariant_fourier_pool,
                ift_grid=grid_s2,
                pool=nn.AdaptiveMaxPool3d((1,1,1)),
            ),
            mlp_channels=[512],
            mlp_factory=linear_bn_relu,
    )

    summary(cnn, input_size=(2, 5, 20, 20, 20))

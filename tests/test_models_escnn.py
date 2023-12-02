import torch
import torch.nn as nn

from atom3d_lba_menagerie.models.escnn import (
        EquivariantCnn,
        conv_bn_fourier,
        conv_bn_gated,
        pool_conv,
        pool_fourier_extreme,
        pool_fourier_avg,
        pool_avg,
        invariant_conv,
        invariant_fourier,
        invariant_fourier_pool,
        linear_relu_dropout,
        linear_relu_bn,
        linear_bn_relu,
)
from atompaint.pooling import FourierExtremePool3D
from atompaint.encoders.layers import (
        make_trivial_field_type, make_fourier_field_types,
)
from atompaint.vendored.escnn_nn_testing import (
        check_invariance, check_equivariance, get_exact_3d_rotations,
)
from escnn.gspaces import rot3dOnR3
from torchinfo import summary
from functools import partial

def test_equivariant_cnn_baseline():
    # This "baseline" model is meant to use all of the simplest, most robustly 
    # equivariant layers.  In later tests, I'll incorporate more complicated, 
    # less equivariant layers one at a time, to isolate any problems that 
    # occur.

    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    so2_z = False, -1
    grid = so3.grid('thomson_cube', N=4)
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
            pool_factory=pool_conv,
            pool_toggles=[False, True],
            invariant_factory=partial(
                invariant_conv,
                out_channels=256,
                kernel_size=3,
            ),
            mlp_channels=[512],
            mlp_factory=linear_bn_relu,
    )

    check_invariance(
            cnn,
            in_tensor=torch.randn(8, 5, 21, 21, 21),
            in_type=field_types[0],
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

def test_equivariant_cnn_fourier_conv():
    gspace = rot3dOnR3()
    so3 = gspace.fibergroup
    so2_z = False, -1
    grid = so3.grid('thomson_cube', N=4)
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
            conv_factory=partial(
                conv_bn_fourier,
                ift_grid=grid,
            ),
            pool_factory=pool_conv,
            pool_toggles=[False, True],
            invariant_factory=partial(
                invariant_conv,
                out_channels=256,
                kernel_size=3,
            ),
            mlp_channels=[512],
            mlp_factory=linear_bn_relu,
    )

    check_invariance(
            cnn,
            in_tensor=torch.randn(8, 5, 21, 21, 21),
            in_type=field_types[0],
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

def test_equivariant_cnn_pool_fourier_extreme():
    # Note that the equivariance of the Fourier extreme pooling module (and 
    # probably all of the pooling modules) depends on all the underlying 
    # filters aligning perfectly with the input tensor.  If a filter starts 
    # touching the edge on one side of an image, but stops one voxel away from 
    # the edge on the other side, then rotated/non-rotated inputs will give 
    # different outputs.
    #
    # The Fourier extreme pooling module requires even input dimensions in 
    # order for this condition to be met.  If N such layers are going to be 
    # used, then the input must have an input dimension divisible by 2N.

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
            pool_factory=partial(
                pool_fourier_extreme,
                ift_grid=grid,
            ),
            pool_toggles=[False, True],

            # Use the invariant layer that pools at the end, because that's the 
            # only good way to deal with 2x2x2 spatial dimensions.
            invariant_factory=partial(
                invariant_fourier_pool,
                ift_grid=grid_s2,
                pool=nn.MaxPool3d(2),
            ),
            mlp_channels=[512],
            mlp_factory=linear_bn_relu,
    )

    check_invariance(
            cnn,
            in_tensor=torch.randn(8, 5, 20, 20, 20),
            in_type=field_types[0],
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-2,
    )

def test_equivariant_cnn_pool_fourier_avg():
    # For a long time I didn't understand why the pointwise antialiased average 
    # pooling layer seemed to be equivariant in the example SE(3) CNN, but not 
    # in my tests (expt #219).  It turns out that this issue was that I was 
    # using wrong-sized inputs.  I realized this when I has trouble getting the 
    # above Fourier extreme pooling test to work, despite knowing having seen 
    # that module be equivariant in other networks, and the issue ended up 
    # being the input size  (see comments in that test for more info).

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
            pool_factory=partial(
                pool_fourier_avg,
                ift_grid=grid,
            ),
            pool_toggles=[False, True],
            invariant_factory=partial(
                invariant_conv,
                out_channels=256,
                kernel_size=3,
            ),
            mlp_channels=[512],
            mlp_factory=linear_bn_relu,
    )

    check_invariance(
            cnn,
            in_tensor=torch.randn(8, 5, 21, 21, 21),
            in_type=field_types[0],
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

def test_equivariant_cnn_pool_avg():
    # I don't understand why the pointwise antialiased average pool layer seems 
    # equivariant here, but not in my other tests.  Is it because my other 
    # tests used input sizes that can't be tiled perfectly by the filter?

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
            pool_factory=pool_avg,
            pool_toggles=[False, True],
            invariant_factory=partial(
                invariant_conv,
                out_channels=256,
                kernel_size=3,
            ),
            mlp_channels=[512],
            mlp_factory=linear_bn_relu,
    )

    check_invariance(
            cnn,
            in_tensor=torch.randn(8, 5, 21, 21, 21),
            in_type=field_types[0],
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )

def test_equivariant_cnn_invariant_fourier_pool():
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
                channels=[64],
                max_frequencies=L,
                subgroup_id=so2_z,
            ),
    ]
    
    cnn = EquivariantCnn(
            field_types=field_types,
            conv_factory=conv_bn_gated,
            pool_factory=pool_conv,
            pool_toggles=[False, True],
            invariant_factory=partial(
                invariant_fourier_pool,
                ift_grid=grid_s2,
                pool=nn.MaxPool3d(3),
            ),
            mlp_channels=[512],
            mlp_factory=linear_bn_relu,
    )

    check_invariance(
            cnn,
            in_tensor=torch.randn(8, 5, 21, 21, 21),
            in_type=field_types[0],
            group_elements=get_exact_3d_rotations(so3),
            atol=1e-4,
    )


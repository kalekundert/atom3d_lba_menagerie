import torch
import polars as pl
import numpy as np
import os

from .utils import LmdbDataModule, convert_to_mmdf_format
from atom3d_menagerie.hparams import if_gpu
from macromol_gym_pretrain import image_from_atoms, ImageParams
from macromol_voxelize import Grid
from macromol_dataframe import (
        transform_atom_coords, prune_hydrogen, prune_water,
)
from atompaint.transform_pred.datasets.utils import sample_coord_frame
from atom3d.util.voxelize import get_center
from pipeline_func import f
from functools import partial
from pathlib import Path

class VoxelizedLbaDataModule(LmdbDataModule):

    def __init__(self, *, img_params, **kwargs):
        super().__init__(
                make_inputs=partial(make_lba_inputs, img_params=img_params),
                **kwargs,
        )

def make_lba_inputs(rng, item, img_params):
    ligand_xyz_i = item['atoms_ligand'][['x', 'y', 'z']].astype(np.float32)
    ligand_center_i = get_center(ligand_xyz_i)

    frame_ix = sample_coord_frame(rng, ligand_center_i)

    atoms_i = pl.concat(
            convert_to_mmdf_format(item[k], is_polymer=pl.lit(is_polymer))
            for k, is_polymer in [
                ('atoms_protein', True), 
                ('atoms_ligand', False),
            ]
    )
    atoms_x = (
            atoms_i
            | f(transform_atom_coords, frame_ix)
            | f(prune_hydrogen)
            | f(prune_water)
    )

    img = image_from_atoms(atoms_x, img_params)
    img = torch.from_numpy(img).float()

    label = torch.tensor(item['scores']['neglog_aff']).reshape(1)

    return img, label


def get_default_lba_data(**kwargs):
    kwargs = {
            **get_default_lba_data_hparams(),
            **kwargs,
    }
    return VoxelizedLbaDataModule(**kwargs)

def get_default_lba_data_hparams():
    return dict(
            data_dir=get_default_lba_data_dir(),
            img_params=get_default_lba_img_params(),

            # The ATOM3D example uses a batch size of 16.  The GPUs have enough 
            # memory to do much larger batches, which would be slightly faster, 
            # but I'm erring on the side of not changing hyperparameters.
            batch_size=if_gpu(16, 2),

            shuffle=True,
    )

def get_default_lba_data_dir():
    return Path(os.environ['ATOM3D_LBA_DATA_DIR'])

def get_default_lba_img_params():
    return ImageParams(
        grid=Grid(
            length_voxels=21,
            resolution_A=1.0,
        ),
        atom_radius_A=0.5,
        element_channels=[['H'], ['C'], ['O'], ['N'], ['*']],
        ligand_channel=False,
    )

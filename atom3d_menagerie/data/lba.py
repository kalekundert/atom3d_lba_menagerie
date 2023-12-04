import torch
import pandas as pd
import numpy as np
import os

from .utils import LmdbDataModule
from atompaint.datasets.voxelize import image_from_atoms, ImageParams, Grid
from atompaint.datasets.atoms import transform_atom_coords
from atompaint.transform_pred.datasets.utils import sample_coord_frame
from atom3d.util.voxelize import get_center
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

    # I can't find the definition of the pocket, so I'm not 100% convinced 
    # that it really contains all the atoms in the vicinity of the ligand.  
    # But the example code uses the pocket atoms after a random rotation 
    # around the ligand center, so I'll just do the same.

    atoms_i = pd.concat([item['atoms_pocket'], item['atoms_ligand']])
    atoms_x = transform_atom_coords(atoms_i, frame_ix)

    img = image_from_atoms(atoms_x, img_params)
    img = torch.from_numpy(img).float()

    label = torch.tensor(item['scores']['neglog_aff']).reshape(1)

    return img, label


def get_default_lba_data():
    return VoxelizedLbaDataModule(
            **get_default_lba_data_hparams(),
    )

def get_default_lba_data_hparams():
    return dict(
            data_dir=get_default_lba_data_dir(),
            img_params=get_default_lba_img_params(),
    )

def get_default_lba_data_dir():
    return Path(os.environ['ATOM3D_LBA_DATA_DIR'])

def get_default_lba_img_params():
    return ImageParams(
        grid=Grid(
            length_voxels=21,
            resolution_A=1.0,
        ),
        channels=['H', 'C', 'O', 'N', '.*'],
        element_radii_A=0.5,
    )

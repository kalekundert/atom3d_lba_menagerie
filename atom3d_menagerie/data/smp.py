import torch
import numpy as np
import os

from .utils import LmdbDataModule
from atom3d_menagerie.hparams import if_gpu
from atompaint.datasets.voxelize import image_from_atoms, ImageParams, Grid
from atompaint.datasets.atoms import transform_atom_coords
from atompaint.transform_pred.datasets.utils import sample_coord_frame
from atom3d.util.voxelize import get_center
from functools import partial
from pathlib import Path

class VoxelizedSmpDataModule(LmdbDataModule):

    def __init__(self, *, img_params, quantum_prop, **kwargs):
        super().__init__(
                make_inputs=partial(
                    make_smp_inputs,
                    img_params=img_params,
                    quantum_prop=quantum_prop,
                ),
                **kwargs,
        )

QUANTUM_PROPS = {
        x: i
        for i, x in enumerate([
            'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve',
            'u0', 'u298', 'h298', 'g298', 'cv',
            'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom', 'cv_atom',
        ])
}

def make_smp_inputs(rng, item, img_params, quantum_prop):
    atoms_i = item['atoms']
    xyz_i = atoms_i[['x', 'y', 'z']].astype(np.float32)
    center_i = get_center(xyz_i)

    frame_ix = sample_coord_frame(rng, center_i)
    atoms_x = transform_atom_coords(atoms_i, frame_ix)

    img = image_from_atoms(atoms_x, img_params)
    img = torch.from_numpy(img).float()

    label = item['labels'][QUANTUM_PROPS[quantum_prop]]
    label = torch.tensor(label).reshape(1)

    return img, label

def get_default_smp_data(**kwargs):
    return VoxelizedSmpDataModule(
            **get_default_smp_data_hparams(),
            **kwargs,
    )

def get_default_smp_data_hparams():
    return dict(
            data_dir=get_default_smp_data_dir(),
            img_params=get_default_smp_img_params(),

            # See expt #243 for the motivation behind this default.
            quantum_prop='mu',

            # The ATOM3D example uses a batch size of 256, but my models are
            # bigger (21 voxels/side vs. 16 voxels/side).  A batch size of
            # 128 fits into 12 GB, which is the smallest amount of VRAM
            # that any GPU on the cluster has.
            batch_size=if_gpu(128, 2),

            shuffle=True,
    )

def get_default_smp_data_dir():
    return Path(os.environ['ATOM3D_SMP_DATA_DIR'])

def get_default_smp_img_params():
    # Resolution and radius optimized in expt #244.
    return ImageParams(
        grid=Grid(
            length_voxels=21,
            resolution_A=1.0,
        ),
        channels=['H', 'C', 'O', 'N', 'F'],
        element_radii_A=1.0,
    )

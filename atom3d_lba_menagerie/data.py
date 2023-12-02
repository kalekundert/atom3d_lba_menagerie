import lightning.pytorch as pl
import torch
import pandas as pd
import numpy as np
import os

from atompaint.datasets.voxelize import image_from_atoms
from atompaint.datasets.atoms import transform_atom_coords
from atompaint.transform_pred.datasets.utils import sample_coord_frame
from atom3d.datasets import LMDBDataset
from atom3d.util.voxelize import dotdict, get_grid, get_center
from torch.utils.data import DataLoader, Dataset, RandomSampler
from functools import partial
from pathlib import Path

class VoxelizedDataModule(pl.LightningDataModule):

    def __init__(
            self,
            *,
            data_dir,
            make_image,
            batch_size,
            shuffle,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.make_image = make_image
        self.dataloader_kwargs = dict(
                batch_size=batch_size,
                shuffle=shuffle,
        )

    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = self._make_dataset('train')
            self.val_dataset = self._make_dataset('val')

        if stage == 'test':
            self.test_dataset = self._make_dataset('test')

    def train_dataloader(self):
        return self._make_dataloader(self.train_dataset)

    def val_dataloader(self):
        return self._make_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._make_dataloader(self.test_dataset)

    def _make_dataloader(self, dataset):
        kwargs = self.dataloader_kwargs.copy()
        shuffle = kwargs.pop('shuffle')

        indices = range(len(dataset))
        if shuffle:
            indices = RandomSampler(indices)

        sampler = EpochSampler(indices)

        return DataLoader(dataset, sampler=sampler, **kwargs)

    def _make_dataset(self, split_dir):
        atom_dataset = LMDBDataset(self.data_dir / split_dir)
        return VoxelizedDataset(atom_dataset, self.make_image)

class VoxelizedDataset(Dataset):

    def __init__(self, raw_dataset, make_image):
        self.raw_dataset = raw_dataset
        self.make_image = make_image

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, epoch_i):
        epoch, i = epoch_i
        item = self.raw_dataset[i]

        # Sample a new random orientation for each epoch.

        rng = np.random.default_rng(abs(hash(epoch_i)))

        ligand_coords_i = item['atoms_ligand'][['x', 'y', 'z']].astype(np.float32)
        ligand_center_i = get_center(ligand_coords_i)

        frame_ix = sample_coord_frame(rng, ligand_center_i)

        # I can't find the definition of the pocket, so I'm not 100% convinced 
        # that it really contains all the atoms in the vicinity of the ligand.  
        # But the example code uses the pocket atoms after a random rotation 
        # around the ligand center, so I'll just do the same.

        atoms_i = pd.concat([item['atoms_pocket'], item['atoms_ligand']])
        atoms_x = transform_atom_coords(atoms_i, frame_ix)

        img = self.make_image(atoms_x)
        img = torch.from_numpy(img).float()

        label = torch.tensor(item['scores']['neglog_aff']).reshape(1)

        return img, label

class EpochSampler:

    def __init__(self, indices, *, curr_epoch=0):
        self.indices = indices
        self.curr_epoch = curr_epoch

    def __iter__(self):
        for i in self.indices:
            yield self.curr_epoch, i

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

def make_point_image(atoms, grid_config):
    grid = get_grid(
            atoms,
            center=np.zeros(3),
            config=grid_config,
    )

    # Last dimension is atom channel, so we need to move it to the front
    # per pytorch style
    grid = np.moveaxis(grid, -1, 0)

    return grid

def make_sphere_image(atoms, img_params):
    return image_from_atoms(atoms, img_params)


def get_default_data():
    return VoxelizedDataModule(
            **get_default_data_hparams(),
    )

def get_default_data_dir():
    return os.environ['ATOM3D_LBA_DATA_DIR']

def get_default_data_hparams():
    return dict(
            data_dir=get_default_data_dir(),
            make_image=partial(
                make_point_image,
                grid_config=get_default_point_image_config(),
            ),
    )

def get_default_point_image_config():
    return dotdict({
        'element_mapping': {
            'H': 0,
            'C': 1,
            'O': 2,
            'N': 3,
            'F': 4,
        },
        'radius_A': 10.0,
        'resolution_A': 1.0,
        'num_directions': 20,
        'num_rolls': 20,
    })

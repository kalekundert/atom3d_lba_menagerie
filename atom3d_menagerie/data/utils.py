import lightning.pytorch as L
import polars as pl
import numpy as np
import os

from atom3d.datasets import LMDBDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
from pathlib import Path

class LmdbDataModule(L.LightningDataModule):

    def __init__(
            self,
            *,
            data_dir,
            make_inputs,
            batch_size,
            shuffle=True,
            num_workers=None,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.make_inputs = make_inputs
        self.dataloader_kwargs = dict(
                batch_size=batch_size,
                num_workers=(
                    num_workers or int(os.getenv('SLURM_JOB_CPUS_PER_NODE', 1))
                ),
                drop_last=True,
                **kwargs,
        )
        self.shuffle = shuffle

    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = self._make_dataset('train')
            self.val_dataset = self._make_dataset('val')

        if stage == 'test':
            self.test_dataset = self._make_dataset('test')

    def train_dataloader(self):
        return self._make_dataloader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self):
        return self._make_dataloader(self.val_dataset)

    def test_dataloader(self):
        return self._make_dataloader(self.test_dataset)

    def _make_dataset(self, split_dir):
        atom_dataset = LMDBDataset(self.data_dir / split_dir)
        return AugmentedDataset(atom_dataset, self.make_inputs)

    def _make_dataloader(self, dataset, shuffle=False):
        kwargs = self.dataloader_kwargs.copy()
        indices = self._pick_indices(dataset, shuffle)
        sampler = EpochSampler(indices)
        return DataLoader(dataset, sampler=sampler, **kwargs)

    def _pick_indices(self, dataset, shuffle):
        indices = range(len(dataset))
        if shuffle:
            indices = RandomSampler(indices)
        return indices


class AugmentedDataset(Dataset):

    def __init__(self, raw_dataset, make_inputs):
        self.raw_dataset = raw_dataset
        self.make_inputs = make_inputs

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, epoch_i):
        epoch, i = epoch_i
        item = self.raw_dataset[i]
        rng = np.random.default_rng(abs(hash(epoch_i)))
        return self.make_inputs(rng, item)

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



def convert_to_mmdf_format(atoms, **extra_cols):
    # Most of these columns aren't needed for training; they're only needed for 
    # generating mmCIF output that can be nicely visualized in PyMOL via 
    # `mmdf.write_mmcif()`.  Extracting these columns is a bit inefficient, but 
    # it makes debugging easier.
    return (
            pl.from_pandas(atoms)
            .select(
                chain_id=pl.col('chain').cast(str),
                subchain_id=pl.col('segid').cast(str).str.strip_chars().replace('', 'A'),
                alt_id=pl.col('altloc').cast(str).str.strip_chars().replace('', '.'),
                seq_id=pl.col('residue').cast(int),
                comp_id=pl.col('resname').cast(str),
                atom_id=pl.col('name').cast(str),
                element=pl.col('element').cast(str),
                x=pl.col('x').cast(float),
                y=pl.col('y').cast(float),
                z=pl.col('z').cast(float),
                occupancy=pl.col('occupancy').cast(float),
                b_factor=pl.col('bfactor').cast(float),
            )
            .with_columns(
                **extra_cols,
            )
    )


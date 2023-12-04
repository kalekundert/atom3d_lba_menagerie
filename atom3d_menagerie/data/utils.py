import lightning.pytorch as pl
import numpy as np
import os

from atom3d.datasets import LMDBDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler
from pathlib import Path

class LmdbDataModule(pl.LightningDataModule):

    def __init__(
            self,
            *,
            data_dir,
            make_inputs,
            batch_size,
            shuffle,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.make_inputs = make_inputs
        self.dataloader_kwargs = dict(
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=int(os.getenv('SLURM_JOB_CPUS_PER_NODE', 1)),
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
        return AugmentedDataset(atom_dataset, self.make_inputs)

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



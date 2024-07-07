import torch
import polars as pl
import numpy as np
import macromol_dataframe as mmdf
import macromol_voxelize as mmvox
import os
import re

from atom3d_menagerie.data.utils import LmdbDataModule, convert_to_mmdf_format
from atompaint.transform_pred.datasets.utils import sample_coord_frame
from macromol_voxelize import Grid
from dataclasses import dataclass
from functools import partial
from pipeline_func import f
from pathlib import Path

class VoxelizedMspDataModule(LmdbDataModule):

    def __init__(self, *, img_params, **kwargs):
        super().__init__(
                make_inputs=partial(make_msp_inputs, img_params=img_params),
                **kwargs,
        )

    def _pick_indices(self, dataset, shuffle):
        indices = range(len(dataset))

        if shuffle:
            labels = [int(item['label']) for item in dataset.raw_dataset]
            classes, class_sample_count = np.unique(labels, return_counts=True)
            # Weighted sampler for imbalanced classification (1:1 ratio for each class)
            weight = 1. / class_sample_count
            sample_weights = torch.tensor([weight[t] for t in labels])
            indices = torch.utils.data.WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(dataset),
                    replacement=True,
            )

        return indices


@dataclass
class ImageParams:
    grid: Grid
    atom_radius_A: float
    element_channels: list[str]
    mutation_channel: bool

@dataclass
class Tag:
    entry_id: str
    chain_1: str
    chain_2: str
    chain_mut: str
    resn_mut: str
    resn_wt: str
    resi: int

def parse_tag(tag):
    pattern = r'''(?x)
            (?P<entry_id>\w{4})_
            (?P<chain_1>[A-Z]+)_
            (?P<chain_2>[A-Z]+)_
            (?P<resn_wt>[A-Z])
            (?P<chain_mut>[A-Z])
            (?P<resi>[0-9]+)
            (?P<resn_mut>[A-Z])'''

    match = re.match(pattern, tag)
    if match is None:
        raise ValueError(f"found tag with unexpected format: {tag}")

    return Tag(
            entry_id=match['entry_id'],
            chain_1=match['chain_1'],
            chain_2=match['chain_2'],
            chain_mut=match['chain_mut'],
            resn_mut=match['resn_mut'],
            resn_wt=match['resn_wt'],
            resi=int(match['resi']),
    )

def make_msp_inputs(rng, item, img_params):
    tag = parse_tag(item['id'])

    atoms_i = {
            k: convert_to_mmdf_format(
                item[k],
                is_mutation=(
                    (pl.col('seq_id') == tag.resi) &
                    (pl.col('chain_id') == tag.chain_mut)
                )
            )
            for k in ['original_atoms', 'mutated_atoms']
    }
    center_i = mmdf.get_atom_coords(
            atoms_i['original_atoms']
            .filter('is_mutation')
            .mean()
    )
    frame_ix = sample_coord_frame(rng, center_i)
    atoms_x = {
            k: (
                atoms_i[k]
                | f(mmdf.transform_atom_coords, frame_ix)
                | f(mmdf.prune_hydrogen)
                | f(mmdf.prune_water)
            )
            for k in atoms_i
    }

    def process_filtered_atoms(atoms):
        channels = img_params.element_channels
        atoms = mmvox.set_atom_radius_A(atoms, img_params.atom_radius_A)
        atoms = mmvox.set_atom_channels_by_element(atoms, channels)

        if img_params.mutation_channel:
            atoms = mmvox.add_atom_channel_by_expr(atoms, 'is_mutation', len(channels))

        return atoms

    mmvox_img_params = mmvox.ImageParams(
            channels=(
                len(img_params.element_channels) +
                img_params.mutation_channel
            ),
            grid=img_params.grid,
            process_filtered_atoms=process_filtered_atoms,
            max_radius_A=img_params.atom_radius_A,
    )
    img_pair = np.stack([
            mmvox.image_from_atoms(df, mmvox_img_params)
            for df in atoms_x.values()
    ])
    label = int(item['label'])

    return torch.from_numpy(img_pair).float(), torch.tensor(label).float()

def get_default_msp_data_dir():
    return Path(os.environ['ATOM3D_MSP_DATA_DIR'])


if __name__ == '__main__':
    dm = VoxelizedMspDataModule(
            data_dir=os.environ['ATOM3D_MSP_DATA_DIR'],
            img_params=ImageParams(
                grid=Grid(
                    length_voxels=20,
                    resolution_A=1,
                ),
                atom_radius_A=0.5,
                element_channels=[['C'], ['N'], ['O'], ['S', 'SE'], ['P'], ['*']],
                mutation_channel=True,
            ),
            batch_size=1,
            shuffle=False,
    )
    dm.setup('fit')

    dl = dm.train_dataloader()

    x, y = next(iter(dl))
    print(x.shape, y)



from atom3d_lba_menagerie.hparams import label_hparams
from dataclasses import dataclass

def test_label_hparams_str_int():
    assert label_hparams('i_{0}', 1, 2) == {'i_1': 1, 'i_2': 2}

def test_label_hparams_str_dataclass():
    @dataclass
    class HParams:
        x: int
        y: int

    assert label_hparams(
            'x_{x}_y_{y}',
            HParams(1, 2),
            HParams(3, 4),
    ) == {
            'x_1_y_2': HParams(1, 2),
            'x_3_y_4': HParams(3, 4),
    }

def test_label_hparams_callable():
    k = lambda i: f'i_{i + 1}'
    assert label_hparams(k, 1, 2) == {'i_2': 1, 'i_3': 2}


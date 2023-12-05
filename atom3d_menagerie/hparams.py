from dataclasses import asdict
from itertools import product
from functools import partial
from collections.abc import Mapping

def make_hparams(factory, **kwargs):
    hparams = []
    keys = kwargs.keys()

    for values in product(*kwargs.values()):
        factory_kwargs = dict(zip(keys, values))
        hparams.append(factory(**factory_kwargs))

    return hparams

def label_hparams(key, *hparams):
    if isinstance(key, str):
        key = partial(interpolate, key)

    assert callable(key)

    return {
            key(x): x
            for x in hparams
    }

def require_hparams(key, hparams):
    if not key:
        for known_key in hparams:
            print(known_key)
        raise SystemExit

    else:
        print('Hyperparameters:', x := hparams[key])
        return x

def interpolate(template, obj):
    try:
        obj = asdict(obj)
    except TypeError:
        pass

    if isinstance(obj, Mapping):
        return template.format_map(obj)
    else:
        return template.format(obj)

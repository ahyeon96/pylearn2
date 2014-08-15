import numpy as np
import pandas as pd
import os
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.rng import make_np_rng


class Higgs(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, path, which_set, seed, frac_train, **kwargs):
        self.kwargs = kwargs
        self.seed = seed
        self.frac_train = frac_train

        if which_set not in ['train', 'valid']:
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","valid"].')
        if not os.path.isfile(path):
            raise ValueError('file not found: '+str(path))
        X = pd.read_hdf(path, key='X')
        y = pd.read_hdf(path, key='y').values[...,np.newaxis]

        n_ex = X.shape[0]
        rng = np.random.RandomState(seed)
        order = rng.permutation(n_ex)
        n_train = int(frac_train*n_ex)

        assert n_ex == y.shape[0]
        assert frac_train > 0. and frac_train <= 1.

        nan_fill = X.replace(-999., np.nan)
        train_idx = order[:n_train]
        valid_idx = order[n_train:]
        train_mean = nan_fill.loc[train_idx].mean(axis=0).values
        train_std = nan_fill.loc[train_idx].std(axis=0).values

        X = X.values

        if which_set == 'train':
            X = X[train_idx]
        else:
            X = X[valid_idx]

        X = X-train_mean
        X = X/train_std

        super(Higgs, self).__init__(X=X.astype(np.float32), y=y.astype(np.int8), **kwargs)

    def get_valid_set(self):
        return Higgs(self.path, 'valid', self.seed, self.frac_train, **self.kwargs)

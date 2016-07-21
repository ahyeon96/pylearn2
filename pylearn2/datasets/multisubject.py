"""
Multisubject dataset wraps dense_design_matrix
"""
__authors__ = "Jesse Livezey"
import functools

import logging
import warnings

import numpy as np
from theano.compat.six.moves import xrange

from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)
from pylearn2.utils.multisubject_iteration import MultiSubjectDatasetIterator

from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.space import CompositeSpace
from pylearn2.utils import safe_izip
from pylearn2.utils.rng import make_np_rng


logger = logging.getLogger(__name__)


class MultiSubject(Dataset):

    """
    A class for wrapping multiple dense design matrix instances.

    Parameters
    ----------
    datasets : list or tuple
        Iterable of dataset objects.
    """
    _default_seed = (18, 2, 946)

    def __init__(self, datasets, rng=_default_seed):
        self.datasets = datasets
        spaces = []
        sources = []
        for ds in datasets:
            assert isinstance(ds, DenseDesignMatrix)
            spaces.append(ds.data_specs[0])
            sources.append(ds.data_specs[1])

        self.data_specs = (CompositeSpace(spaces), tuple(sources))

        self.rng = make_np_rng(rng, which_method="random_integers")
        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('random_uniform')

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None,
                 return_tuple=False):

        iterators = []
        
        if data_specs is not None:
            spaces = data_specs[0]
            sources = data_specs[1]
            data_specs = zip(spaces.components, sources)
        else:
            data_specs = len(self.datasets) * [None]

        dataset_data_specs = []
        for ii, ds in enumerate(self.datasets):
            ds_i_spaces = []
            ds_i_sources = []
            for d_s in data_specs:
                sp, so = d_s
                if str(ii) in so:
                    ds_i_spaces.append(sp)
                    ds_i_sources.append(so)
            dataset_data_specs.append((CompositeSpace(tuple(ds_i_spaces)), tuple(ds_i_sources)))

        for dataset, data_spec in safe_izip(self.datasets, dataset_data_specs):
            iterators.append(dataset.iterator(mode,
                                              batch_size=batch_size,
                                              num_batches=num_batches,
                                              rng=rng,
                                              data_specs=data_spec,
                                              return_tuple=return_tuple))

        return MultiSubjectDatasetIterator(iterators)

    def get_num_examples(self):
        return max([ds.get_num_examples() for ds in self.datasets])
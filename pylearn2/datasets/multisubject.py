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
    _default_seed = (17, 2, 946)

    def __init__(self, datasets):
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
            data_specs = zip(spaces, sources)
        else:
            data_specs = len(self.datasets) * [None]

        for dataset, data_spec in safe_izip(self.datasets, data_specs):
            iterators.append(dataset.iterator(mode,
                                              batch_size=batch_size,
                                              num_batches=num_batches,
                                              rng=rng,
                                              data_specs=data_spec,
                                              return_tuple=return_tuple))

        return MultiSubjectDatasetIterator(iterators)

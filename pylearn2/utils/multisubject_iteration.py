"""
Iterators providing indices for different kinds of iteration over
datasets.

Presets:

- sequential: iterates through fixed slices of the dataset in sequence
- shuffled_sequential: iterates through a shuffled version of the dataset
  in sequence
- random_slice: on each call to next, returns a slice of the dataset,
  chosen uniformly at random over contiguous slices.
  Samples with replacement, but still reports that
  container is empty after num_examples / batch_size calls
- random_uniform: on each call to next, returns a random subset of the
  dataset. Samples with replacement, but still reports that
  container is empty after num_examples / batch_size calls
"""
from __future__ import division

import warnings
import numpy as np
from theano.compat import six

from pylearn2.utils.iteration import FiniteDatasetIterator, SubsetIterator
from pylearn2.utils import safe_izip, wraps


class MultiSubjectDatasetIterator(object):
    """
    A wrapper around multiple FiniteDatasetIterators.

    Parameters
    ----------
    datasets : list of `Dataset` object
        List of the dataset over which to iterate.
    subset_iterator : object
        An iterator object that returns slice objects or lists of
        examples, conforming to the interface specified by
        :py:class:`SubsetIterator`.
    data_specs : tuple
        A `(space, source)` tuple. See :ref:`data_specs` for a full
        description. Must not contain nested composite spaces.
    return_tuple : bool, optional
        Always return a tuple, even if there is exactly one source
        of data being returned. Defaults to `False`.
    convert : list of callables
        A list of callables, in the same order as the sources
        in `data_specs`, that will be called on the individual
        source batches prior to any further processing.
    """
    stochastic = True

    def __init__(self, iterators):
        self._iterators = iterators
        assert all([isinstance(it, FiniteDatasetIterator) for it in iterators])

    def __iter__(self):
        return self

    @wraps(SubsetIterator.next)
    def next(self):
        """
        Retrieves the next batch of examples.

        Returns
        -------
        next_batch : object
            An object representing a mini-batch of data, conforming
            to the space specified in the `data_specs` constructor
            argument to this iterator. Will be a tuple if more
            than one data source was specified or if the constructor
            parameter `return_tuple` was `True`.

        Raises
        ------
        StopIteration
            When there are no more batches to return.
        """
        features = []
        targets = []
        for it in self._iterators:
            X, Y = it.next()
            features.append(X)
            targets.append(Y)
        return tuple(features + targets)

    def __next__(self):
        return self.next()

    @property
    @wraps(SubsetIterator.num_examples, assigned=(), updated=())
    def num_examples(self):
        num_ex = [it.num_examples for it in self._iterators]
        assert len(set(num_ex)) == 1
        return num_ex[0]

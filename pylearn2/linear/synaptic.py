"""
.. todo::

    WRITEME
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

from theano.compat.six.moves import xrange
from theano import tensor as T

from pylearn2.linear.linear_transform import LinearTransform
import functools
import numpy as np
from pylearn2.utils import sharedX
from pylearn2.utils.rng import make_np_rng

class Tanh(object):
    def __call__(self, x):
        return T.tanh(x)

class Sigmoid(object):
    def __call__(self, x):
        return T.nnet.sigmoid(x)

class Rectified(object):
    def __call__(self, x):
        return T.switch(x>0, x, 0.)

class Synaptic(LinearTransform):
    """
    """

    def __init__(self, W, nonlinearity):
        """
        Sets the initial values of the matrix
        """
        self._W = W
        self.nonlinearity = nonlinearity

    @functools.wraps(LinearTransform.get_params)
    def get_params(self):
        """
        .. todo::

            WRITEME
        """
        return [self._W]

    def lmul(self, x):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        x : ndarray, 1d or 2d
            The input data
        """
        linear = x.dimshuffle(0, 1, 'x')*self._W.dimshuffle('x', 0, 1) 
        synaptic = self.nonlinearity(linear)
        return synaptic.sum(1)

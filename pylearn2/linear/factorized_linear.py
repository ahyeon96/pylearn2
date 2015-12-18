import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d

from pylearn2.linear.linear_transform import LinearTransform
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.linear.conv2d import Conv2D


class FactorizedMatrixMul(MatrixMul):
    """
    Matrix multiply with the weight matrix factorized by the
    the topological shape on the input.
    Parameters
    ----------
    zero : WRITEME
    one : shared variable
    c : shared variable
    dim : int
    """

    def __init__(self, out_dim, zero=None, one=None, c=None, axes=('b', 0, 1, 'c'):
        """
        Sets the initial values of the matrix
        """
        params = [zero, one, c]
        self.params = [param for param in params if param is not None]
        assert len(self.params) > 0
        self._dim = out_dim
        if zero is not None:
            zero = zero.dimshuffle('x', 'x', 0, 1)
        if one is not None:
            one = one.dimshuffle('x', 0, 'x', 1)
        if c is not None:
            c = c.dimshuffle(0, 'x', 'x', 1)
        reshaped_params = [zero, one, c]
        reshaped_params = [param for param in reshaped_params if param is not None]
        W = 1.
        for param in reshaped_params:
            W = W*param
        self._W = W.reshape((-1, out_dim))

    @functools.wraps(LinearTransform.get_params)
    def get_params(self):
        return self.params

    def get_weights(self):
        return [self._W]


class FactorizedConv2D(Conv2D):
    """
    Convolution with factorized filters
    """

    def __init__(self, zero=None, one=None, c=None, axes=('b', 0, 1, 'c'):
        """
        Sets the initial values of the matrix
        """
        params = [zero, one, c]
        self.params = [param for param in params if param is not None]
        if len(self.params) > 0:
        self._dim = out_dim
        if zero is not None:
            zero = zero.dimshuffle(0, 'x', 1, 'x')
        if one is not None:
            one = one.dimshuffle(0, 'x', 'x', 1)
        if c is not None:
            c = c.dimshuffle(0, 1, 'x', 'x')
        reshaped_params = [zero, one, c]
        reshaped_params = [param for param in reshaped_params if param is not None]
        cfilter = 1.
        for param in reshaped_params:
            cfilter = cfilter*param
        self._filters = cfilter

    @functools.wraps(P2LT.get_params)
    def get_params(self):
        return self.params

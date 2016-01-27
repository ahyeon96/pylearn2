"""
Multi-scale convolutions
"""
__authors__ = "Jesse Livezey"

import logging
import math
import operator
import sys
import warnings

import numpy as np
from theano.compat import six
from theano.compat.six.moves import reduce, xrange
from theano import config
from theano.gof.op import get_debug_values
from theano.sandbox.cuda import cuda_enabled
from theano.sandbox.cuda.dnn import dnn_available, dnn_pool
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.signal.downsample import max_pool_2d
import theano.tensor as T
from theano.tensor.nnet.abstract_conv import get_conv_output_shape

from pylearn2.compat import OrderedDict
from pylearn2.costs.mlp import Default
from pylearn2.models.mlp import ConvElemwise
# Try to import the fast cudnn library, else fallback to conv2d
if cuda_enabled and dnn_available():
    try:
        from pylearn2.linear import cudnn2d as conv2d
    except ImportError:
        from pylearn2.linear import conv2d
else:
    from pylearn2.linear import conv2d
from pylearn2.model_extensions.norm_constraint import MaxL2FilterNorm
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.utils import safe_union
from pylearn2.utils import safe_zip
from pylearn2.utils import safe_izip
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.utils import contains_inf
from pylearn2.utils import isfinite
from pylearn2.utils.data_specs import DataSpecsMapping


logger = logging.getLogger(__name__)


class MultiScaleConvElemwise(ConvElemwise):
    """
    Generic multiscale convolutional elemwise layer.
    Takes the ConvNonlinearity object as an argument and implements
    a stack of convolutional layers with the specified nonlinearity
    at different scales.

    Parameters
    ----------
    output_channels : int
        The number of output channels per stack the layer should have.
    kernel_shape_min : tuple
        The shape of the smallest convolution kernel. All shapes must
        be odd.
    kernel_shape_step : tuple
        The step of kernel sizes. Must be even.
    n_scales : int 
        The number of scales.
    pool_shape : tuple
        The shape of the spatial max pooling. A two-tuple of ints.
    pool_stride : tuple
        The stride of the spatial max pooling. Also must be square.
    layer_name : str
        A name for this layer that will be prepended to monitoring channels
        related to this layer.
    nonlinearity : object
        An instance of a nonlinearity object which might be inherited
        from the ConvNonlinearity class.
    irange : float, optional
        if specified, initializes each weight randomly in
        U(-irange, irange)
    istdev : float, optional
        if specified, initializes each weight randomly in
        N(0, istdev)
    border_mode : str, optional
        A string indicating the size of the output:

          - "full" : The output is the full discrete linear convolution of the
            inputs.
          - "half" : The output is the half discrete linear convolution of the
            inputs.
          - "valid" : The output consists only of those elements that do not
            rely on the zero-padding. (Default)
    sparse_init : WRITEME
    include_prob : float, optional
        probability of including a weight element in the set of weights
        initialized to U(-irange, irange). If not included it is initialized
        to 1.0.
    init_bias : float, optional
        All biases are initialized to this number. Default is 0.
    W_lr_scale : float or None
        The learning rate on the weights for this layer is multiplied by this
        scaling factor
    b_lr_scale : float or None
        The learning rate on the biases for this layer is multiplied by this
        scaling factor
    max_kernel_norm : float or None
        If specified, each kernel is constrained to have at most this norm.
    pool_type : str or None
        The type of the pooling operation performed the convolution.
        Default pooling type is max-pooling.
    tied_b : bool, optional
        If true, all biases in the same channel are constrained to be the
        same as each other. Otherwise, each bias at each location is
        learned independently. Default is true.
    detector_normalization : callable or None
        See `output_normalization`.
        If pooling argument is not provided, detector_normalization
        is not applied on the layer.
    output_normalization : callable  or None
        if specified, should be a callable object. the state of the
        network is optionally replaced with normalization(state) at each
        of the 3 points in processing:

          - detector: the maxout units can be normalized prior to the
            spatial pooling
          - output: the output of the layer, after sptial pooling, can
            be normalized as well
    subsample : 2-tuple of ints, optional
        The stride of the convolution kernel. Default is (1, 1).
    """

    def __init__(self,
                 output_channels,
                 kernel_shape,
                 layer_name,
                 nonlinearity,
                 irange=None,
                 istdev=None,
                 border_mode='valid',
                 sparse_init=None,
                 include_prob=1.0,
                 init_bias=0.,
                 W_lr_scale=None,
                 b_lr_scale=None,
                 max_kernel_norm=None,
                 pool_type=None,
                 pool_shape=None,
                 pool_stride=None,
                 tied_b=None,
                 detector_normalization=None,
                 output_normalization=None,
                 subsample=(1, 1),
                 monitor_style="classification"):

        if (irange is None) and (istdev is None) and (sparse_init is None):
            raise AssertionError("You should specify either irange, istdev, or "
                                 "sparse_init when calling the constructor of "
                                 "ConvElemwise.")
        elif (irange is not None) and (istdev is not None) and (sparse_init is not None):
            raise AssertionError("You should specify either irange, istdev, or "
                                 "sparse_init when calling the constructor of "
                                 "ConvElemwise and not both.")

        if pool_type is not None:
            assert pool_shape is not None, (
                "You should specify the shape of the spatial %s-pooling." %
                pool_type)
            assert pool_stride is not None, (
                "You should specify the strides of the spatial %s-pooling." %
                pool_type)

        assert nonlinearity is not None

        # Default behavior
        if tied_b is None:
            tied_b = True

        super(ConvElemwise, self).__init__()
        self.nonlin = nonlinearity
        self.__dict__.update(locals())
        assert monitor_style in ['classification', 'detection'], (
            "%s.monitor_style should be either"
            "detection or classification" % self.__class__.__name__)
        del self.self

        if max_kernel_norm is not None:
            self.extensions.append(
                MaxL2FilterNorm(max_kernel_norm, axis=(1, 2, 3))
            )

    def initialize_transformer(self, rng):
        """
        This function initializes the transformer of the class. Re-running
        this function will reset the transformer.

        Parameters
        ----------
        rng : object
            random number generator object.
        """
        if ((self.irange is not None) or
            (self.istdev is not None)):
            assert self.sparse_init is None
            assert not ((self.irange is not None) and
                        (self.istdev is not None))

            self.transformer = conv2d.make_random_conv2D(
                irange=self.irange,
                istdev=self.istdev,
                input_space=self.input_space,
                output_space=self.detector_space,
                kernel_shape=self.kernel_shape,
                subsample=self.subsample,
                border_mode=self.border_mode,
                rng=rng)
        elif self.sparse_init is not None:
            self.transformer = conv2d.make_sparse_random_conv2D(
                num_nonzero=self.sparse_init,
                input_space=self.input_space,
                output_space=self.detector_space,
                kernel_shape=self.kernel_shape,
                subsample=self.subsample,
                border_mode=self.border_mode,
                rng=rng)
        else:
            raise ValueError('irange and sparse_init cannot be both None')

    def initialize_output_space(self):
        """
        Initializes the output space of the ConvElemwise layer by taking
        pooling operator and the hyperparameters of the convolutional layer
        into consideration as well.
        """
        dummy_batch_size = self.mlp.batch_size

        if dummy_batch_size is None:
            dummy_batch_size = 2
        dummy_detector =\
            sharedX(self.detector_space.get_origin_batch(dummy_batch_size))

        if self.pool_type is not None:
            assert self.pool_type in ['max', 'mean']
            if self.pool_type == 'max':
                dummy_p = max_pool(bc01=dummy_detector,
                                   pool_shape=self.pool_shape,
                                   pool_stride=self.pool_stride,
                                   image_shape=self.detector_space.shape)
            elif self.pool_type == 'mean':
                dummy_p = mean_pool(bc01=dummy_detector,
                                    pool_shape=self.pool_shape,
                                    pool_stride=self.pool_stride,
                                    image_shape=self.detector_space.shape)
            dummy_p = dummy_p.eval()
            self.output_space = Conv2DSpace(shape=[dummy_p.shape[2],
                                                   dummy_p.shape[3]],
                                            num_channels=self.output_channels,
                                            axes=('b', 'c', 0, 1))
        else:
            dummy_detector = dummy_detector.eval()
            self.output_space = Conv2DSpace(shape=[dummy_detector.shape[2],
                                            dummy_detector.shape[3]],
                                            num_channels=self.output_channels,
                                            axes=('b', 'c', 0, 1))

        logger.info('Output space: {0}'.format(self.output_space.shape))

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        """ Note: this function will reset the parameters! """

        self.input_space = space

        if not isinstance(space, Conv2DSpace):
            raise BadInputSpaceError(self.__class__.__name__ +
                                     ".set_input_space "
                                     "expected a Conv2DSpace, got " +
                                     str(space) + " of type " +
                                     str(type(space)))

        rng = self.mlp.rng

        output_shape = get_conv_output_shape((None, None) +
                                             self.input_space.shape,
                                             [None, None] +
                                             self.kernel_shape,
                                             self.border_mode,
                                             self.subsample)
        output_shape = output_shape[2:]

        self.detector_space = Conv2DSpace(shape=output_shape,
                                          num_channels=self.output_channels,
                                          axes=('b', 'c', 0, 1))

        self.initialize_transformer(rng)

        W, = self.transformer.get_params()
        W.name = self.layer_name + '_W'

        if self.tied_b:
            self.b = sharedX(np.zeros((self.detector_space.num_channels)) +
                             self.init_bias)
        else:
            self.b = sharedX(self.detector_space.get_origin() + self.init_bias)

        self.b.name = self.layer_name + '_b'

        logger.info('Input shape: {0}'.format(self.input_space.shape))
        logger.info('Detector space: {0}'.format(self.detector_space.shape))

        self.initialize_output_space()

    @wraps(Layer.get_params)
    def get_params(self):
        assert self.b.name is not None
        W, = self.transformer.get_params()
        assert W.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * T.sqr(W).sum()

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W, = self.transformer.get_params()
        return coeff * abs(W).sum()

    @wraps(Layer.set_weights)
    def set_weights(self, weights):

        W, = self.transformer.get_params()
        W.set_value(weights)

    @wraps(Layer.set_biases)
    def set_biases(self, biases):

        self.b.set_value(biases)

    @wraps(Layer.get_biases)
    def get_biases(self):

        return self.b.get_value()

    @wraps(Layer.get_weights_format)
    def get_weights_format(self):

        return ('v', 'h')

    @wraps(Layer.get_lr_scalers)
    def get_lr_scalers(self):
        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    @wraps(Layer.get_weights_topo)
    def get_weights_topo(self):

        outp, inp, rows, cols = range(4)
        raw = self.transformer._filters.get_value()

        return np.transpose(raw, (outp, rows, cols, inp))

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):

        W, = self.transformer.get_params()

        assert W.ndim == 4

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=(1, 2, 3)))

        rval = OrderedDict([
                           ('kernel_norms_min', row_norms.min()),
                           ('kernel_norms_mean', row_norms.mean()),
                           ('kernel_norms_max', row_norms.max()),
                           ])

        cst = self.cost
        orval = self.nonlin.get_monitoring_channels_from_state(state,
                                                               targets,
                                                               cost_fn=cst)

        rval.update(orval)

        return rval

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        self.input_space.validate(state_below)

        z = self.transformer.lmul(state_below)
        if not hasattr(self, 'tied_b'):
            self.tied_b = False

        if self.tied_b:
            b = self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            b = self.b.dimshuffle('x', 0, 1, 2)

        z = z + b
        d = self.nonlin.apply(z)

        if self.layer_name is not None:
            d.name = self.layer_name + '_z'
            self.detector_space.validate(d)

        if self.pool_type is not None:
            # Format the input to be supported by max pooling
            if not hasattr(self, 'detector_normalization'):
                self.detector_normalization = None

            if self.detector_normalization:
                d = self.detector_normalization(d)

            assert self.pool_type in ['max', 'mean'], ("pool_type should be"
                                                       "either max or mean"
                                                       "pooling.")

            if self.pool_type == 'max':
                p = max_pool(bc01=d, pool_shape=self.pool_shape,
                             pool_stride=self.pool_stride,
                             image_shape=self.detector_space.shape)

            elif self.pool_type == 'mean':
                p = mean_pool(bc01=d, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)

            self.output_space.validate(p)
        else:
            p = d

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p

    @wraps(Layer.cost, append=True)
    def cost(self, Y, Y_hat):
        """
        Notes
        -----
        The cost method calls `self.nonlin.cost`
        """

        batch_axis = self.output_space.get_batch_axis()
        return self.nonlin.cost(Y=Y, Y_hat=Y_hat, batch_axis=batch_axis)

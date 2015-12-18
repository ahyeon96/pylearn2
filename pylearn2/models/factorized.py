from pylearn2.compat import OrderedDict
from pylearn2.linear.matrixmul import FactorizedMatrixMul
from pylearn2.linear.conv2d import FactorizedConv2D
from pylearn2.models import mlp
from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2.model_extensions.norm_constraint import MaxL2FilterNorm
from pylearn2.utils import sharedX, wraps
from pylearn2.expr.nnet import multi_class_prod_misclass

import theano.tensor as T
import numpy as np


class FactorizedLinear(mlp.Linear):
    @wraps(mlp.Layer.set_input_space)
    def set_input_space(self, space):
        assert isinstance(space, Conv2DSpace)

        self.input_space = space
        self.requires_reformat = True
        self.input_dim = space.get_total_dimension()
        self.desired_space = VectorSpace(self.input_dim)
        self.output_space = VectorSpace(self.dim)

        rng = self.mlp.rng
        assert self.sparse_init is None
        if space.num_channels > 1:
            c = rng.randn(space.num_channels, self.dim) * self.istdev
            c = sharedX(c)
            c.name = self.layer_name + '_c'
        else:
            c = None
        if space.shape[0] > 1:
            zero = rng.randn(space.shape[0], self.dim) * self.istdev
            zero = sharedX(zero)
            zero.name = self.layer_name + '_zero'
        else:
            zero = None
        if space.shape[1] > 1:
            one = rng.randn(space.shape[1], self.dim) * self.istdev
            one = sharedX(one)
            one.name = self.layer_name + '_one'
        else:
            one = None

        if all([param is None for param in [zero, one, c]]):
            c = rand_func(space.num_channels, self.dim) * self.istdev
            c = sharedX(c)
            c.name = self.layer_name + '_c'

        self.transformer = FactorizedMatrixMul(self.dim, zero, one, c)

        params = self.transformer.get_params()
        for param in params:
            assert param.name is not None

        if self.mask_weights is not None:
            raise NotImplementedError

    @wraps(mlp.Layer.get_params)
    def get_params(self):

        params = self.transformer.get_params()
        for param in params:
            assert param.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        if self.use_bias:
            assert self.b.name is not None
            assert self.b not in rval
            rval.append(self.b)
        return rval

    @wraps(mlp.Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        params = self.transformer.get_weights()
        return sum([coeff * T.sqr(param).sum() for param in params])

    @wraps(mlp.Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        params = self.transformer.get_weights()
        return sum([coeff * abs(param).sum() for param in params])

    @wraps(mlp.Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        params = self.transformer.get_params()
        rval = OrderedDict()
        for param in params:
            assert param.ndim == 2

            sq_param = T.sqr(param)

            row_norms = T.sqrt(sq_param.sum(axis=1))
            col_norms = T.sqrt(sq_param.sum(axis=0))
            name = param.name

            rval.update({name+'_row_norms_min':  row_norms.min(),
                         name+'_row_norms_mean': row_norms.mean(),
                         name+'_row_norms_max':  row_norms.max(),
                         name+'_col_norms_min':  col_norms.min(),
                         name+'_col_norms_mean': col_norms.mean(),
                         name+'_col_norms_max':  col_norms.max()})

        if (state is not None) or (state_below is not None):
            if state is None:
                state = self.fprop(state_below)

            mx = state.max(axis=0)
            mean = state.mean(axis=0)
            mn = state.min(axis=0)
            rg = mx - mn

            rval['range_x_max_u'] = rg.max()
            rval['range_x_mean_u'] = rg.mean()
            rval['range_x_min_u'] = rg.min()

            rval['max_x_max_u'] = mx.max()
            rval['max_x_mean_u'] = mx.mean()
            rval['max_x_min_u'] = mx.min()

            rval['mean_x_max_u'] = mean.max()
            rval['mean_x_mean_u'] = mean.mean()
            rval['mean_x_min_u'] = mean.min()

            rval['min_x_max_u'] = mn.max()
            rval['min_x_mean_u'] = mn.mean()
            rval['min_x_min_u'] = mn.min()

        return rval

class Tanh(FactorizedLinear):
    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        p = T.tanh(p)
        return p

class Sigmoid(FactorizedLinear):
    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        p = T.nnet.sigmoid(p)
        return p

class RectifiedLinear(FactorizedLinear):
    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        p = T.switch(p > 0., p, 0.)
        return p


class FactorizedConvElemwise(ConvElemwise):
    """
    Generic convolutional elemwise layer.
    Takes the ConvNonlinearity object as an argument and implements
    convolutional layer with the specified nonlinearity.

    This function can implement:

    * Linear convolutional layer
    * Rectifier convolutional layer
    * Sigmoid convolutional layer
    * Tanh convolutional layer

    based on the nonlinearity argument that it recieves.

    Parameters
    ----------
    output_channels : int
        The number of output channels the layer should have.
    kernel_shape : tuple
        The shape of the convolution kernel.
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
            raise NotImplementedError
            self.extensions.append(
                MaxL2FilterNorm(max_kernel_norm, axis=(1, 2, 3))
            )

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
                                             (None, None) +
                                             self.kernel_shape,
                                             self.border_mode,
                                             self.subsample)
        output_shape = output_shape[2:]

        self.detector_space = Conv2DSpace(shape=output_shape,
                                          num_channels=self.output_channels,
                                          axes=('b', 'c', 0, 1))

        if self.irange is not None:
            assert self.sparse_init is None
            rand_func = rng.rand
        elif self.istdev is not None:
            assert self.sparse_init is None
            rand_func = rng.randn
        else:
            raise NotImplementedError

        if space.num_channels > 1:
            c = rand_func(self.output_channels, space.num_channels) * self.istdev
            c = sharedX(c)
            c.name = self.layer_name + '_c'
        else:
            c = None
        if space.shape[0] > 1:
            zero = rand_func(self.output_channels, space.shape[0]) * self.istdev
            zero = sharedX(zero)
            zero.name = self.layer_name + '_zero'
        else:
            zero = None
        if space.shape[1] > 1:
            one = rand_func(self.output_channels, space.shape[1]) * self.istdev
            one = sharedX(one)
            one.name = self.layer_name + '_one'
        else:
            one = None

        if all([param is None for param in [zero, one, c]]):
            c = rand_func(self.output_channels, space.num_channels) * self.istdev
            c = sharedX(c)
            c.name = self.layer_name + '_c'


        self.transformer = FactorizedConv2D(zero, one, c)

        params = self.transformer.get_params()
        for param in params:
            assert param.name is not None

        if self.mask_weights is not None:
            raise NotImplementedError

        if self.tied_b:
            self.b = sharedX(np.zeros((self.detector_space.num_channels)) +
                             self.init_bias)
        else:
            self.b = sharedX(self.detector_space.get_origin() + self.init_bias)

        self.b.name = self.layer_name + '_b'

        logger.info('Input shape: {0}'.format(self.input_space.shape))
        logger.info('Detector space: {0}'.format(self.detector_space.shape))

        self.initialize_output_space()

    @wraps(mlp.Layer.get_params)
    def get_params(self):

        params = self.transformer.get_params()
        for param in params:
            assert param.name is not None
        rval = self.transformer.get_params()
        assert not isinstance(rval, set)
        rval = list(rval)
        assert self.b not in rval
        rval.append(self.b)
        return rval

    @wraps(mlp.Layer.get_weight_decay)
    def get_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        params = self.transformer.get_weights()
        return sum([coeff * T.sqr(param).sum() for param in params])

    @wraps(mlp.Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeff):

        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        params = self.transformer.get_weights()
        return sum([coeff * abs(param).sum() for param in params])

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):

        params = self.transformer.get_params()

        rval = OrderedDict()

        for param in params:
            assert param.ndim == 4
            sq_p = T.sqr(param)
            name = param.name

            row_norms = T.sqrt(sq_p.sum(axis=(1, 2, 3)))

            rval.update({'kernel_norms_min'+name: row_norms.min(),
                         'kernel_norms_mean'+name: row_norms.mean(),
                         'kernel_norms_max'+name: row_norms.max()})

        cst = self.cost
        orval = self.nonlin.get_monitoring_channels_from_state(state,
                                                               targets,
                                                               cost_fn=cst)

        rval.update(orval)

        return rval

"""
Multilayer Perceptron
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"


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
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from pylearn2.compat import OrderedDict
from pylearn2.costs.mlp import Default
from pylearn2.model_extensions.norm_constraint import MaxL2FilterNorm
from pylearn2.models.mlp import MLP
from pylearn2.monitor import get_monitor_doc
from pylearn2.expr.nnet import arg_of_softmax
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace, IndexSpace
from pylearn2.utils import function
from pylearn2.utils import is_iterable
from pylearn2.utils import py_float_types
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_union
from pylearn2.utils import safe_zip
from pylearn2.utils import safe_izip
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.utils import contains_inf
from pylearn2.utils import isfinite
from pylearn2.utils.data_specs import DataSpecsMapping

from pylearn2.expr.nnet import (elemwise_kl, kl, compute_precision,
                                compute_recall, compute_f1)

# Only to be used by the deprecation warning wrapper functions
from pylearn2.costs.mlp import L1WeightDecay as _L1WD
from pylearn2.costs.mlp import WeightDecay as _WD


logger = logging.getLogger(__name__)


class MultiSubjectMLP(MLP):

    """
    A multilayer perceptron.

    Note that it's possible for an entire MLP to be a single layer of a larger
    MLP.

    Parameters
    ----------
    layers : list
        A list of Layer objects. The final layer specifies the output space
        of this MLP.
    batch_size : int, optional
        If not specified then must be a positive integer. Mostly useful if
        one of your layers involves a Theano op like convolution that
        requires a hard-coded batch size.
    nvis : int, optional
        Number of "visible units" (input units). Equivalent to specifying
        `input_space=VectorSpace(dim=nvis)`. Note that certain methods require
        a different type of input space (e.g. a Conv2Dspace in the case of
        convnets). Use the input_space parameter in such cases. Should be
        None if the MLP is part of another MLP.
    input_space : Space object, optional
        A Space specifying the kind of input the MLP accepts. If None,
        input space is specified by nvis. Should be None if the MLP is
        part of another MLP.
    input_source : string or (nested) tuple of strings, optional
        A (nested) tuple of strings specifiying the input sources this
        MLP accepts. The structure should match that of input_space. The
        default is 'features'. Note that this argument is ignored when
        the MLP is nested.
    target_source : string or (nested) tuple of strings, optional
        A (nested) tuple of strings specifiying the target sources this
        MLP accepts. The structure should match that of target_space. The
        default is 'targets'. Note that this argument is ignored when
        the MLP is nested.
    layer_name : name of the MLP layer. Should be None if the MLP is
        not part of another MLP.
    seed : WRITEME
    monitor_targets : bool, optional
        Default: True
        If true, includes monitoring channels that are functions of the
        targets. This can be disabled to allow monitoring on monitoring
        datasets that do not include targets.
    kwargs : dict
        Passed on to the superclass.
    """

    def __init__(self, layers, batch_size=None, input_space=None,
                 input_source='features', target_source='targets',
                 nvis=None, seed=None, layer_name=None, monitor_targets=True,
                 **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.seed = seed

        assert isinstance(layers, list)
        assert all(isinstance(layer, Layer) for layer in layers)
        assert len(layers) >= 1

        self.layer_name = layer_name

        self.layer_names = set()
        for layer in layers:
            assert layer.get_mlp() is None
            if layer.layer_name in self.layer_names:
                raise ValueError("MLP.__init__ given two or more layers "
                                 "with same name: " + layer.layer_name)

            layer.set_mlp(self)

            self.layer_names.add(layer.layer_name)

        self.layers = layers

        self.batch_size = batch_size
        self.force_batch_size = batch_size

        self._input_source = input_source
        self._target_source = target_source

        self.monitor_targets = monitor_targets

        if input_space is not None or nvis is not None:
            self._nested = False
            self.setup_rng()

            # check if the layer_name is None (the MLP is the outer MLP)
            assert layer_name is None

            if nvis is not None:
                input_space = VectorSpace(nvis)

            # Check whether the input_space and input_source structures match
            try:
                DataSpecsMapping((input_space, input_source))
            except ValueError:
                raise ValueError("The structures of `input_space`, %s, and "
                                 "`input_source`, %s do not match. If you "
                                 "specified a CompositeSpace as an input, "
                                 "be sure to specify the data sources as well."
                                 % (input_space, input_source))

            self.input_space = input_space

            self._update_layer_input_spaces()
        else:
            self._nested = True

        self.freeze_set = set([])

    @wraps(Layer.get_default_cost)
    def get_default_cost(self):

        return Default()

    @wraps(Layer.get_output_space)
    def get_output_space(self):

        return self.layers[-1].get_output_space()

    @wraps(Layer.get_target_space)
    def get_target_space(self):

        return self.layers[-1].get_target_space()

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        if hasattr(self, "mlp"):
            assert self._nested
            self.rng = self.mlp.rng
            self.batch_size = self.mlp.batch_size

        self.input_space = space

        self._update_layer_input_spaces()

    def _update_layer_input_spaces(self):
        """
        Tells each layer what its input space should be.

        Notes
        -----
        This usually resets the layer's parameters!
        """
        layers = self.layers
        try:
            layers[0].set_input_space(self.get_input_space())
        except BadInputSpaceError as e:
            raise TypeError("Layer 0 (" + str(layers[0]) + " of type " +
                            str(type(layers[0])) +
                            ") does not support the MLP's "
                            + "specified input space (" +
                            str(self.get_input_space()) +
                            " of type " + str(type(self.get_input_space())) +
                            "). Original exception: " + str(e))
        for i in xrange(1, len(layers)):
            layers[i].set_input_space(layers[i - 1].get_output_space())

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        # if the MLP is the outer MLP \
        # (ie MLP is not contained in another structure)

        if self.monitor_targets:
            X, Y = data
        else:
            X = data
            Y = None
        rval = self.get_layer_monitoring_channels(state_below=X,
                                                  targets=Y)

        return rval

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):

        rval = OrderedDict()
        state = state_below

        for layer in self.layers:
            # We don't go through all the inner layers recursively
            state_below = state
            state = layer.fprop(state)
            args = [state_below, state]
            if layer is self.layers[-1] and targets is not None:
                args.append(targets)
            ch = layer.get_layer_monitoring_channels(*args)
            if not isinstance(ch, OrderedDict):
                raise TypeError(str((type(ch), layer.layer_name)))
            for key in ch:
                value = ch[key]
                doc = get_monitor_doc(value)
                if doc is None:
                    doc = str(type(layer)) + \
                        ".get_monitoring_channels_from_state did" + \
                        " not provide any further documentation for" + \
                        " this channel."
                doc = 'This channel came from a layer called "' + \
                    layer.layer_name + '" of an MLP.\n' + doc
                value.__doc__ = doc
                rval[layer.layer_name + '_' + key] = value

        return rval

    def get_monitoring_data_specs(self):
        """
        Returns data specs requiring both inputs and targets.

        Returns
        -------
        data_specs: TODO
            The data specifications for both inputs and targets.
        """

        if not self.monitor_targets:
            return (self.get_input_space(), self.get_input_source())
        space = CompositeSpace((self.get_input_space(),
                                self.get_target_space()))
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)

    @wraps(Layer.get_params)
    def get_params(self):

        if not hasattr(self, "input_space"):
            raise AttributeError("Input space has not been provided.")

        rval = []
        for layer in self.layers:
            for param in layer.get_params():
                if param.name is None:
                    logger.info(type(layer))
            layer_params = layer.get_params()
            assert not isinstance(layer_params, set)
            for param in layer_params:
                if param not in rval:
                    rval.append(param)

        rval = [elem for elem in rval if elem not in self.freeze_set]

        assert all([elem.name is not None for elem in rval])

        return rval

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeffs):

        # check the case where coeffs is a scalar
        if not hasattr(coeffs, '__iter__'):
            coeffs = [coeffs] * len(self.layers)

        layer_costs = []
        for layer, coeff in safe_izip(self.layers, coeffs):
            if coeff != 0.:
                layer_costs += [layer.get_weight_decay(coeff)]

        if len(layer_costs) == 0:
            return T.constant(0, dtype=config.floatX)

        total_cost = reduce(operator.add, layer_costs)

        return total_cost

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeffs):

        # check the case where coeffs is a scalar
        if not hasattr(coeffs, '__iter__'):
            coeffs = [coeffs] * len(self.layers)

        layer_costs = []
        for layer, coeff in safe_izip(self.layers, coeffs):
            if coeff != 0.:
                layer_costs += [layer.get_l1_weight_decay(coeff)]

        if len(layer_costs) == 0:
            return T.constant(0, dtype=config.floatX)

        total_cost = reduce(operator.add, layer_costs)

        return total_cost

    @wraps(Model.set_batch_size)
    def set_batch_size(self, batch_size):

        self.batch_size = batch_size
        self.force_batch_size = batch_size

        for layer in self.layers:
            layer.set_batch_size(batch_size)

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):

        for layer in self.layers:
            layer.modify_updates(updates)

    def dropout_fprop(self, state_below, default_input_include_prob=0.5,
                      input_include_probs=None, default_input_scale=2.,
                      input_scales=None, per_example=True):
        """
        Returns the output of the MLP, when applying dropout to the input and
        intermediate layers.


        Parameters
        ----------
        state_below : WRITEME
            The input to the MLP
        default_input_include_prob : WRITEME
        input_include_probs : WRITEME
        default_input_scale : WRITEME
        input_scales : WRITEME
        per_example : bool, optional
            Sample a different mask value for every example in a batch.
            Defaults to `True`. If `False`, sample one mask per mini-batch.


        Notes
        -----
        Each input to each layer is randomly included or
        excluded for each example. The probability of inclusion is independent
        for each input and each example. Each layer uses
        `default_input_include_prob` unless that layer's name appears as a key
        in input_include_probs, in which case the input inclusion probability
        is given by the corresponding value.

        Each feature is also multiplied by a scale factor. The scale factor for
        each layer's input scale is determined by the same scheme as the input
        probabilities.
        """

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self._validate_layer_names(list(input_include_probs.keys()))
        self._validate_layer_names(list(input_scales.keys()))

        theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))

        for layer in self.layers:
            layer_name = layer.layer_name

            if layer_name in input_include_probs:
                include_prob = input_include_probs[layer_name]
            else:
                include_prob = default_input_include_prob

            if layer_name in input_scales:
                scale = input_scales[layer_name]
            else:
                scale = default_input_scale

            state_below = self.apply_dropout(
                state=state_below,
                include_prob=include_prob,
                theano_rng=theano_rng,
                scale=scale,
                mask_value=layer.dropout_input_mask_value,
                input_space=layer.get_input_space(),
                per_example=per_example
            )
            state_below = layer.fprop(state_below)

        return state_below

    def _validate_layer_names(self, layers):
        """
        .. todo::

            WRITEME
        """
        if any(layer not in self.layer_names for layer in layers):
            unknown_names = [layer for layer in layers
                             if layer not in self.layer_names]
            raise ValueError("MLP has no layer(s) named %s" %
                             ", ".join(unknown_names))

    def get_total_input_dimension(self, layers):
        """
        Get the total number of inputs to the layers whose
        names are listed in `layers`. Used for computing the
        total number of dropout masks.

        Parameters
        ----------
        layers : WRITEME

        Returns
        -------
        WRITEME
        """
        self._validate_layer_names(layers)
        total = 0
        for layer in self.layers:
            if layer.layer_name in layers:
                total += layer.get_input_space().get_total_dimension()
        return total

    @wraps(Layer.fprop)
    def fprop(self, state_below, return_all=False):

        if not hasattr(self, "input_space"):
            raise AttributeError("Input space has not been provided.")

        rval = self.layers[0].fprop(state_below)

        rlist = [rval]

        for layer in self.layers[1:]:
            rval = layer.fprop(rval)
            rlist.append(rval)

        if return_all:
            return rlist
        return rval

    def apply_dropout(self, state, include_prob, scale, theano_rng,
                      input_space, mask_value=0, per_example=True):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        state: WRITEME
        include_prob : WRITEME
        scale : WRITEME
        theano_rng : WRITEME
        input_space : WRITEME
        mask_value : WRITEME
        per_example : bool, optional
            Sample a different mask value for every example in a batch.
            Defaults to `True`. If `False`, sample one mask per mini-batch.
        """
        if include_prob in [None, 1.0, 1]:
            return state
        assert scale is not None
        if isinstance(state, tuple):
            return tuple(self.apply_dropout(substate, include_prob,
                                            scale, theano_rng, mask_value)
                         for substate in state)
        # TODO: all of this assumes that if it's not a tuple, it's
        # a dense tensor. It hasn't been tested with sparse types.
        # A method to format the mask (or any other values) as
        # the given symbolic type should be added to the Spaces
        # interface.
        if per_example:
            mask = theano_rng.binomial(p=include_prob, size=state.shape,
                                       dtype=state.dtype)
        else:
            batch = input_space.get_origin_batch(1)
            mask = theano_rng.binomial(p=include_prob, size=batch.shape,
                                       dtype=state.dtype)
            rebroadcast = T.Rebroadcast(*zip(xrange(batch.ndim),
                                             [s == 1 for s in batch.shape]))
            mask = rebroadcast(mask)
        if mask_value == 0:
            rval = state * mask * scale
        else:
            rval = T.switch(mask, state * scale, mask_value)
        return T.cast(rval, state.dtype)

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat, other_mlp_weight=1.):
        cost = None
        for mlp, ds in safe_izip(model, data):
            space, sources = self.get_data_specs(mlp)
            space.validate(ds)
            (X, Y) = ds
            Y_hat = mlp.dropout_fprop(
                X,
                default_input_include_prob=self.default_input_include_prob,
                input_include_probs=self.input_include_probs,
                default_input_scale=self.default_input_scale,
                input_scales=self.input_scales,
                per_example=self.per_example
            )
            c = model.cost(Y, Y_hat)
            if cost is None:
                cost = c
            else:
                cost = cost + self.other_mlp_weight * c
        return cost

        # return self.layers[-1].cost(Y, Y_hat)

    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat):

        return self.layers[-1].cost_matrix(Y, Y_hat)

    @wraps(Layer.cost_from_cost_matrix)
    def cost_from_cost_matrix(self, cost_matrix):

        return self.layers[-1].cost_from_cost_matrix(cost_matrix)

    def cost_from_X(self, data):
        """
        Computes self.cost, but takes data=(X, Y) rather than Y_hat as an
        argument.

        This is just a wrapper around self.cost that computes Y_hat by
        calling Y_hat = self.fprop(X)

        Parameters
        ----------
        data : WRITEME
        """

        for mlp, ds in safe_izip(model.mlps, data): 
            self.cost_from_X_data_specs()[0].validate(data)
            X, Y = data
            Y_hat = self.fprop(X)
            return self.cost(Y, Y_hat)

    def cost_from_X_data_specs(self):
        """
        Returns the data specs needed by cost_from_X.

        This is useful if cost_from_X is used in a MethodCost.
        """
        
        for mlp, ds in safe_izip(model.mlps, data): 
            space = CompositeSpace((self.get_input_space(),
                                    self.get_target_space()))
            source = (self.get_input_source(), self.get_target_source())
            return (space, source)

    def __str__(self):
        """
        Summarizes the MLP by printing the size and format of the input to all
        layers. Feel free to add reasonably concise info as needed.
        """
        rval = []
        for layer in self.layers:
            rval.append(layer.layer_name)
            input_space = layer.get_input_space()
            rval.append('\tInput space: ' + str(input_space))
            rval.append('\tTotal input dimension: ' +
                        str(input_space.get_total_dimension()))
        rval = '\n'.join(rval)
        return rval

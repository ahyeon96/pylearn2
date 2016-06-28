"""
Costs for use with the MLP model class.
"""
__authors__ = 'Vincent Archambault-Bouffard, Ian Goodfellow'
__copyright__ = "Copyright 2013, Universite de Montreal"

from functools import wraps
import operator
import warnings

import theano
from theano import tensor as T
from theano.compat.six.moves import reduce

from pylearn2.costs.costs import Cost
from pylearn2.costs.mlp import Default as SingleDefault
from pylearn2.costs.mlp import WeightDecay as SingleWeightDecay
from pylearn2.costs.mlp import L1WeightDecay as SingleWeightDecay
from pylearn2.costs.mlp.dropout import Dropout as SingleDropout
from pylearn2.utils import safe_izip
from pylearn2.utils.exc import reraise_as


class Dropout(SingleDropout):
    """
    Implements the dropout training technique described in
    "Improving neural networks by preventing co-adaptation of feature
    detectors"
    Geoffrey E. Hinton, Nitish Srivastava, Alex Krizhevsky, Ilya Sutskever,
    Ruslan R. Salakhutdinov
    arXiv 2012

    This paper suggests including each unit with probability p during training,
    then multiplying the outgoing weights by p at the end of training.
    We instead include each unit with probability p and divide its
    state by p during training. Note that this means the initial weights should
    be multiplied by p relative to Hinton's.
    The SGD learning rate on the weights should also be scaled by p^2 (use
    W_lr_scale rather than adjusting the global learning rate, because the
    learning rate on the biases should not be adjusted).

    During training, each input to each layer is randomly included or excluded
    for each example. The probability of inclusion is independent for each
    input and each example. Each layer uses "default_input_include_prob"
    unless that layer's name appears as a key in input_include_probs, in which
    case the input inclusion probability is given by the corresponding value.

    Each feature is also multiplied by a scale factor. The scale factor for
    each layer's input scale is determined by the same scheme as the input
    probabilities.

    Parameters
    ----------
    default_input_include_prob : float
        The probability of including a layer's input, unless that layer appears
        in `input_include_probs`
    input_include_probs : dict
        A dictionary mapping string layer names to float include probability
        values. Overrides `default_input_include_prob` for individual layers.
    default_input_scale : float
        During training, each layer's input is multiplied by this amount to
        compensate for fewer of the input units being present. Can be
        overridden by `input_scales`.
    input_scales : dict
        A dictionary mapping string layer names to float values to scale that
        layer's input by. Overrides `default_input_scale` for individual
        layers.
    per_example : bool
        If True, chooses separate units to drop for each example. If False,
        applies the same dropout mask to the entire minibatch.
    """

    supervised = True

    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
                 default_input_scale=2., input_scales=None, per_example=True,
                 other_mlp_weight=1.):

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    @wraps(Cost.expr)
    def expr(self, model, data, ** kwargs):

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

    @wraps(Cost.cost_per_example)
    def cost_per_example(self, model, data, ** kwargs):
        cost = []
        for mlp, ds in safe_izip(model, data):
            space, sources = self.get_data_specs(mlp)
            space.validate(ds)
            (X, Y) = ds
            Y_hat = model.dropout_fprop(
                X,
                default_input_include_prob=self.default_input_include_prob,
                input_include_probs=self.input_include_probs,
                default_input_scale=self.default_input_scale,
                input_scales=self.input_scales,
                per_example=self.per_example
            )
            c = model.cost_matrix(Y, Y_hat).sum(axis=1)
            if len(cost) == 0:
                cost.append(c)
            else:
                cost.append(self.other_mlp_weight * c)
        return cost

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return True


class Default(SingleDefault):
    """The default Cost to use with an MLP.

    It simply calls the MLP's cost_from_X method.
    """
    supervised = True

    def __init__(self, other_mlp_weight=1.):
        self.other_mlp_weight = other_mlp_weight

    def expr(self, model, data, **kwargs):
        """Returns a theano expression for the cost function.

        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        rval : theano.gof.Variable
            The cost obtained by calling model.cost_from_X(data)
        """
        cost = None
        for mlp, ds in safe_izip(model.mlps, data):
            space, sources = self.get_data_specs(mlp)
            space.validate(ds)
            c = model.cost_from_X(ds)
            if cost is None:
                cost = c
            else:
                cost = cost + self.other_mlp_weight * c
        return cost


    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False


class WeightDecay(SingleWeightDecay):
    """L2 regularization cost for MLP.

    coeff * sum(sqr(weights)) for each set of weights.

    Parameters
    ----------
    coeffs : dict
        Dictionary with layer names as its keys,
        specifying the coefficient to multiply
        with the cost defined by the squared L2 norm of the weights for
        each layer.

        Each element may in turn be a list, e.g., for CompositeLayers.
    """

    def __init__(self, coeffs):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """Returns a theano expression for the cost function.

        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        total_cost : theano.gof.Variable
            coeff * sum(sqr(weights))
            added up for each set of weights.
        """
        assert T.scalar() != 0.  # make sure theano semantics do what I want

        def wrapped_layer_cost(layer, coeff):
            try:
                return layer.get_weight_decay(coeff)
            except NotImplementedError:
                if coeff == 0.:
                    return 0.
                else:
                    reraise_as(NotImplementedError(str(type(layer)) +
                                                   " does not implement "
                                                   "get_weight_decay."))

        layer_costs = []
        for mlp, ds in save_izip(model.mlps, data):
            self.get_data_specs(mlp)[0].validate(ds)

            if isinstance(self.coeffs, list):
                raise ValueError("Coefficients should be given as a dictionary " +
                                 "with layer names as key.")
            else:
                for layer in mlp.layers:
                    layer_name = layer.layer_name
                    if layer_name in self.coeffs:
                        cost = wrapped_layer_cost(layer, self.coeffs[layer_name])
                        if cost != 0.:
                            layer_costs.append(cost)

            if len(layer_costs) == 0:
                rval = T.as_tensor_variable(0.)
                rval.name = '0_weight_decay'
                return rval
            else:
                total_cost = reduce(operator.add, layer_costs)
            total_cost.name = 'MLPs_WeightDecay'

            assert total_cost.ndim == 0

            return total_cost

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False


class L1WeightDecay(L1WeightDecay):
    """L1 regularization cost for MLP.

    coeff * sum(abs(weights)) for each set of weights.

    Parameters
    ----------
    coeffs : dict
        Dictionary with layer names as its keys,
        specifying the coefficient to multiply
        with the cost defined by the squared L2 norm of the weights for
        each layer.

        Each element may in turn be a list, e.g., for CompositeLayers.
    """

    def __init__(self, coeffs):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """Returns a theano expression for the cost function.

        Parameters
        ----------
        model : MLP
        data : tuple
            Should be a valid occupant of
            CompositeSpace(model.get_input_space(),
            model.get_output_space())

        Returns
        -------
        total_cost : theano.gof.Variable
            coeff * sum(abs(weights))
            added up for each set of weights.
        """

        assert T.scalar() != 0.  # make sure theano semantics do what I want
        layer_costs = []
        for mlp, ds in safe_izip(model.mlps, data):
            self.get_data_specs(mlp)[0].validate(ds)
            if isinstance(self.coeffs, list):
                raise ValueError("Coefficients should be given as a dictionary " +
                                 "with layer names as key.")
            else:
                for layer in model.layers:
                    layer_name = layer.layer_name
                    if layer_name in self.coeffs:
                        cost = layer.get_l1_weight_decay(self.coeffs[layer_name])
                        if cost != 0.:
                            layer_costs.append(cost)

        if len(layer_costs) == 0:
            rval = T.constant(0., dtype=theano.config.floatX)
            rval.name = '0_l1_penalty'
            return rval
        else:
            total_cost = reduce(operator.add, layer_costs)
        total_cost.name = 'MLPs_L1Penalty'

        assert total_cost.ndim == 0

        return total_cost

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False

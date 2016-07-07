"""
MultiSubject MLP
"""
__authors__ = "Jesse Livezey, Ahyeon Hwang"


import logging
import operator

from theano.compat.six.moves import reduce, xrange
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from pylearn2.compat import OrderedDict
from pylearn2.costs.mlp import Default
from pylearn2.models.mlp import Layer, MLP, Softmax
from pylearn2.models import Model
from pylearn2.space import CompositeSpace
from pylearn2.space import VectorSpace, IndexSpace
from pylearn2.utils import safe_izip
from pylearn2.utils import wraps
from pylearn2.utils.data_specs import DataSpecsMapping


logger = logging.getLogger(__name__)


class MultiSubjectMLP(Layer):

    """
    A multilayer perceptron.

    Note that it's possible for an entire MLP to be a single layer of a larger
    MLP.

    Parameters
    ----------
    mlps : list
        A list of MLP objects. The first MLP is the target subject.
    num_shared_layers : int
        Number of final layers the MLPs should share parameters across.
    input_space : Space object
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
    share_biases : bool
        Whether to share biases across shared layers.
    monitor_targets : bool, optional
        Default: True
        If true, includes monitoring channels that are functions of the
        targets. This can be disabled to allow monitoring on monitoring
        datasets that do not include targets.
    kwargs : dict
        Passed on to the superclass.
    """

    def __init__(self, mlps, num_shared_layers,
                 input_space, input_source=None,
                 target_source=None, share_biases=False,
                 other_mlp_weight=1., monitor_targets=True,
                 **kwargs):
        super(MultiSubjectMLP, self).__init__(**kwargs)

        self.seed = seed

        assert isinstance(mlps, list)
        assert all(isinstance(mlp, MLP) for mlp in mlps)
        assert len(mlps) >= 1

        self.num_shared_layers = num_shared_layers
        self.share_biases = share_biases
        self.mlp_names = set()
        for mlp in mlps:
            if mlp.layer_name in self.mlp_names:
                raise ValueError("MultisubjectMLP.__init__ given two or more mlps "
                                 "with same name: " + str(mlp.layer_name))
            self.mlp_names.add(mlp.layer_name)

        self.mlps = mlps

        if input_source is None:
            input_source = len(mlps) * ('features',)
        if target_source is None:
            target_source = len(mlps) * ('targets',)

        self._input_source = input_source
        self._target_source = target_source

        self.monitor_targets = monitor_targets

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
        self.set_input_space(input_space)

        for mlp, sp in safe_izip(mlps, input_space):
            assert sp == mlp.get_input_space()

        self.freeze_set = set([])

    def _share_parameters(self):
        main_mlp = self.mlps[0]
        for mlp in self.mlps[1:]:
            for layer_n in xrange(-1, -self.num_shared_layers-1, -1):
                main_layer = main_mlp.layers[layer_n]
                layer = mlp.layers[layer_n]
                if isinstance(main_layer, Softmax):
                    layer.W = main_layer.W
                else:
                    params = main_layer.transformer.get_params()
                    layer.transformer.set_params(params)
                if self.share_biases:
                    mlp.layers[layer_n].b = main_mlp.layers[layer_n].b

    @wraps(Layer.get_default_cost)
    def get_default_cost(self):

        return Default()

    @wraps(Layer.get_output_space)
    def get_output_space(self):
        spaces = []
        for mlp in self.mlps:
            spaces.append(mlp.layers[-1].get_output_space())
        return CompositeSpace(tuple(spaces))

    @wraps(Layer.get_target_space)
    def get_target_space(self):
        spaces = []
        for mlp in self.mlps:
            spaces.append(mlp.layers[-1].get_target_space())
        return CompositeSpace(tuple(spaces))

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space
        for mlp, sp in safe_izip(self.mlps, space):
            mlp.set_input_space(sp)

        self._share_parameters()

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        rvals = OrderedDict()
        for mlp, ds in safe_izip(self.mlps, data):
            if self.monitor_targets:
                X, Y = data
            else:
                X = data
                Y = None
            rval = mlp.get_monitoring_channels(state_below=X,
                                                      targets=Y)
            for key in rval.keys():
                rvals[key] = rval[key]

        return rvals

    @wraps(Layer.get_params)
    def get_params(self):

        if not hasattr(self, "input_space"):
            raise AttributeError("Input space has not been provided.")

        rval = []
        for mlp in self.mlps:
            rval.extend(mlp.get_params())

        rval = set(rval)

        rval = [elem for elem in rval if elem not in self.freeze_set]

        assert all([elem.name is not None for elem in rval])

        return rval

    @wraps(Layer.get_weight_decay)
    def get_weight_decay(self, coeffs):

        mlp_costs = []
        for mlp in self.mlps:
            mlp_costs.append(mlp.get_weight_decay())

        total_cost = reduce(operator.add, mlp_costs)

        return total_cost

    @wraps(Layer.get_l1_weight_decay)
    def get_l1_weight_decay(self, coeffs):

        mlp_costs = []
        for mlp in self.mlps:
            mlp_costs.append(mlp.get_l1_weight_decay())

        total_cost = reduce(operator.add, mlp_costs)

        return total_cost

    @wraps(Layer._modify_updates)
    def _modify_updates(self, updates):

        for mlp in self.mlps:
            mlp._modify_updates(updates)

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

        Y_hat_list = []    
        for mlp, state_belowi in safe_izip(self.mlps, state_below):

            Y_hat_list.append(mlp.dropout_fprop(default_input_include_prob=0.5,
                      input_include_probs=None, default_input_scale=2.,
                      input_scales=None, per_example=True))
        return Y_hat_list

    def _validate_layer_names(self, layers):
        """
        .. todo::

            WRITEME
        """
        for layer in layers:
            in_mlps = [layer in mlp.layer_names for mlp in self.mlps]
            assert sum(in_mlps) == 1, layer + ' not found in any mlp'

    @wraps(Layer.fprop)
    def fprop(self, state_below, return_all=False):

        if not hasattr(self, "input_space"):
            raise AttributeError("Input space has not been provided.")

        rval_list = []
        for mlp, state_belowi in safe_izip(self.mlps, state_below):
            rval_list.append(mlp.fprop(state_belowi, return_all))
        return rval_list

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):
        cost = None
        for mlp, Yi, Y_hati in safe_izip(self.mlps, Y, Y_hat):
            c = mlp.cost(Yi, Y_hati)
            if cost is None:
                cost = c
            else:
                cost = cost + self.other_mlp_weight * c
        return cost

    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat):
        cost = []
        for mlp, Yi, Y_hati in safe_izip(self.mlps, Y, Y_hat):
            c = mlp.cost_matrix(Yi, Y_hati)
            if len(cost) == 0:
                cost.append(c)
            else:
                cost.append(self.other_mlp_weight * c)
        return cost

    @wraps(Layer.cost_from_cost_matrix)
    def cost_from_cost_matrix(self, cost_matrix):
        cost = None
        for mlp, cost_matrixi in safe_izip(self.mlps, cost_matrix):
            c = mlp.cost_from_cost_matrix(cost_matrixi)
            if cost is None:
                cost = c
            else:
                cost = cost + self.other_mlp_weight * c
        return cost

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

        cost = None
        for mlp, ds in safe_izip(model.mlps, data): 
            self.cost_from_X_data_specs()[0].validate(data)
            X, Y = data
            Y_hat = mlp.fprop(X)

            c = mlp.cost(Y, Y_hat)
            if cost is None:
                cost = c
            else:
                cost = cost + self.other_mlp_weight * c
        return cost

    def cost_from_X_data_specs(self):
        """
        Returns the data specs needed by cost_from_X.

        This is useful if cost_from_X is used in a MethodCost.
        """
        spaces = []
        sources = []
        for mlp in self.mlps: 
            spaces.append(CompositeSpace((self.get_input_space(),
                                          self.get_target_space())))
            sources.append((self.get_input_source(), self.get_target_source()))
        return (CompositeSpace(tuple(spaces)), tuple(sources))

    def __str__(self):
        """
        Summarizes the MLP by printing the size and format of the input to all
        layers. Feel free to add reasonably concise info as needed.
        """
        rval = []
        for mlp in self.mlps:
            rval.append(mlp.name)
            """
            input_space = layer.get_input_space()
            rval.append('\tInput space: ' + str(input_space))
            rval.append('\tTotal input dimension: ' +
                        str(input_space.get_total_dimension()))
            """
        rval = '\n'.join(rval)
        return rval

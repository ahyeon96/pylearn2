from pylearn2.models import mlp
from pylearn2.linear import synaptic

class Synaptic(mlp.Linear):
    def __init__(self,
                 dim,
                 layer_name,
                 nonlinearity,
                 irange=None,
                 istdev=None,
                 max_col_norm=None):
        super(Synaptic, self).__init__(dim,
                                       layer_name,
                                       irange=irange,
                                       istdev=istdev,
                                       max_col_norm=max_col_norm)
        self.nonlinearity = nonlinearity

    def set_input_space(self, space):
        super(Synaptic, self).set_input_space(space)
        W, = self.transformer.get_params()
        self.transformer = synaptic.Synaptic(W, self.nonlinearity)


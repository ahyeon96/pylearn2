from pylearn2.models import mlp
from pylearn2.utils import wraps


class GlobalAffine(mlp.Layer):
    """
    A layer that applied the same affine transformation
    to every element of the state.
    
    Parameters
    ----------
    m : float
        Value to multiply state by.
    b : float
        Value to add to scaled state.
    """

    def __init__(self, m=1., b=0.):
        super(GlobalAffine, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self._params = []

    @wraps(mlp.Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space
        self.output_space = space

    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):

        return self.m*state_below+self.b

class Identity(mlp.Layer):
    """
    A layer that passes the state through unchanged.
    """
    def __init__(self):
        super(Identity, self).__init__()
        self.__dict__.update(locals())
        del self.self
        self._params = []

    @wraps(mlp.Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space
        self.output_space = space

    @wraps(mlp.Layer.fprop)
    def fprop(self, state_below):

        return state_below

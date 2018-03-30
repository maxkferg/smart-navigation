import gym.spaces
import numpy as np

def prepend_dimension(shape, dimension):
	"""Insert the batch size as the first dimesion of a shape tuple"""
	shape = list(shape)
	shape.insert(0, dimension)
	return tuple(shape)


class Space(gym.spaces.Box):
    """A space with limits"""
    def __init__(self, low, high, shape=None, **kwargs):
        super().__init__(low, high, shape, **kwargs)
        self.n = np.prod(shape)


class ObservationSpace(Space):
    pass


class ActionSpace(Space):
    pass

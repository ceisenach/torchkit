import numpy as np
import gym

class NanActionError(Exception):
    """Should be raised when environment receives NaN as action"""

class NumpyEnv(gym.Env):

    def step(self, action):
        action = np.array(action,dtype=np.float32)
        if np.isnan(action).any():
            raise NanActionError('Received Action:%s' % str(action))
        return self._step(action)


################
# Import Names
from .nav2d import Platform2D

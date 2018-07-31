import gym
import torch
import logging
logger = logging.getLogger(__name__)

from utils import MultiRingBuffer
from running_stat import ZFilter


def _torch_from_float(val):
    tval = torch.FloatTensor(1)
    tval[0] = float(val)
    return tval


class Sampler(object):
    """
    Samples batches using policy for an environment specified by kwargs
    """ 
    def __init__(self,policy,**kwargs):
        env = gym.make(kwargs['env'])
        env.seed(kwargs['seed'])
        exp_shapes = [env.observation_space.shape,env.action_space.shape,(1,),(1,)]
        self._base_env = env
        self._policy = policy
        self._N = kwargs['N']
        self._gamma = kwargs['gamma']
        # self._experience_buffer = MultiRingBuffer(exp_shapes,self._N+1,tensor_type=torch.DoubleTensor)
        self._experience_buffer = MultiRingBuffer(exp_shapes,self._N+1,tensor_type=torch.FloatTensor)
        self._running_state = ZFilter((env.observation_space.shape[0],), clip=5)
        self._running_reward = ZFilter((1,), demean=False, clip=10)
        self._cr = 0.0
        self._terminal = False
        self._t = 0

    def sample(self):
        """
        Get a batch of samples
        """
        self._experience_buffer.reset()
        crs = []
        s_t_numpy = self._base_env.state
        num_steps = 0

        while num_steps < self._N:
            if self._terminal:
                s_t_numpy = self._base_env.reset()
                crs.append(self._cr)
                self._cr,self._t = 0,0
                self._terminal = False

            # take step
            # s_t_numpy = self._running_state(s_t_numpy)
            s_t = torch.from_numpy(s_t_numpy)
            a_t = self._policy.action(s_t)
            a_t_numpy = a_t.numpy()
            s_tp1_numpy, r_tp1_f, self._terminal, _ = self._base_env.step(a_t_numpy)
            self._cr += (self._gamma**self._t)*r_tp1_f

            # terminal state mask
            m_t_f = 0 if self._terminal else 1

            # append to buffer
            m_t = _torch_from_float(m_t_f)
            r_tp1 = _torch_from_float(r_tp1_f)
            self._experience_buffer.append(s_t,a_t,m_t,r_tp1)

            # increment
            s_t_numpy = s_tp1_numpy
            self._t += 1
            num_steps += 1
            
        batch = self._experience_buffer.get_data()
        return batch,crs

    def reset(self):
        self._base_env.reset()
        self._cr,self._t = 0,0
        self._terminal = False


class BatchSampler(object):
    """
    Sample from multiple independent copies of the same environment.
    """
    def __init__(self,policy,**kwargs):
        self._samplers = []
        for i in range(kwargs['num_env']):
            smp = Sampler(policy,**kwargs)
            self._samplers.append(smp)

    def sample(self):
        """
        Get multiple minibatches of samples
        """
        bl,crs = [],[]
        for smp in self._samplers:
            b,c = smp.sample()
            bl.append(b)
            crs = crs + c
        return bl,crs

    def reset(self):
        """
        Resets underlying environment objects
        """
        for smp in self._samplers:
            smp.reset()

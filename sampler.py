import gym
import torch
import logging
import random
import copy
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
        self._tr = 0.0
        self._terminal = False
        self._t = 0
        self._s_t_numpy = None
        self._debug = kwargs['debug']

    def sample(self):
        """
        Get a batch of samples
        """
        self._experience_buffer.reset()
        crs,trs,els = [],[],[]
        s_t_numpy = self._s_t_numpy
        num_steps = 0

        while num_steps < self._N:
            if self._terminal:
                s_t_numpy = self._base_env.reset()
                crs.append(self._cr)
                trs.append(self._tr)
                els.append(self._t)
                self._tr,self._cr,self._t = 0,0,0
                self._terminal = False

            # take step
            s_t_numpy = self._running_state(s_t_numpy)
            s_t = torch.from_numpy(s_t_numpy).float()
            a_t = self._policy.action(s_t)
            a_t_numpy = a_t.numpy()
            try:
                s_tp1_numpy, r_tp1_f, self._terminal, _ = self._base_env.step(a_t_numpy)
            except Exception as e:
                if self._debug:
                    import pdb; pdb.set_trace()
                else:
                    raise e

            self._cr += (self._gamma**self._t)*r_tp1_f
            self._tr += r_tp1_f

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
            
        self._s_t_numpy = s_t_numpy
        batch = self._experience_buffer.get_data()
        return batch,crs,trs,els

    def reset(self):
        self._s_t_numpy = self._base_env.reset()
        self._cr,self._tr,self._t = 0,0,0
        self._terminal = False


class BatchSampler(object):
    """
    Sample from multiple independent copies of the same environment.
    """
    def __init__(self,policy,**kwargs):
        self._samplers = []
        for i in range(kwargs['num_env']):
            sk = copy.deepcopy(kwargs)
            sk['seed'] = random.randint(1,1e8) # give different seed to each sampler
            smp = Sampler(policy,**sk)
            self._samplers.append(smp)

    def sample(self):
        """
        Get multiple minibatches of samples
        """
        bl,crs,trs,els = [],[],[],[]
        for smp in self._samplers:
            b,c,t,e = smp.sample()
            bl.append(b)
            crs,trs,els = crs + c, trs + t, els + e
        return bl,crs,trs,els

    def reset(self):
        """
        Resets underlying environment objects
        """
        for smp in self._samplers:
            smp.reset()

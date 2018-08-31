# IMPORTS
import numpy as np
import random
from collections import namedtuple
import gym
from gym import spaces
from gym.envs.registration import register
import logging
logger = logging.getLogger(__name__)

__all__ = ['OptimalProduction']


## Setup
Material = namedtuple('Material', ['name', 'price', 'store_cost'])
RecipeItem = namedtuple('RecipeItem', ['material', 'quantity'])
Product = namedtuple('Product', ['name', 'recipe', 'price', 'store_cost'])

def register_envs():
    # Simple Production Problem
    pl = []
    for pn,pp in [('I',1.75),('II',2.0),('III',2.25),('IV',2.5)]:
    # for pn,pp in [('I',3.0),('II',3.25),('III',3.5),('IV',3.75)]:
        pr = [RecipeItem(Material('%s.A'%pn,0.5,0.1),1.0),
                RecipeItem(Material('%s.B'%pn,0.5,0.1),1.0),
                RecipeItem(Material('%s.C'%pn,0.5,0.1),1.0)]
        pl.append(Product(pn,pr,pp,0.1))
    gamma_sampler = lambda : np.random.gamma(np.array([9.0]*4,dtype = np.float32),np.array([0.5]*4,dtype = np.float32))
    register(
        id='OptimalProduction1-v0',
        entry_point='environment:OptimalProduction',
        kwargs={'product_list':pl,'demand_distribution':gamma_sampler,'name':'1','T':1000}
    )



## Environments
def _process_product_list(product_list):
    num_goods = len(product_list)
    num_materials_per_good = [len(g.recipe) for g in product_list]
    num_materials = sum(num_materials_per_good)
    state_names = [g.name for g in product_list] + [ri.material.name for g in product_list for ri in g.recipe]
    item_prices = np.array([g.price for g in product_list] + [ri.material.price for g in product_list for ri in g.recipe],dtype = np.float32)
    store_costs = np.array([g.store_cost for g in product_list] + [ri.material.store_cost for g in product_list for ri in g.recipe],dtype = np.float32)
    material_quantities = np.array([ri.quantity for g in product_list for ri in g.recipe],dtype = np.float32)
    product_idxs,last_idx = [],0
    for i in range(num_goods):
        product_idxs.append((last_idx,last_idx + num_materials_per_good[i]))
        last_idx = last_idx + num_materials_per_good[i]
    return num_goods,num_materials,state_names,item_prices,store_costs,product_idxs,material_quantities

def _max_production(mstate,prod_idx,mat_q):
    amts = mstate / mat_q
    return np.array([min(amts[si:ei]) for i,(si,ei) in enumerate(prod_idx)],dtype=np.float32)


class OptimalProduction(gym.Env):
    """
    Optimal Production Task
    Ordering of State Representation is goods then ingredients, in the order they appear.
    """
    def __init__(self,product_list,demand_distribution,name='',T=1000):
        self.__version__ = "0.1.0"
        print("OptimalProduction{} - Version {}".format(name,self.__version__))
        self._product_list = product_list
        self._num_goods,self._num_materials,self._state_names,self._item_prices,self._store_costs, \
        self._product_idxs,self._material_quantities = _process_product_list(product_list)

        # Action Space, State Space
        low = np.array([0.0]*(self._num_goods+self._num_materials),dtype = np.float32)
        high = np.array([np.finfo(np.float32).max]*(self._num_goods+self._num_materials),dtype = np.float32)
        self.action_space = spaces.Box(low,high,dtype=np.float32)
        self.observation_space = spaces.Box(low,high,dtype=np.float32)

        # environment config
        self._state = None
        self._time = 0
        self._T = T
        self._demand_distribution = demand_distribution

        # initial state config
        self.reset()

    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        u_t = np.array(action,dtype=np.float32) # make sure it is array
        assert u_t.shape == self.action_space.shape, "Action Shape: %s. Should be: %s." % (str(u_t.shape),str(self.action_space.shape))
        u_t = np.maximum(u_t,0.0).astype(np.float32) # threshold
        if self._time >= self._T:
            raise RuntimeError("Episode is done")

        # update state
        N = self._num_goods
        s_t = self.state
        l_t = _max_production(s_t[N:],self._product_idxs,self._material_quantities) # max production
        u_t[:N] = np.minimum(u_t[:N],l_t) # threshold action
        q_t = np.array([u_t[i]*self._material_quantities[j] for i,(si,ei) in enumerate(self._product_idxs) \
                         for j in range(si,ei)],dtype=np.float32) # number of each material used
        d_t = self._demand_distribution() # realized demand
        n_t = np.minimum(s_t[:N]+u_t[:N],d_t) # number sold
        if self._time % 50 == 0:
            logger.debug(('Time %d -- State: ' + ','.join(['%.2f']*len(s_t))) % (self._time,*tuple(s_t)))
            logger.debug(('Time %d -- Action: ' + ','.join(['%.2f']*len(u_t))) % (self._time,*tuple(u_t)))

        s_tp1 = s_t
        s_tp1[:N] = np.maximum(0.0,s_t[:N]+u_t[:N] - d_t)
        s_tp1[N:] = s_t[N:] + u_t[N:] - q_t

        # calculate reward
        r_tp1 = sum(self._item_prices[:N]*n_t) - sum(self._item_prices[N:]*u_t[N:]) - sum(self._store_costs*s_tp1)

        # return (S,R,T,I)
        self._time += 1
        self._state = s_tp1
        terminal = self._time == self._T

        return s_tp1, r_tp1, terminal, {}

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.
        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self._time = 0
        self._state = np.array([0.0]*(self._num_goods+self._num_materials),dtype = np.float32)
        return self._state

    @property
    def state(self):
        return self._state

    @property
    def time(self):
        return self._time

    def _render(self, mode='human', close=False):
        return

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)


register_envs()

if __name__ =='__main__':
    pass
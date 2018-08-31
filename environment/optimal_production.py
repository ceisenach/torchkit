# basic navigation task

# IMPORTS
import numpy as np
import random
from collections import namedtuple
import gym
from gym import spaces
from gym.envs.registration import register
import logging
logger = logging.getLogger(__name__)

register(
    id='OptimalProduction1-v1',
    entry_point='environment:Platform2D',
)

Material = namedtuple('Material', ['name', 'price', 'store_cost'])
RecipeItem = namedtuple('RecipeItem', ['material', 'quantity'])
Product = namedtuple('Product', ['name', 'recipe', 'price', 'store_cost'])

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
        self._product_idxs,self._material_quantitites = _process_product_list(product_list)

        # Action Space, State Space
        low = np.array([0.0]*(self._num_goods+self._num_materials),dtype = np.float32)
        high = np.array([np.finfo(np.float32).max]*(self._num_goods+self._num_materials),dtype = np.float32)
        self.action_space = spaces.Box(low,high,dtype=np.float32)
        self.observation_space = spaces.Box(low,high,dtype=np.float32)

        # environment config
        self._state = None
        self._time = 0
        self._T = T

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
        action = np.array(action,dtype=np.float32) # make sure it is array
        assert action.shape == self.action_space.shape, "Action Shape: %s. Should be: %s." % (str(action.shape),str(self.action_space.shape))
        action = np.maximum(action,0.0).astype(np.float32) # threshold
        if self._time >= self._T:
            raise RuntimeError("Episode is done")

        # update state
        s_t = self.state
        l_t = _max_production()



        self._update_state(action)

        # return (S,R,T,I)
        self._time += 1
        s_tp1 = self.state
        r_tp1 = self._reward(s_t,s_tp1)
        terminal = self._time == self._T

        return s_tp1, r_tp1, terminal, {}

    def _reward(self,old_state,new_state):
        pass

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


if __name__ =='__main__':
    pl = []
    for pn,pp in [('I',1.75),('II',2.0),('III',2.25),('IV',2.5)]:
        pr = [RecipeItem(Material('%s.A'%pn,0.5,0.25),1.0),
                RecipeItem(Material('%s.B'%pn,0.5,0.25),1.0),
                RecipeItem(Material('%s.C'%pn,0.5,0.25),1.0)]
        pl.append(Product(pn,pr,pp,0.25))
    # print(pl)

    _,_,sn,ip,sc,pi,mq =  _process_product_list(pl)



    inv = np.array([0.0]*12,dtype = np.float32)
    inv = np.array([1.,3.,4.,1.,0.5,1.,2.,3.,4.,2.,3.,4.],dtype=np.float32)

    print(_max_production(inv,pi,mq))


    env = OptimalProduction(pl,None)
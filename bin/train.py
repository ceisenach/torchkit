# Train model on Platform 2D environment
import sys
sys.path.append('./') # allows it to be run from parent dir
import os
import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)
import gym
import json

import sampler
import algorithm
import utils
import model
import environment
import policy

if __name__ == '__main__':
    #############################
    # SETUP
    parser = utils.experiment_argparser()
    args = parser.parse_args()
    train_config = utils.train_params_from_args(args)
    episode_results_path = os.path.join(train_config['odir'],'episode_results.npy')

    utils.make_directory(train_config['odir'])
    with open(os.path.join(train_config['odir'],'train_config.json'), 'w') as fp:
        json.dump(train_config, fp, sort_keys=True, indent=4)

    log_level = logging.DEBUG if train_config['debug'] else logging.INFO
    if not train_config['console']:     
        logging.basicConfig(filename=os.path.join(train_config['odir'],'log_main.log'),
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',)
    else:
        logging.basicConfig(level=log_level)

    ###############################
    # MAKE NET AND POLICY
    env = gym.make(train_config['env'])
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    torch.manual_seed(train_config['seed'])

    actor_net = model.Policy(num_inputs, num_actions,**train_config['ac_kwargs'])
    critic_net = model.Value(num_inputs,**train_config['ac_kwargs'])
    plc = policy.GaussianPolicy(actor_net)

    ###############################
    # CREATE ENVIRONMENT AND RUN
    if train_config['alg'] == 'TRPO':
        algo = algorithm.TRPO(plc,critic_net,train_config)
    elif train_config['alg'] == 'TRPO_v2':
        algo = algorithm.TRPO_v2(plc,critic_net,train_config)
    elif train_config['alg'] == 'NAC':
        algo = algorithm.NACGauss(plc,critic_net,train_config)
    elif train_config['alg'] == 'A2C':
        algo = algorithm.A2C(plc,critic_net,train_config)
    else:
        raise RuntimeError('Algorithm not found')

    sampler = sampler.BatchSampler(plc,**train_config)
    episode_results = np.array([]).reshape((0,5))
    cur_update = 0
    finished_episodes = 0
    sampler.reset()

    while cur_update < train_config['num_updates']:
        batch,crs,trs,els = sampler.sample()
        algo.update(batch)

        # save episode results
        for i,(cr,tr,el) in enumerate(zip(crs,trs,els)):
            finished_episodes += 1
            episode_results = np.concatenate((episode_results,np.array([cur_update,finished_episodes,el,tr,cr],ndmin=2)),axis=0)
            np.save(episode_results_path, episode_results)
            logger.info('Update Number: %06d, Finished Episode: %04d ---  Length: %.3f, TR: %.3f, CDR: %.3f'% (cur_update,finished_episodes,el,tr,cr))

        # checkpoint
        if cur_update % train_config['save_interval'] == 0:
            plc.save_model(os.path.join(train_config['odir'],'model_update_%06d.pt'%(cur_update)))
        cur_update += 1
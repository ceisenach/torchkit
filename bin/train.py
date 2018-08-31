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
import random
import time

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
    episode_results_path = os.path.join(train_config['odir'],'episode_results_run_%d.npy' % train_config['run'])

    utils.make_directory(train_config['odir'])
    with open(os.path.join(train_config['odir'],'train_config_run_%d.json' % train_config['run']), 'w') as fp:
        json.dump(train_config, fp, sort_keys=True, indent=4)

    log_level = logging.DEBUG if train_config['debug'] else logging.INFO
    if not train_config['console']:     
        logging.basicConfig(filename=os.path.join(train_config['odir'],'log_run_%d.log' % train_config['run']),
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',)
    else:
        logging.basicConfig(level=log_level)

    ###############################
    # MAKE NET AND POLICY
    env = gym.make(train_config['env'])
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    if train_config['seed'] is not None:
        random.seed(train_config['seed'])
        ts = random.randint(1,1e8)
        torch.manual_seed(ts)

    log_str = '\r\n###################################################\r\n' + \
              '\tEnvironment: %s\r\n' % train_config['env'] + \
              '\tAlgorithm: %s\r\n' % train_config['alg'] + \
              '\tPolicy Class: %s\r\n' % train_config['policy'] + \
              '\tNetwork Types: (%s,%s)\r\n' % tuple(train_config['ac_types']) + \
              '\tNetwork Params: %s \r\n' % str(train_config['ac_kwargs']) + \
              '\tN, Total Updates, Save Interval: (%d,%d,%d) \r\n' % (train_config['N'],train_config['num_updates'],train_config['save_interval']) + \
              '###################################################'
    logger.info(log_str)

    actor_net = getattr(model,train_config['ac_types'][0])(num_inputs, num_actions,**train_config['ac_kwargs'])
    critic_net = getattr(model,train_config['ac_types'][1])(num_inputs,**train_config['ac_kwargs'])
    plc = None
    try:
        plc_class = getattr(policy,train_config['policy'])
        plc = plc_class(actor_net)
    except AttributeError as e:
        raise RuntimeError('Algorithm "%s" not found' % train_config['policy'])

    ###############################
    # CREATE ENVIRONMENT AND RUN
    algo = None
    try:
        algo_class = getattr(algorithm,train_config['alg'])
        algo = algo_class(plc,critic_net,train_config)
    except AttributeError as e:
        raise RuntimeError('Algorithm "%s" not found' % train_config['alg'])

    sampler = sampler.BatchSampler(plc,**train_config)
    episode_results = np.array([]).reshape((0,6))
    cur_update = 0
    finished_episodes = 0
    sampler.reset()
    samples_per_update = train_config['N'] * train_config['num_env']
    start = time.time()
    while cur_update < train_config['num_updates']:
        batch,crs,trs,els = sampler.sample()
        algo.update(batch)

        # save episode results
        for i,(cr,tr,el) in enumerate(zip(crs,trs,els)):
            finished_episodes += 1
            total_samples = cur_update * samples_per_update
            # stores: total_updates, total_episodes, total_samples, current_episode_length, current_total_reward, current_cumulative_reward
            episode_results = np.concatenate((episode_results,np.array([cur_update,finished_episodes,total_samples,el,tr,cr],ndmin=2)),axis=0)
            np.save(episode_results_path, episode_results)
            logger.info('Update Number: %06d, Finished Episode: %04d ---  Length: %.3f, TR: %.3f, CDR: %.3f'% (cur_update,finished_episodes,el,tr,cr))

        # checkpoint
        if cur_update % train_config['save_interval'] == 0:
            plc.save_model(os.path.join(train_config['odir'],'model_update_%06d_run_%d.pt' % (cur_update,train_config['run'])))
        cur_update += 1

    end = time.time()
    print(end - start)
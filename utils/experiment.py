import numpy as np
import torch
import os
import argparse
import time
import math
import copy
import ast
import random


def make_directory(dirpath):
    os.makedirs(dirpath,exist_ok=True)

def experiment_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lra', default=1e-3, type=float, help='learning rate actor')
    parser.add_argument('--lrc', default=1e-3, type=float, help='learning rate critic')
    parser.add_argument("-o", "--odir", type=str, default=None, help="output directory")
    parser.add_argument("-d", "--debug", action="store_true", help="debug")
    parser.add_argument("-p",'--policy' ,type=str, default='GaussianML', help="policy type to use")
    parser.add_argument("-u",'--num_updates', type=float, default=1e4, help="number of gradient updates")
    parser.add_argument("-g","--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("-N", type=int, default=15000, help="Batch Size")
    parser.add_argument("-V", '--num-env', type=int, default=1, help="Num Env")
    parser.add_argument("-s",'--save_interval', type=float, default=1e3, help="Model Save Interval")
    parser.add_argument("-l",'--log_interval', type=int, default=50, help="Log Interval")
    parser.add_argument("-E",'--env', type=str, default="BipedalWalker-v2", help="Environment to use")
    parser.add_argument("-co", "--console", action="store_true", help="log to console")
    parser.add_argument('--tau', type=float, default=0.98, metavar='G',help='gae (default: 0.97)')
    parser.add_argument('--l2_pen', type=float, default=1e-3, metavar='G',help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--max_kl', type=float, default=1e-2, metavar='G',help='max kl value (default: 1e-2)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',help='random seed (default: 543). -1 indicates no seed')
    parser.add_argument('--damping', type=float, default=1e-1, metavar='G',help='damping (default: 1e-1)')
    parser.add_argument('-a','--alg',type=str, default='TRPO', metavar='G',help='algorithm to use')
    parser.add_argument('-b','--backend',type=str, default='numpy', metavar='G',help='backend to use for Jacobian')
    parser.add_argument("--nk",  type=str,default=None, help="kwargs for actor and critic nets")
    parser.add_argument('--nt',type=str,nargs='+',default=['GaussianML','Value'], help="which network architecture to use")
    parser.add_argument("--run",  type=int,default=0, help="indicates what run number if running multiple of same experiment")
    parser.add_argument("--rstat", action="store_true", help="use running stats to normalize states (used in Schulman's TRPO)")

    return parser

def train_params_from_args(args):
    experiment_config = {'gamma' : args.gamma,
                         'alg' : args.alg,
                         'tau' : args.tau,
                         'max_kl' : args.max_kl,
                         'l2_pen' : args.l2_pen,
                         'lr_actor' : args.lra,
                         'lr_critic' : args.lrc,
                         'damping' : args.damping,
                         'backend' : args.backend,
                         'num_env' : args.num_env,
                         'ac_kwargs': get_kwargs(args.nk),
                         'ac_types': {'actor':args.nt[0],'critic':args.nt[1]},
                         'debug' : args.debug,
                         'policy' : args.policy,
                         'running_stat' : args.rstat,
                         'seed' : args.seed if args.seed != -1 else random.randint(0,1e8), # make a random seed
                         'num_updates' : int(args.num_updates),
                         'N' : args.N,
                         'env' : args.env,
                         'console' : args.console,
                         'run' : args.run,
                         'odir' : args.odir if args.odir is not None else 'out/experiment_%s' % time.strftime("%Y.%m.%d_%H.%M.%S"),
                         'save_interval' : int(args.save_interval)}

    return experiment_config


def run_config_from_args(args):
    experiment_config = {'backend' : args.backend,
                         'debug' : args.debug,
                         'console' : args.console,
                         'odir' : args.odir if args.odir is not None else 'out/experiment_%s' % time.strftime("%Y.%m.%d_%H.%M.%S"),
                         'save_interval' : int(args.save_interval)}

    return experiment_config



def get_kwargs(arg_str):
    if arg_str is not None:
        kwargs = ast.literal_eval(arg_str)
        return kwargs
    return {}


def run_training(train_config):
    import os
    import logging
    logger = logging.getLogger(__name__)
    import gym
    import json
    import time

    import sampler
    import algorithm
    import model
    import environment
    import policy
    #############################
    # SETUP
    episode_results_path = os.path.join(train_config['odir'],'episode_results_run_%d.npy' % train_config['run'])

    make_directory(train_config['odir'])
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
              '\tNetwork Types: (%s,%s)\r\n' % (train_config['ac_types']['actor'],train_config['ac_types']['critic']) + \
              '\tNetwork Params: %s \r\n' % str(train_config['ac_kwargs']) + \
              '\tN, Total Updates, Save Interval: (%d,%d,%d) \r\n' % (train_config['N'],train_config['num_updates'],train_config['save_interval']) + \
              '###################################################'
    logger.info(log_str)

    actor_net = getattr(model,train_config['ac_types']['actor'])(num_inputs, num_actions,**train_config['ac_kwargs'])
    critic_net = getattr(model,train_config['ac_types']['critic'])(num_inputs,**train_config['ac_kwargs'])
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

    smp = sampler.BatchSampler(plc,**train_config)
    episode_results = np.array([]).reshape((0,6))
    cur_update = 0
    finished_episodes = 0
    smp.reset()
    samples_per_update = train_config['N'] * train_config['num_env']
    start = time.time()
    while cur_update < train_config['num_updates']:
        batch,crs,trs,els = smp.sample()
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

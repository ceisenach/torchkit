import torch
from torch.multiprocessing import Pool
import json
import csv
import argparse
import os
import copy
import sys
sys.path.append('./') # allows it to be run from parent dir
import utils.experiment as utils


EXPERIMENT_DIR = './experiments'

def add_val(di,keys,vals):
    udi = copy.deepcopy(di)
    for k,v in zip(keys,vals):
        udi[k] = v

    return udi

def get_train_configs(data,seeds):
    list_params = []
    for k in data:
        if isinstance(data[k],list):
            list_params.append(k)
    tc = []
    if len(list_params) == 0:
        tc = tc + [add_val(data,['seed','run'],[s,i]) for i,s in enumerate(seeds)]
    elif len(list_params) == 1:
        k = list_params[0]
        for p in data[k]:
            subdir = k + '_%s' % str(p)
            odir = os.path.join(data['odir'],os.path.join(args.name,subdir))
            tc = tc + [add_val(data,['seed','run',list_params[0],'odir'],[s,i,p,odir]) for i,s in enumerate(seeds)]
    else:
        raise RuntimeError('Not Supported Yet')

    return tc

if __name__ == '__main__':
    # LOAD CONFIG FILE
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='name')
    parser.add_argument('-n','--num_proc', type=int, default=1, help='How many processes to use')    
    args = parser.parse_args()

    with open(os.path.join(EXPERIMENT_DIR,args.name+'.json')) as f:
        config = json.load(f)
    with open(os.path.join(EXPERIMENT_DIR,'seeds.txt')) as f:
        seedreader = csv.reader(f)
        seeds = [int(''.join(a)) for a in seedreader]

    # MAKE TRAIN CONFIGS
    num_seeds = config.pop('num_seeds')
    assert num_seeds <= len(seeds),"Too many seeds specified"
    seeds = seeds[:num_seeds]
    tcs = get_train_configs(config,seeds)

    # RUN EXPERIMENTS
    with Pool(args.num_proc) as p:
        p.map(utils.run_training, tcs)
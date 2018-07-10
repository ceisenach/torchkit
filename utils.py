import numpy as np
import torch
import os
import argparse
import time
import math

def make_directory(dirpath):
    os.makedirs(dirpath,exist_ok=True)

def experiment_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument("-o", "--odir", type=str, default=None, help="output directory")
    # parser.add_argument("-p",'--policy' ,type=str, default='angular', help="policy type to use")
    parser.add_argument("-u",'--num_updates', type=float, default=1e4, help="number of gradient updates")
    parser.add_argument("-g","--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("-N", type=int, default=15000, help="Batch Size")
    parser.add_argument("-s",'--save_interval', type=float, default=1e3, help="Model Save Interval")
    parser.add_argument("-l",'--log_interval', type=int, default=50, help="Log Interval")
    parser.add_argument("-E",'--env', type=str, default="BipedalWalker-v2", help="Environment to use")
    parser.add_argument("-co", "--console", action="store_true", help="log to console")
    parser.add_argument('--tau', type=float, default=0.97, metavar='G',help='gae (default: 0.97)')
    parser.add_argument('--l2_pen', type=float, default=1e-3, metavar='G',help='l2 regularization regression (default: 1e-3)')
    parser.add_argument('--max_kl', type=float, default=1e-2, metavar='G',help='max kl value (default: 1e-2)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',help='random seed (default: 1)')
    parser.add_argument('--damping', type=float, default=1e-1, metavar='G',help='damping (default: 1e-1)')
    return parser

def train_config_from_args(args):
    experiment_config = {'gamma' : args.gamma,
                         'tau' : args.tau,
                         'max_kl' : args.max_kl,
                         'l2_pen' : args.l2_pen,
                         'lr' : args.lr,
                         'damping' : args.damping,
                         # 'policy' : args.policy,
                         'seed' : args.seed,
                         'num_updates' : int(args.num_updates),
                         'N' : args.N,
                         'env' : args.env,
                         'console' : args.console,
                         'odir' : args.odir if args.odir is not None else 'out/experiment_%s' % time.strftime("%Y.%m.%d_%H.%M.%S"),
                         'save_interval' : int(args.save_interval)}

    return experiment_config


class MultiRingBuffer(object):
    """
    Ring buffer that supports multiple data of different widths
    """
    def __init__(self, experience_shapes, max_len,tensor_type=torch.FloatTensor):
        assert isinstance(experience_shapes, list)
        assert len(experience_shapes) > 0
        assert isinstance(experience_shapes[0], tuple)
        self.maxlen = max_len
        self.start = 0
        self.length = 0
        self.dataList = [tensor_type(max_len, *shape).zero_() for shape in experience_shapes]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return [data[(self.start + idx) % self.maxlen] for data in self.dataList]

    def append(self, *vs):
        """
        Append to buffer
        """
        vs = list(vs)
        for i, v in enumerate(vs):
            if isinstance(v, np.ndarray):
                vs[i] = torch.from_numpy(v)
        if self.length < self.maxlen:
            # Space Available
            self.length += 1
        elif self.length == self.maxlen:
            # No Space, remove the first item
            self.start = (self.start + 1) % self.maxlen
        else:
            # Should not happen
            raise RuntimeError()
        for data, v in zip(self.dataList, vs):
            data[(self.start + self.length - 1) % self.maxlen] = v.squeeze()

    def reset(self):
        """
        Clear replay buffer
        """
        self.start = 0
        self.length = 0

    def get_data(self):
        """
        Get all data in the buffer
        """
        if self.length < self.maxlen:
            return [data[0:self.length] for data in self.dataList]
        return self.dataList

# from ikostrikov
def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad


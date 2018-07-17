# from ikostrikov

import torch
import torch.nn as nn
import autodiff.functional as F
import autodiff.modules as mod


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std



class Policy2(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy2, self).__init__()
        self.affine1 = mod.Linear(num_inputs, 10)

        self.action_mean = mod.Linear(10, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(num_outputs))


    def forward(self, x,save_for_jacobian=False,**kwargs):
        x = self.affine1(x,save_for_jacobian)
        x = F.tanh(x,save_for_jacobian)
        if save_for_jacobian:
            x = self.action_mean(x,save_for_jacobian)
            return x

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)

        return action_mean, action_log_std, action_std



class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.tanh(self.affine1(x))
        x = F.tanh(self.affine2(x))

        state_values = self.value_head(x)
        return state_values

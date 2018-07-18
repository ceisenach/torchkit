# from ikostrikov

import torch
# import torch.nn as nn
import autodiff.functional as F
import autodiff.modules as nn


class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)

        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(data=torch.zeros(num_outputs))


    def forward(self, x,save_for_jacobian=False,**kwargs):
        x = self.affine1(x,save_for_jacobian)
        x = F.tanh(x,save_for_jacobian)
        x = self.affine2(x,save_for_jacobian)
        x = F.tanh(x,save_for_jacobian)
        action_mean = self.action_mean(x,save_for_jacobian)
        action_log_std = self.action_log_std.expand_as(action_mean.data)

        return action_mean, action_log_std
        


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

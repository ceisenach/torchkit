import torch
import torch.nn.init as init
import autodiff.functional as F
import autodiff.modules as nn


# Original TRPO uses 2 hidden layers with width 64

class FFNet(nn.Module):
    def __init__(self,in_size,out_size=1,width=32,hidden_layers = 2,**kwargs):
        super(FFNet, self).__init__()
        self._hidden_layers = hidden_layers
        self._layer_name = 'affine_%d'

        # add layers
        for i in range(self._hidden_layers):
            layer_in = in_size if i == 0 else width
            self.add_module(self._layer_name % i, nn.Linear(layer_in, width))

        # initialize layers
        for i in range(self._hidden_layers):
            layer_module = getattr(self,self._layer_name % i)
            init.xavier_uniform_(layer_module.weight)
            init.constant_(layer_module.bias,0.)

        # output layer
        self.affine_out = nn.Linear(width, out_size)
        init.xavier_uniform_(self.affine_out.weight)
        init.constant_(self.affine_out.bias,0.)

    def forward(self,x,save_for_jacobian=False,**kwargs):
        for i in range(self._hidden_layers):
            x = F.tanh(getattr(self,self._layer_name % i)(x,save_for_jacobian),save_for_jacobian)
        x = self.affine_out(x,save_for_jacobian)
        return x

# Alias
Value = FFNet

class Policy(FFNet):
    def __init__(self,in_size,out_size,width=32,hidden_layers = 2,**kwargs):
        super(Policy, self).__init__(in_size,out_size,width,hidden_layers,**kwargs)
        self.action_log_std = nn.Parameter(data=torch.zeros(out_size))

    def forward(self,x,save_for_jacobian=False,**kwargs):
        out = super(Policy,self).forward(x,save_for_jacobian,**kwargs)
        action_log_std = self.action_log_std.expand_as(out.data)

        return out, action_log_std
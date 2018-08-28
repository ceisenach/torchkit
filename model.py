import torch
import torch.nn.init as init
import autodiff.functional as F
import autodiff.modules as nn
import torch.nn.functional as TF

# Original TRPO uses 2 hidden layers with width 64
# normc is initialization procedure from open AI baselines
def normc_initializer(param,std=1.0, axis=-1):
    with torch.no_grad():
        d = torch.randn(param.shape)
        d2 = d.pow(2)
        csum = d2.sum(axis,keepdim=True)
        csum_sqrt = torch.sqrt(csum)
        scale = std / csum_sqrt
        param.data = d*scale

class FFNet_base(nn.Module):
    def __init__(self,in_size,out_size=1,width=32,hidden_layers = 2,init_std=1.0,**kwargs):
        super(FFNet_base, self).__init__()
        self._hidden_layers = hidden_layers
        self._layer_name = 'affine_%d'

        # add layers
        for i in range(self._hidden_layers):
            layer_in = in_size if i == 0 else width
            self.add_module(self._layer_name % i, nn.Linear(layer_in, width))

        # initialize layers
        for i in range(self._hidden_layers):
            layer_module = getattr(self,self._layer_name % i)
            normc_initializer(layer_module.weight,init_std,axis=-1)
            normc_initializer(layer_module.bias,init_std,axis=-1)

        # output layer
        self.affine_out = nn.Linear(width, out_size)
        normc_initializer(self.affine_out.weight,init_std,axis=-1)
        normc_initializer(self.affine_out.bias,init_std,axis=-1)


    def forward(self,x,save_for_jacobian=False,**kwargs):
        for i in range(self._hidden_layers):
            x = F.tanh(getattr(self,self._layer_name % i)(x,save_for_jacobian),save_for_jacobian)
        x = self.affine_out(x,save_for_jacobian)
        return x

# Alias

class Value(FFNet_base):
    def __init__(self,in_size,out_size=1,width=32,hidden_layers = 2,**kwargs):
        super(Value, self).__init__(in_size,out_size,width,hidden_layers,init_std=1.0,**kwargs)

class Policy(FFNet_base):
    def __init__(self,in_size,out_size,width=32,hidden_layers = 2,**kwargs):
        super(Policy, self).__init__(in_size,out_size,width,hidden_layers,init_std=1.0,**kwargs)
        self.action_log_std = nn.Parameter(data=torch.zeros(out_size))

        #override output init
        normc_initializer(self.affine_out.weight,0.01,axis=-1)
        normc_initializer(self.affine_out.bias,0.01,axis=-1)

    def forward(self,x,save_for_jacobian=False,**kwargs):
        out = super(Policy,self).forward(x,save_for_jacobian,**kwargs)
        action_log_std = self.action_log_std.expand_as(out.data)

        return out, action_log_std

class PolicyBeta(nn.Module):
    def __init__(self,in_size,out_size,width=32,hidden_layers = 2,**kwargs):
        super(PolicyBeta, self).__init__()
        self.alpha = FFNet_base(in_size,out_size,width,hidden_layers,init_std=1.0,**kwargs)
        self.beta = FFNet_base(in_size,out_size,width,hidden_layers,init_std=1.0,**kwargs)

        #override output init
        normc_initializer(self.alpha.affine_out.weight,0.01,axis=-1)
        normc_initializer(self.alpha.affine_out.bias,0.01,axis=-1)
        normc_initializer(self.beta.affine_out.weight,0.01,axis=-1)
        normc_initializer(self.beta.affine_out.bias,0.01,axis=-1)

    def forward(self,x,save_for_jacobian=False,**kwargs):
        aout = F.softplus(self.alpha(x,save_for_jacobian,**kwargs))
        bout = F.softplus(self.beta(x,save_for_jacobian,**kwargs))

        return aout,bout
import torch
import torch.nn.init as init
import autodiff.functional as F
import autodiff.modules as nn
import torch.nn.functional as TF
import itertools

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

class OrderedModule(nn.Module):

    def parameters(self,ordered=False,**kwargs):
        if not ordered or not hasattr(self,'_param_order'):
            return super(OrderedModule,self).parameters(**kwargs)
        return itertools.chain(*[m.parameters() for m in self._param_order])
        

class FFNet_base(OrderedModule):
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

class PolicyGauss2(OrderedModule):
    def __init__(self,in_size,out_size,width=32,hidden_layers = 2,**kwargs):
        super(PolicyGauss2, self).__init__()
        self.param_1 = FFNet_base(in_size,out_size,width,hidden_layers,init_std=1.0,**kwargs)
        self.param_2 = nn.Parameter(data=torch.zeros(out_size))
        self.action_log_std = self.param_2

        #override output init
        normc_initializer(self.param_1.affine_out.weight,0.01,axis=-1)
        normc_initializer(self.param_1.affine_out.bias,0.01,axis=-1)

        # self._param_order = [self.param_1,self.param_2]

    def forward(self,x,save_for_jacobian=False,**kwargs):
        out = self.param_1.forward(x,save_for_jacobian,**kwargs)
        action_log_std = F.expand(self.action_log_std,out.data.shape,save_for_jacobian=save_for_jacobian)

        return out, action_log_std


class PolicyExp2P(OrderedModule):
    def __init__(self,in_size,out_size,width=32,hidden_layers = 2,**kwargs):
        super(PolicyExp2P, self).__init__()
        self.param_1 = FFNet_base(in_size,out_size,width,hidden_layers,init_std=1.0,**kwargs)
        self.param_2 = FFNet_base(in_size,out_size,width,hidden_layers,init_std=1.0,**kwargs)

        #override output init
        normc_initializer(self.param_1.affine_out.weight,0.01,axis=-1)
        normc_initializer(self.param_1.affine_out.bias,0.01,axis=-1)
        normc_initializer(self.param_2.affine_out.weight,0.01,axis=-1)
        normc_initializer(self.param_2.affine_out.bias,0.01,axis=-1)

        self._param_order = [self.param_1,self.param_2]

    def forward(self,x,save_for_jacobian=False,**kwargs):
        return self.param_1.forward(x,save_for_jacobian,**kwargs),self.param_2.forward(x,save_for_jacobian,**kwargs)


class PolicyBeta(PolicyExp2P):
    def __init__(self,*args,**kwargs):
        super(PolicyBeta, self).__init__(*args,**kwargs)
        self.alpha = self.param_1
        self.beta = self.param_2

    def forward(self,x,save_for_jacobian=False,**kwargs):
        aout,bout = super(PolicyBeta,self).forward(x,save_for_jacobian=save_for_jacobian,**kwargs)
        aout = F.softplus(aout,save_for_jacobian=save_for_jacobian)
        bout = F.softplus(bout,save_for_jacobian=save_for_jacobian)
        return aout,bout


class PolicyGauss(PolicyExp2P):
    def __init__(self,*args,**kwargs):
        super(PolicyGauss, self).__init__(*args,**kwargs)

    def forward(self,x,save_for_jacobian=False,**kwargs):
        aout,bout = super(PolicyGauss,self).forward(x,save_for_jacobian=save_for_jacobian,**kwargs)
        return aout,bout

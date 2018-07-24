import torch
import time
import autodiff.util as util

shape1 = [100, 4, 4]
shape2 = [100,1,64]

for i in range(10000):
    # a = torch.ones(shape1)
    # b = torch.ones(shape2)
    # I_oT = util.bkron(a,b)
    # del I_oT


    time.sleep(0.01)
import torch



def tensor_almost_equal(a,b,eps=1e-6):
    diff = torch.abs(a-b)
    delta = diff / (torch.abs(a)*torch.abs(b) + eps) # make sure not div by zero

    # get max deviation
    delta_imax,imax = delta.max(dim=0)
    diff_imax = diff[imax]

    delta_imax,imax,diff_imax = delta_imax.item(),imax.item(),diff_imax.item()

    # as long as one metric is close enough
    almost_equal = delta_imax < eps or diff_imax < eps

    return almost_equal,delta_imax,imax,diff_imax
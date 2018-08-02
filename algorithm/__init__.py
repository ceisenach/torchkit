import torch

#######################################
# Core Classes

class AlgorithmBase(object):

    def __init__(self,policy,critic,args,**kwargs):
        self._args = args
        self._policy = policy
        self._actor = policy.net
        self._critic = critic

    def _batch_prepare(self,batch):
        raise NotImplementedError()

    def _batch_prepare_gae_td0_return(self,batch):
        """
        compute advantages for each batch using GAE on full trajectories. G is TD(0) return
        """
        with torch.no_grad():
            value_net,gamma,tau = self._critic, self._args['gamma'], self._args['tau']
            S,A,M,R = batch
            R,M = R.view(-1,1),M.view(-1,1)
            V = self._critic(S).view(-1,1)
            N = S.shape[0]

            G = torch.zeros(N,1) # Bootstrapped Returns
            delta = torch.zeros(N,1)
            U = torch.zeros(N,1)

            G[-1,:] = V[-1,:]
            for i in range(N-1):
                G[N-i-2,:] = R[N-i-2,:] + self._args['gamma'] * G[N-i-1,:] * M[N-i-2,:]
                delta[N-i-2,:] = R[N-i-2,:] + self._args['gamma'] * V[N-i-1,:] * M[N-i-2,:] - V[N-i-2,:]
                U[N-i-2,:] = delta[N-i-2,:] + self._args['gamma'] * self._args['tau'] * U[N-i-1,:] * M[N-i-2,:]

            S,A,G,U = S[:-1],A[:-1],G[:-1],U[:-1]
            U = (U - U.mean()) / U.std()
            return S,A,G,U

    def _batch_prepare_gae_lambda_return(self,batch):
        """
        compute advantages for each batch using GAE on full trajectories and also return TD(lambda) return
        instead of TD(0) return. Analogous to TRPO_MPI in baselines
        """
        with torch.no_grad():
            value_net,gamma,tau = self._critic, self._args['gamma'], self._args['tau']
            S,A,M,R = batch
            R,M = R.view(-1,1),M.view(-1,1)
            V = self._critic(S).view(-1,1)
            N = S.shape[0]

            delta = torch.zeros(N,1)
            U = torch.zeros(N,1)

            for i in range(N-1):
                delta[N-i-2,:] = R[N-i-2,:] + self._args['gamma'] * V[N-i-1,:] * M[N-i-2,:] - V[N-i-2,:]
                U[N-i-2,:] = delta[N-i-2,:] + self._args['gamma'] * self._args['tau'] * U[N-i-1,:] * M[N-i-2,:]

            S,A,U = S[:-1],A[:-1],U[:-1]
            G = U + V[:-1,:]
            return S,A,G,U

    def _batch_prepare_advantages(self,batch):
        # Compute advantages
        with torch.no_grad():
            S,A,M,R = batch
            N = S.shape[0]
            r_tp1 = R[:-1].view(-1,1)

            G = torch.zeros(N,1) # Bootstrapped Returns
            V = self._critic(S) # Value estimates
            G[-1,:] = V[-1]

            for i in range(N-1):
                G[N-i-2,:] = r_tp1[N-i-2,:] + self._args['gamma'] * G[N-i-1,:] * M[N-i-2,:]
            G = G[:-1]
            U = G - V[:-1] # Advantages
            return S[:-1],A[:-1],G,U

    def _batch_merge(self,batch_list):
        """
        merge independent batches together -- should always return a 4-tuple
        """
        if not isinstance(batch_list[0],list):
            import pdb; pdb.set_trace()
            return self._batch_prepare(batch_list)
        S,A,R,U = [],[],[],[]
        for b in filter(lambda bt: bt[0] is not None, batch_list):
            s,a,r,u = self._batch_prepare(b)
            S.append(s)
            A.append(a)
            U.append(u)
            R.append(r)
        if len(S) == 0:
            return None
        return torch.cat(S,dim=0),torch.cat(A,dim=0),torch.cat(R,dim=0),torch.cat(U,dim=0)




###########################################
# Import commonly used names
from .trpo import *
from .nac import *
from .a2c import *
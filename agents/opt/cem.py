import numpy as np
from agents.opt.base import *
import torch as th

# Adapted from https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/misc/optimizers/cem.py
class CEM(baseOptimizer):
    def __init__(self,
            nA,
            traj_len,
            num_particles,
            min_val,
            max_val,
            obj_function,
            elite_ratio=0.2,
            num_iters=5,
            epsilon=0.001,
            alpha=0.25,
            init_mu=0.0,
            init_var=1.0,
            return_states=False,
            ):
        super(CEM,self).__init__(nA, traj_len, num_particles, min_val, max_val, obj_function, return_states)
        self.eps = epsilon
        self.alpha = alpha
        self.num_elite = min(num_particles, int(num_particles*elite_ratio))
        self.num_iters = num_iters
        self.init_mu = init_mu
        self.init_var = init_var

    def reset(self):
        pass

    def get_sol(self, init_state, rollout=None):
        t = 0
        if not rollout:
            rollout = self.traj_len

        self.mu = np.tile(self.init_mu, (rollout, self.nA))
        self.var = np.tile(self.init_var, (rollout, self.nA))
        mu, var = self.mu, self.var
        samples = None
        state_trajs = tensor = th.reshape(th.tensor(()), (0,))


        while t < self.num_iters and np.max(var) > self.eps:
            min_dist, max_dist = mu - self.min, self.max - mu
            constrained_var = np.minimum(np.minimum(np.square(min_dist / 2), np.square(max_dist / 2)), var)
            samples = np.random.normal(mu, np.sqrt(constrained_var), size=(self.num_particles, rollout, self.nA))

            rew_traj = self.cost(init_state, samples, self.return_states)
            if self.return_states:
                rew_traj, state_trajs = rew_traj

            rews = th.sum(rew_traj, dim=1).detach().numpy()

            # Convert rews back to a PyTorch tensor if needed
            rews = th.tensor(rews) if isinstance(rews, np.ndarray) else rews

            elites = samples[th.argsort(rews, descending=True)][-self.num_elite:,:,:]
            #print(rew_traj)
            #print(rews)

            new_mu = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1

        sol, solvar = mu, var
        sort = th.argsort(rews, descending=True)
        samples = samples[sort,:,:]
        state_trajs = th.gather(state_trajs, sort) if self.return_states else None
        return (samples, state_trajs) if self.return_states else sol





class MPPI(baseOptimizer):
    def __init__(self,
                 nA,  
                 traj_len,  
                 num_particles, 
                 min_val,  
                 max_val,  
                 obj_function,  
                 lambda_=1.0,  
                 horizon=10,  
                 alpha=0.1, 
                 epsilon=0.01,  
                 return_states=False):  
        super(MPPI, self).__init__(nA, traj_len, num_particles, min_val, max_val, obj_function, return_states)
        
        self.lambda_ = lambda_
        self.horizon = horizon
        self.alpha = alpha
        self.epsilon = epsilon

    def reset(self):
        pass  

    def get_sol(self, init_state, rollout=None):
        t = 0
        if not rollout:
            rollout = self.traj_len

       
        mu = np.zeros((self.horizon, self.nA))  
        var = np.ones((self.horizon, self.nA))  

        while t < self.horizon: 
            samples = np.random.normal(mu, np.sqrt(var), size=(self.num_particles, self.horizon, self.nA))
            samples = th.tensor(samples)
            rew_traj = self.cost(init_state, samples, self.return_states)
            if self.return_states:
                rew_traj, state_trajs = rew_traj
            rew_traj = th.tensor(rew_traj) if not isinstance(rew_traj, th.Tensor) else rew_traj
            rews = th.sum(rew_traj, axis=1)  
            weights = th.exp(-self.lambda_ * rews)  
            weights /= th.sum(weights) 
            new_mu = th.sum(samples * weights.unsqueeze(-1), dim=0) 
            new_var = th.sum((samples - new_mu) ** 2 * weights.unsqueeze(-1), dim=0) 
            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            var = self.alpha * var + (1 - self.alpha) * new_var
            if np.max(var) < self.epsilon:
                break

            t += 1
        rews = rews.detach().numpy() 
        sort = th.argsort(rews, descending=True)
        samples = samples[sort, :, :]
        state_trajs = th.gather(state_trajs, sort) if self.return_states else None
        return (samples, state_trajs) if self.return_states else mu

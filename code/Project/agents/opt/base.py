class baseOptimizer():
    def __init__(self,
            act_dim,
            traj_len,
            num_particles,
            min_val,
            max_val,
            obj_function,
            return_states,
            ):
        self.nA = act_dim
        self.traj_len = traj_len
        self.num_particles = num_particles
        self.max = max_val
        self.min = min_val
        self.cost = obj_function
        self.return_states = return_states

    def reset(self):
        pass

    def get_sol(self, init_state, rollout=None):
        raise NotImplementedError

    def get_sols(self, init_states, rollout=None):
        if rollout == None:
            rollout = self.traj_len
        sols = []
        for i in range(len(init_states)):
            sols.append(self.get_sol(init_states[i], rollout=rollout))

        return sols

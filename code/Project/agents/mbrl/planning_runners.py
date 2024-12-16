# Adopted from https://keras.io/examples/rl/ddpg_pendulum/
from base import *
from agents.mbrl.model_runners import modelRunner

class planningRunner(modelRunner):
    def __init__(self,
            env,
            agent,
            collect_random_exp=False,
            *agent_args,
            ):
        super(planningRunner, self).__init__(env, agent, agent_args)
        self._collect_random_exp = collect_random_exp

    # Train the agent with the specified number of episodes and maximum steps per episode

    def train_ep(self, num_steps, render):
        if self._collect_random_exp:
            self.collect_experiences()
        prev_obs = self._env.reset()[0]
        for i in range(len(self._agents)):
            self._agents[i].reset()

        tot_rew = 0.0
        info_list = []

        for i in range(num_steps):
            for j in range(len(self._agents)):
                self._agents[j].reset()
            if render:
                self._env.render()
            acts = self._sample_actions(prev_obs)
            obs, rew, done,truncated, info = self._env.step(acts)
            tot_rew += rew

            model_hist = self._train_agents(prev_obs, acts, rew, obs, done or truncated)
            if model_hist is not None:
                info_list.append({'info':info, 'model_history':model_hist})

            if done or truncated:
                break

            prev_obs = obs

        return tot_rew, info_list 

    def collect_validation_data(self, num_eps=50):
        for j in range(num_eps*500):
            prev_obs = self._env.reset()[0]
           
            acts = self._sample_actions(prev_obs)
            obs, rew, done,truncated, info = self._env.step(acts)

            self._push_val_data(prev_obs, acts, rew, obs, done)
            
            prev_obs = obs


    def _push_val_data(self, prev_obs, acts, rew, obs, done):
        for i in range(len(self._agents)):
            a_obs = prev_obs[i]
            act = acts[i]
            a_nobs = obs[i]
            self._agents[i].push_val_data(a_obs, act, a_nobs, rew)

    def collect_experiences(self):
        self._env.reset()
        while not self._agents[0]._trained:
            prev_obs = self._env.reset()[0]

            for i in range(100):
                acts = self._sample_actions(prev_obs)
                obs, rew, done,truncated, info = self._env.step(acts)

                self._train_agents(prev_obs, acts, rew, obs, done)
             
                prev_obs = obs
                if done or truncated:
                    break

    def eval_ep(self, num_steps, render, render_mode, reset_env=True, prev_obs=None):

        if reset_env:
            prev_obs = self._env.reset()[0]
    
        for i in range(len(self._agents)):
            self._agents[i].reset()

        tot_rew = 0.0
        info_list = []
        imgs = []

        for i in range(num_steps):
            if render:
                imgs.append(self._env.render(render_mode))

            acts, pred_traj = self._noiseless_sample_actions(prev_obs)
            obs, rew, done,truncated, info = self._env.step(acts)
            tot_rew += rew

            #if render:
            #    self._env.renderer.draw_onetime_traj(prev_obs, pred_traj)
 
            info_list.append(info)

            if done or truncated:
                break

            prev_obs = obs

        return tot_rew, info_list, imgs

    # Take the observations and split them up in order to sample actions for all of the agents
    def _noiseless_sample_actions(self, obs):
        actions = []
        pred_states = []
        for i in range(len(obs)):
            acts, states = self._agents[i].noiseless_sample_action(obs[i])
            actions.append(acts)
            pred_states.append(states)

        return actions, pred_states

# class goalPlanningRunner(planningRunner):
#     def __init__(self, 
#             env,
#             agent,
#             option_dim=3,
#             *agent_args
#             ):
#         super(goalModelRunner, self).__init__(env, agent_args)
#         self.goal_pos = (0, 0, 0)

#     def set_goal(self, goal):
#         self.goal_pos = goal

#     # Take the observations and split them up in order to sample actions for all of the agents
#     def _sample_actions(self, obs):
#         actions = []
#         for i in range(len(obs)):
#             actions.append(self._agents[i].sample_action(obs[i], self.goal_pos))

#         return actions

#     # Take the observations and split them up in order to sample actions for all of the agents
#     def _noiseless_sample_actions(self, obs):
#         actions = []
#         for i in range(len(obs)):
#             actions.append(self._agents[i].noiseless_sample_action(obs[i], self.goal_pos))

#         return actions



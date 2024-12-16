# Adopted from https://keras.io/examples/rl/ddpg_pendulum/
from base import *
import torch
import numpy as np

# This class encapsulates each individual learning agent and handles the training and action sampling processes
class modelRunner(BaseRunner):
    def __init__(self, 
            env, agent, *agent_args
            ):
        super(modelRunner, self).__init__(env)
        # We spawn the number of agents implied by the shape of the environments observations
        for i in range(self._num_agents):
            ag = agent(env, self._num_agents)#, agent_args)
            self._agents.append(ag)

    # Run a single training episode
    def train_ep(self, num_steps, render):
        prev_obs = self._env.reset()[0]
        for i in range(len(self._agents)):
            self._agents[i].reset()

        tot_rew = 0.0
        info_list = []

        for i in range(num_steps):
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

    def eval_ep(self, num_steps, render, render_mode):
        prev_obs = self._env.reset()[0]
        for i in range(len(self._agents)):
            self._agents[i].reset()

        tot_rew = 0.0
        info_list = []
        imgs = []

        for i in range(num_steps):
            if render:
                imgs.append(self._env.render(render_mode))
            acts = self._noiseless_sample_actions(prev_obs)
            obs, rew, done,truncated, info = self._env.step(acts)
            tot_rew += rew

            info_list.append(info)

            if done or truncated:
                break

            prev_obs = obs

        return tot_rew, info_list, imgs


    # Take the observations and split them up in order to sample actions for all of the agents
    # def _sample_actions(self, obs):
    #     actions = []
    #     for i in range(len(obs)):
    #         actions.append(self._agents[i].sample_action(obs[i]))

    #     return actions
    def _sample_actions(self, obs):
        return self._agents[0].sample_action(obs)

    # Take the observations and split them up in order to sample actions for all of the agents
    # def _noiseless_sample_actions(self, obs):
    #     actions = []
    #     for i in range(len(obs)):
    #         actions.append(self._agents[i].noiseless_sample_action(obs[i]))

    #     return actions
    def _noiseless_sample_actions(self, obs):
        return self._agents[0].noiseless_sample_action(obs)

  

    def _train_agents(self, observations, acts, rew, next_observations, done):
        model_hist = [[]]*len(self._agents)
        for i in range(len(self._agents)):
            obs = observations[i]
            act = acts[i]
            next_obs = next_observations[i]
            model_hist[i].append(self._agents[i].train_step(obs, act, rew, next_obs, done))
        return model_hist

class goalModelRunner(modelRunner):
    def __init__(self, 
            env,
            agent,
            option_dim=3,
            ):
        super(goalModelRunner, self).__init__(env)
        self.goal_pos = (0, 0, 0)

    def set_goal(self, goal):
        self.goal_pos = goal

    # Take the observations and split them up in order to sample actions for all of the agents
    def _sample_actions(self, obs):
        actions = []
        for i in range(len(obs)):
            actions.append(self._agents[i].sample_action(obs[i], self.goal_pos))

        return actions

    # Take the observations and split them up in order to sample actions for all of the agents
    def _noiseless_sample_actions(self, obs):
        actions = []
        for i in range(len(obs)):
            actions.append(self._agents[i].noiseless_sample_action(obs[i], self.goal_pos))

        return actions


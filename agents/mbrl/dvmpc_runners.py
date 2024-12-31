# Adopted from https://keras.io/examples/rl/ddpg_pendulum/
from base import *
from agents.mbrl.planning_runners import *
from agents.mbrl.dvmpc import *
from agents.mbrl.model_runners import modelRunner
import torch
import numpy as np
from agents.utils import * 
from tqdm import tqdm
import gymnasium as gym


class DVPMCRunner:
    def __init__(
            self,
            env,
            *agent_args
            ):
        self._collect_random_exp = False
        self.count=0
        self._env = env
        self._best_ep_rew = float('-inf')
        self._best_avg_rew = float('-inf')
        self._agent = DVPMCAgent(env)
        self._full_dyn_model_hist = {"tloss":[],"vloss":[]}
        self._full_val_model_hist = []

    def train_ep(self, num_steps, render):
        self.count+=1
        print("this is the main train episode")
        print(f"Train ep: {self.count}")
        if self._collect_random_exp:
            self.collect_experiences()
        prev_obs = self._env.reset()[0]
        self._agent.reset()

        tot_rew = 0.0
        info_list = []

        for i in range(num_steps):
            if render:
                self._env.render()
            acts = self._sample_actions(prev_obs)
            obs, rew, done,truncated, info= self._env.step(acts)
            tot_rew += rew
            
            dyn_model_hist, val_model_hist  = self._train_agent(prev_obs, acts, rew, obs, done)

            if dyn_model_hist is not None:
                self._full_dyn_model_hist["tloss"].extend(dyn_model_hist["tloss"])
                self._full_dyn_model_hist["vloss"].extend(dyn_model_hist["vloss"])
                info_list.append({'info':info, 'step':i+1, 'value_loss':val_model_hist, 'dyn_model_history':dyn_model_hist})
            elif val_model_hist is not None:
                self._full_val_model_hist.append(val_model_hist)
                info_list.append({'info':info, 'step':i+1, 'value_loss':val_model_hist})

            if done or truncated:
                break

            prev_obs = obs

        return tot_rew, info_list 


    def collect_validation_data(self, num_eps=50):
        for j in range(num_eps*500):
            prev_obs = self._env.reset()[0]  
            acts = self._sample_actions(prev_obs)
            # print("First ",acts,prev_obs)
            obs, rew, done,truncated, info = self._env.step(acts)
            # print("Second ",rew,obs)
            self._push_val_data(prev_obs, acts, rew, obs, done)
            prev_obs = obs


    def _push_val_data(self, prev_obs, acts, rew, obs, done):
        a_obs = prev_obs
        act = acts
        a_nobs = obs
        self._agent.push_val_data(a_obs, act, a_nobs, rew)

    def collect_experiences(self):
        self._env.reset()
        while not self._agent._trained:
            prev_obs = self._env.reset()[0]
            for i in range(100):
                acts = self._sample_actions(prev_obs)
                obs, rew, done,truncated, info = self._env.step(acts)

                self._train_agent(prev_obs, acts, rew, obs, done)
             
                prev_obs = obs
                if done or truncated:
                    break

    # Train the agent with the specified number of episodes and maximum steps per episode
    def train(self, num_eps, num_steps=1000, render=False, render_step=10):

        ep_rew_max=1000
        ep_rew_max_count = 0
        self.collect_validation_data()  
        #Function for collecting data, if the model is untrained, 
        #actions are random sampled from a random distribution,
        #populates the val buffer of the Dynamics model

        ep_rewards = [] #Collecting all episode rewards
        
        info = []
        t = tqdm(range(num_eps), desc='Episodic reward: ', position=0, leave=True)
        for i in t:
            #if i%10 == 0:
            #    self.collect_validation_data(1)
            print("training episode???????????????????????????????")
            ep_rew, tmp_info = self.train_ep(num_steps, (render and (i%render_step)==0))
            print("training episode?Done")
            ep_rewards.append(ep_rew)
            avg = np.average(ep_rewards[-10:]) #only last 10 rewards
            info.extend(tmp_info)
            t.set_description("Episodic reward: {} | Avg reward: {} ".format(ep_rew, avg), refresh=True)

            if ep_rew > self._best_ep_rew:
                self._best_ep_rew = ep_rew
                print('Best rew: {}'.format(ep_rew))
                # for j in range(len(self._agents)):
                    # self._agents[j].set_best()
                self._agent.set_best()

            if (avg > self._best_avg_rew) and i>=9:
                self._best_avg_rew = np.squeeze(avg)
                print('Best avg rew: {}'.format(avg))
                # for j in range(len(self._agents)):
                #     self._agents[j].set_best_avg()
                self._agent.set_best_avg()
            if avg>900:
                ep_avg_count+=1
                if ep_rew_max_count==8: break

        return ep_rewards, info

    def eval(self, num_eps, num_steps=10000, render=False, render_mode='human'):
        ep_rewards = []
        in_rewards =[]
        info = []
        imgs = []
        t = tqdm(range(num_eps), desc='Episodic reward: ', position=0, leave=True)
        for i in t:
            ep_rew, tmp_info, img = self.eval_ep(num_steps, render, render_mode)
            ep_rewards.append(ep_rew)
            info.extend(tmp_info)
            imgs.append(img)
            t.set_description("Episodic reward: " + str(ep_rew), refresh=True)

        return ep_rewards, info, imgs
    

    def save_agent(self, save_dir, save_type='best'):
        self._agent.save_models(save_dir + save_type + '/' + "agent", save_type)

    # Load the given agent neural nets
    def load_agent(self, save_dir):
        self._agent.load_models(save_dir + "agent")

    def dump_agent_attrs(self, save_dir):
        attr = self._agent.__dict__
        with open(save_dir + 'agent' + '.txt', 'w') as f:
            f.write(str(attr))

    # def _train_agent(self, observations, acts, rew, next_observations, done):
    #     model_hist = []
    #     obs = observations
    #     act = acts
    #     next_obs = next_observations
    #     model_hist.append(self._agent.train_step(obs, act, rew, next_obs, done))
    #     return model_hist


    def _train_agent(self, observations, acts, rew, next_observations, done):
        obs = observations
        act = acts
        next_obs = next_observations
        model_hist, v_model_loss = self._agent.train_step(obs, act, rew, next_obs, done)
        return model_hist, v_model_loss

    def eval_ep(self, num_steps, render, render_mode, reset_env=True, prev_obs=None):

        if reset_env:
            prev_obs = self._env.reset()[0]
    
        
        self._agent.reset()

        tot_rew = 0.0
        info_list = []
        imgs = []

        for i in range(num_steps):
            if render:
                imgs.append(self._env.render(render_mode))

            acts, pred_traj, times = self._noiseless_sample_actions(prev_obs)
            obs, rew, done,truncated, info = self._env.step(acts)
            tot_rew += rew

            #if render:
            #    self._env.renderer.draw_onetime_traj(prev_obs, pred_traj)
            info['times'] = times
 
            info_list.append(info)

            if done or truncated:

                break

            prev_obs = obs



        return tot_rew, info_list, imgs

    def _noiseless_sample_action(self, obs):
        acts, states, time = self._agent.noiseless_sample_action(obs)
        actions.append(act)
        pred_states.append(stat)
        return action, pred_state, time
    
    def _sample_actions(self, obs):
        return self._agent.sample_action(obs)


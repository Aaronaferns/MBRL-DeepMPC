# Adopted from https://keras.io/examples/rl/ddpg_pendulum/

import numpy as np
from agents.utils import * 
from tqdm import tqdm
import gymnasium as gym

# A base class for the running/training objects for the agents
class BaseRunner():
    def  __init__(
           self,
           env,
           ):
        self._env = env
        self._agent = None
        # self._agents = []
        # self._num_agents = 1 # I use this for single agent. In multi agent this would depend on the environment
        self._best_ep_rew = float('-inf')
        self._best_avg_rew = float('-inf')

    # Train the agent with the specified number of episodes and maximum steps per episode
    def train(self, num_eps, num_steps=1000, render=False, render_step=10):


        self.collect_validation_data()   #Function for collecting data

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

        return ep_rewards, info

    # Runs a single training episode
    def train_ep(self, num_steps, render):
        raise NotImplementedError

    def collect_validation_data(self, num_eps=50):
        pass

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

    def eval_ep(self, num_steps, render, render_mode):
        raise NotImplementedError

    # Save all of the agents neural nets; For our case it's just one agent
    # def save_agents(self, save_dir, save_type='best'):
    #     for i, agent in enumerate(self._agents):
    #         agent.save_models(save_dir + save_type + '/' + "agent" + str(i), save_type)
    def save_agent(self, save_dir, save_type='best'):
        self._agent.save_models(save_dir + save_type + '/' + "agent", save_type)

    # # Load the given agents neural nets
    # def load_agents(self, save_dir):
    #     for i, agent in enumerate(self._agents):
    #         agent.load_models(save_dir + "agent" + str(i))

    # Load the given agent neural nets
    def load_agent(self, save_dir):
        self._agent.load_models(save_dir + "agent")

    # def dump_agent_attrs(self, save_dir):
    #     for i, agent in enumerate(self._agents):
    #         attr = self._agents[i].__dict__
    #         with open(save_dir + 'agent' + str(i) + '.txt', 'w') as f:
    #             f.write(str(attr))

    def dump_agent_attrs(self, save_dir):
        attr = self._agent.__dict__
        with open(save_dir + 'agent' + '.txt', 'w') as f:
            f.write(str(attr))


# A base class for individual learning agents
class BaseAgent(object):
    def __init__(
            self,
            env
            ):
        self._nS = env.observation_space.shape[0]
        self._obs_space = env.observation_space
        self._act_space = env.action_space


        if isinstance(env.action_space, gym.spaces.Box):
            print("Action space is Continuous")
            # For continuous action spaces (Box)
            self._nA = env.action_space.shape[0]  # For Box action spaces, this gives the number of dimensions
            self._max = env.action_space.high[0]  # Max value of the action space
            self._min = env.action_space.low[0]   # Min value of the action space
        elif isinstance(env.action_space, gym.spaces.Discrete):
            print("Action space is Discrete")
            # For discrete action spaces
            self._nA = env.action_space.n  # Number of discrete actions
            self._max = env.action_space.n - 1  # Max action index (0 to n-1)
            self._min = 0  # Min action index (always 0 for Discrete spaces)


    def sample_action(self, state):
        raise NotImplementedError

    def noiseless_sample_action(self, state):
        raise NotImplementedError

    def train_step(self, state, action, reward, next_state, done):
        raise NotImplementedError

    def save_models(self, save_dir, save_type):
        raise NotImplementedError

    def load_models(self, save_dir):
        raise NotImplementedError

    def set_best(self):
        raise NotImplementedError

    def set_best_avg(self):
        raise NotImplementedError


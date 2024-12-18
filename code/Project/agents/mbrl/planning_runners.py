# Adopted from https://keras.io/examples/rl/ddpg_pendulum/
from base import *
from agents.mbrl.model_runners import modelRunner

#Base runner implements train, eval, load, save_agent, load_agent, dump_agent_attrs
#modelRunner implements train_agent
class planningRunner(modelRunner):
    def __init__(self,
            env,
            agent,
            collect_random_exp=False,
            *agent_args,
            ):
        # super(planningRunner, self).__init__(env, agent, agent_args)



        self._env = env
        self._best_ep_rew = float('-inf')
        self._best_avg_rew = float('-inf')
        self._agent = agent(env)
        self._collect_random_exp = collect_random_exp

    # Train the agent with the specified number of episodes and maximum steps per episode

    def train_ep(self, num_steps, render):
        raise NotImplementedError 

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

    def _train_agent(self, observations, acts, rew, next_observations, done):
        model_hist = []
        obs = observations
        act = acts
        next_obs = next_observations
        model_hist.append(self._agent.train_step(obs, act, rew, next_obs, done))
        return model_hist





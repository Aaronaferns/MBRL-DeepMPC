# Adopted from https://keras.io/examples/rl/ddpg_pendulum/
from base import *
from agents.mbrl.planning_runners import *
from agents.mbrl.dvmpc import *
# from agents.mbrl.her_dvpmc import *

class dvpmcBaseRunner(planningRunner):
    def __init__(self,
            env,
            agent,
            collect_random_exp=True,
            *agent_args,
            ):
        super(planningRunner, self).__init__(env, agent, agent_args)
        self._collect_random_exp = collect_random_exp
        self.count=0

    def train_ep(self, num_steps, render):
        self.count+=1
        print(f"Train ep: {self.count}")
        if self._collect_random_exp:
            self.collect_experiences()
        prev_obs = self._env.reset()[0]
        for i in range(len(self._agents)):
            self._agents[i].reset()

        tot_rew = 0.0
        info_list = []

        for i in range(num_steps):
            if render:
                self._env.render()
            acts = self._sample_actions(prev_obs)
            obs, rew, done,truncated, info= self._env.step(acts)
            tot_rew += rew
            

            dyn_model_hist, val_model_hist  = self._train_agents(prev_obs, acts, rew, obs, done or truncated)
            if dyn_model_hist is not None:
                info_list.append({'info':info, 'step':i+1, 'value_loss':val_model_hist, 'dyn_model_history':dyn_model_hist})
            elif val_model_hist is not None:
                info_list.append({'info':info, 'step':i+1, 'value_loss':val_model_hist})

            if done or truncated:
                break

            prev_obs = obs

        return tot_rew, info_list 

    def _train_agents(self, observations, acts, rew, next_observations, done):
        model_hists = [[]]*len(self._agents)
        v_model_hists = [[]]*len(self._agents)
        for i in range(len(self._agents)):
            obs = observations[i]
            act = acts[i]
            next_obs = next_observations[i]
            model_hist, v_model_loss = self._agents[i].train_step(obs, act, rew, next_obs, done)
            model_hists[i].append(model_hist)
            v_model_hists[i].append(v_model_loss)
        return model_hists, v_model_hists

class DVPMCRunner(dvpmcBaseRunner):
    def __init__(
            self,
            env,
            *agent_args
            ):
        super(DVPMCRunner, self).__init__(env, DVPMCAgent)

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

    def _noiseless_sample_actions(self, obs):
        actions = []
        pred_states = []
        times = []
        for i in range(len(obs)):
            acts, states, time = self._agents[i].noiseless_sample_action(obs[i])
            actions.append(acts)
            pred_states.append(states)
            times.append(time)

        return actions, pred_states, times

# class hindsightDVPMCRunner(dvpmcBaseRunner):
#     def __init__(
#             self,
#             env,
#             *agent_args
#             ):
#         super(hindsightDVPMCRunner, self).__init__(env, hindsightDVPMCAgent)

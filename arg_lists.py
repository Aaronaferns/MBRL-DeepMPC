from agents.mbrl.dvmpc_runners import *

agent_list = [
        # 'DDPG',
        'DVPMC',
        # 'her-DVPMC'
        ]


agent_dict = {
        # 'DDPG': DDPGRunner,
        'DVPMC': DVPMCRunner,
        # 'her-DVPMC': hindsightDVPMCRunner
        }


# A list of valid environments
env_list = [
        'cartpole',
        'cartpole-v2',
        'pendulum'
        ]

# Dictionary mapping environment names for args to environments
env_dict = {
        'cartpole':'CartPole-v0',
        'cartpole-v2':'CartPole-v1',
        'pendulum':'InvertedPendulum-v5'
        }

import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(rewards, save_dir='', show=True, y_ax_txt='Rewards'):
    avg = np.convolve(np.squeeze(rewards), np.ones(10), 'valid') / 10

    fig = plt.figure()
    plt.plot(np.squeeze(rewards), label="Episodic Reward")
    plt.plot(np.squeeze(avg), label="Average Reward")
    plt.xlabel('Episode')
    plt.ylabel(y_ax_txt)
    plt.legend()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'rew_plot')#, format='eps')

def plot_ep_length(info, save_dir='', show=True):
    ep_length = []
    curr_ep_length = 0
    for inf in info:
        step_length = inf['step']
        if step_length > curr_ep_length:
            curr_ep_length = step_length
        else:
            ep_length.append(curr_ep_length)
            curr_ep_length = 0

    fig = plt.figure()
    plt.plot(ep_length)
    plt.xlabel('Episode')
    plt.ylabel('Episodic Timesteps')
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'ep_length_plot')#.eps', format='eps')

def plot_win_info(info, eps, save_dir='', show=True):
    num_wins = 0
    for inf in info:
        num_wins += inf['win']

    fig = plt.figure()
    plt.bar(['win','loss'], [num_wins, eps-num_wins])
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'rew_plot')#.eps', format='eps')

def plot_qvi_model_hists(info_list, save_dir='', show=True):
    model_hist = []
    value_hist = []
    valid_hists = []
    for i in range(len(info_list)):
        info_mod = info_list[i]['dyn_model_history']
        info_val = info_list[i]['value_loss']
        # TODO: Find a better way to handle this for multi-agent case
        val = np.squeeze(info_val).tolist()
        if val is not None:
            value_hist.append(val)
        hist = np.squeeze(info_mod).tolist()
        if hist is not None:
            model_hist.append(hist)

    fig = plt.figure()
    plt.plot(np.squeeze(model_hist), label='Model Loss')

    plt.xlabel('Training Epoch')
    plt.ylabel('Prediction Mean Squared Error')

    plt.legend()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'model_plot')#.eps', format='eps')

    fig = plt.figure()
    plt.plot(np.squeeze(value_hist), label='Value Appr. Loss')

    plt.xlabel('Training Step')
    plt.ylabel('TD(0) Error')
    plt.legend()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'value_model_plot')#.eps', format='eps')


def plot_vi_model_hists(info_list, save_dir='', show=True):
    model_hist = []
    value_hist = []
    valid_hists = []
    for i in range(len(info_list)):
        info_mod = info_list[i]['dyn_model_history']
        info_val = info_list[i]['value_loss']
        # TODO: Find a better way to handle this for multi-agent case
        value_hist.extend(info_val)
        hist = np.squeeze(info_mod).tolist()
        if hist is not None:
            losses = hist.history['loss']
            model_hist.extend(losses)
            if 'val_loss' in hist.history.keys():
                valid_hists.extend(hist.history['val_loss'])

    fig = plt.figure()
    plt.plot(np.squeeze(model_hist), label='Model Loss')
    if len(valid_hists):
        plt.plot(np.squeeze(valid_hists), label='Model Validation Loss')

    plt.xlabel('Training Epoch')
    plt.ylabel('Prediction Mean Squared Error')

    plt.legend()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'model_plot')#.eps', format='eps')

    fig = plt.figure()
    plt.plot(np.squeeze(value_hist), label='Value Appr. Loss')

    plt.xlabel('Training Step')
    plt.ylabel('TD(0) Error')
    plt.legend()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'value_model_plot')#.eps', format='eps')


def plot_en_model_hists(info_list, save_dir='', show=True):
    num_en = 5#len(info_list[]['model_history'])
    all_hist = [[] for i in range(num_en)]
    val_hists = [[] for i in range(num_en)]
    for i in range(len(info_list)):
        info = info_list[i]['model_history']
        # TODO: Find a better way to handle this for multi-agent case
        hist = np.squeeze(info).tolist()
        if hist is not None:
            for i in range(len(hist)):
                h = np.squeeze(hist[i]).tolist()
                losses = h.history['loss']
                all_hist[i].extend(losses)
                if 'val_loss' in h.history.keys():
                    val_hists[i].extend(h.history['val_loss'])

    fig = plt.figure()
    for i in range(num_en):
        plt.plot(np.squeeze(all_hist[i]), label='Model {} Loss'.format(i))
        if len(val_hists[i]):
            plt.plot(np.squeeze(val_hists[i]), label='Model {} Validation Loss'.format(i))

    plt.legend()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'model_plot.png')

def plot_ma_vi_model_hists(info_list, save_dir='', show=True):
    model_hist = [[]]*len(info_list[0]['dyn_model_history'])
    value_hist = [[]]*len(info_list[0]['value_loss'])
    valid_hists = [[]]*len(info_list[0]['dyn_model_history'])
    for i in range(len(info_list)):
        info_mod = info_list[i]['dyn_model_history']
        info_val = info_list[i]['value_loss']
        # TODO: Find a better way to handle this for multi-agent case
        for j in range(len(info_mod)):
            value_hist[j].append(info_val[j])
            hist = np.squeeze(info_mod[j]).tolist()
            if hist is not None:
                losses = hist.history['loss']
                model_hist[j].extend(losses)
                if 'val_loss' in hist.history.keys():
                    valid_hists[j].extend(hist.history['val_loss'])

    for i in range(len(info_list[0]['dyn_model_history'])):
        fig = plt.figure()
        plt.plot(np.squeeze(model_hist[i]), label='Model Loss')
        if len(valid_hists):
            plt.plot(np.squeeze(valid_hists[i]), label='Model Validation Loss')
 
        plt.xlabel('Training Epoch')
        plt.ylabel('Prediction Mean Squared Error')
        plt.title('Agent {}'.format(i))
 
        plt.legend()
        if show:
            plt.show()
        if save_dir:
            fig.savefig(save_dir + 'model_plot')#.eps', format='eps')
 
        fig = plt.figure()
        plt.plot(np.squeeze(value_hist[i]), label='Value Appr. Loss')
 
        plt.xlabel('Training Step')
        plt.ylabel('TD(0) Error')
        plt.title('Agent {}'.format(i))
        plt.legend()
        if show:
            plt.show()
        if save_dir:
            fig.savefig(save_dir + 'value_model_plot')#.eps', format='eps')

def plot_model_hists(info_list, save_dir='', show=True):
    all_hist = []
    val_hists = []
    for i in range(len(info_list)):
        info = info_list[i]['model_history']
        # TODO: Find a better way to handle this for multi-agent case
        hist = np.squeeze(info).tolist()
        if hist is not None:
            losses = hist.history['loss']
            all_hist.extend(losses)
            if 'val_loss' in hist.history.keys():
                val_hists.extend(hist.history['val_loss'])

    fig = plt.figure()
    plt.plot(np.squeeze(all_hist), label='Model Loss')
    plt.xlabel('Training Epoch')
    plt.ylabel('Prediction Mean Squared Error')
    if len(val_hists):
        plt.plot(np.squeeze(val_hists), label='Model Validation Loss')

    plt.legend()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'model_plot.png')

def plot_1v1_ra_rollout(true_traj, pred_traj, save_dir='', show=True):
    fig = plt.figure()
    true_traj = np.squeeze(true_traj)
    pred_traj = np.squeeze(pred_traj)
    true_x = true_traj[:,0]
    pred_x = pred_traj[:,0]
    true_y = true_traj[:,1]
    pred_y = pred_traj[:,1]
    true_cos_h = true_traj[:,2]
    pred_cos_h = pred_traj[:,2]
    true_sin_h = true_traj[:,3]
    pred_sin_h = pred_traj[:,3]

    true_opp_x = true_traj[:,4]
    pred_opp_x = pred_traj[:,4]
    true_opp_y = true_traj[:,5]
    pred_opp_y = pred_traj[:,5]
    true_opp_cos_h = true_traj[:,6]
    pred_opp_cos_h = pred_traj[:,6]
    true_opp_sin_h = true_traj[:,7]
    pred_opp_sin_h = pred_traj[:,7]

    true_goal_x = true_traj[:,-2]
    pred_goal_x = pred_traj[:,-2]
    true_goal_y = true_traj[:,-1]
    pred_goal_y = pred_traj[:,-1]

    fig, axs = plt.subplots(4)
    axs[0].set_ylabel('x [m]')
    axs[0].plot(true_x, label='True', color='blue')
    axs[0].plot(pred_x, label='Predicted', color='blue', linestyle='dashed')
    axs[0].grid()

    axs[1].set_ylabel(r'y [m]')
    axs[1].plot(true_y, label=r'True', color='blue')
    axs[1].plot(pred_y, label=r'Predicted', color='blue', linestyle='dashed')
    axs[1].grid()

    #axs[2].set_xlabel('Time Step [0.02 s]')
    axs[2].set_ylabel(r'cos($\theta$)')
    axs[2].plot(true_cos_h, label=r'True', color='blue')
    axs[2].plot(pred_cos_h, label=r'Predicted', color='blue', linestyle='dashed')
    #axs[2].legend()
    axs[2].grid()

    axs[3].set_ylabel(r'sin($\theta$)')
    axs[3].plot(true_sin_h, label=r'True', color='blue')
    axs[3].plot(pred_sin_h, label=r'Predicted', color='blue', linestyle='dashed')
    #axs2].legend()
    axs[3].grid()

    fig.tight_layout()
    if show:
        plt.show()
    plt.close()

    fig, axs = plt.subplots(4)
    axs[0].set_ylabel('Opponent x [m]')
    axs[0].plot(true_opp_x, label='True', color='blue')
    axs[0].plot(pred_opp_x, label='Predicted', color='blue', linestyle='dashed')
    axs[0].grid()

    axs[1].set_ylabel(r'Opponent y [m]')
    axs[1].plot(true_opp_y, label=r'True', color='blue')
    axs[1].plot(pred_opp_y, label=r'Predicted', color='blue', linestyle='dashed')
    axs[1].grid()

    axs[2].set_ylabel(r'Opponent cos($\theta$)')
    axs[2].plot(true_opp_cos_h, label=r'True', color='blue')
    axs[2].plot(pred_opp_cos_h, label=r'Predicted', color='blue', linestyle='dashed')
    axs[2].grid()

    axs[3].set_ylabel(r'Opponent sin($\theta$)')
    axs[3].plot(true_opp_sin_h, label=r'True', color='blue')
    axs[3].plot(pred_opp_sin_h, label=r'Predicted', color='blue', linestyle='dashed')
    axs[3].grid()

    fig.tight_layout()
    if show:
        plt.show()
    plt.close()

    fig, axs = plt.subplots(2)
    axs[0].set_ylabel('Goal x [m]')
    axs[0].plot(true_goal_x, label='True', color='blue')
    axs[0].plot(pred_goal_x, label='Predicted', color='blue', linestyle='dashed')
    axs[0].grid()

    axs[1].set_ylabel(r'Goal y [m]')
    axs[1].plot(true_goal_y, label=r'True', color='blue')
    axs[1].plot(pred_goal_y, label=r'Predicted', color='blue', linestyle='dashed')
    axs[1].grid()
    fig.tight_layout()
    if show:
        plt.show()
    plt.close()


def plot_cartpole_rollout(true_traj, pred_traj, save_dir='', show=True):
    true_traj = np.squeeze(true_traj)
    pred_traj = np.squeeze(pred_traj)
    true_x = true_traj[:,0]
    pred_x = pred_traj[:,0]
    true_x_dot = true_traj[:,1]
    pred_x_dot = pred_traj[:,1]
    true_theta = true_traj[:,2]
    pred_theta = pred_traj[:,2]
    true_theta_dot = true_traj[:,3]
    pred_theta_dot = pred_traj[:,3]

    fig, axs = plt.subplots(4)
    #axs[0].set_xlabel('Time Step [0.02 s]')
    axs[0].set_ylabel('x [m]')
    axs[0].plot(true_x, label='True', color='blue')
    axs[0].plot(pred_x, label='Predicted', color='blue', linestyle='dashed')
    #axs[0].legend()
    axs[0].grid()

    #axs[1].set_xlabel('Time Step [0.02 s]')
    axs[1].set_ylabel(r'$\dot x$ [m/s]')
    axs[1].plot(true_x_dot, label=r'True', color='blue')
    axs[1].plot(pred_x_dot, label=r'Predicted', color='blue', linestyle='dashed')
    #axs[1].legend()
    axs[1].grid()

    #axs[2].set_xlabel('Time Step [0.02 s]')
    axs[2].set_ylabel(r'$\theta$ [rad]')
    axs[2].plot(true_theta, label=r'True', color='blue')
    axs[2].plot(pred_theta, label=r'Predicted', color='blue', linestyle='dashed')
    #axs[2].legend()
    axs[2].grid()

    axs[3].set_xlabel('Time Step [0.02 s]')
    axs[3].set_ylabel(r'$\dot \theta$ [rad/s]')
    axs[3].plot(true_theta_dot, label=r'True', color='blue')
    axs[3].plot(pred_theta_dot, label=r'Predicted', color='blue', linestyle='dashed')
    #axs[3].legend()
    axs[3].grid()
    fig.tight_layout()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'roll_stack.eps')
    plt.close()

    fig, axs = plt.subplots(4)
    axs[0].set_ylabel('x [m]')
    axs[0].plot(pred_x - true_x, color='blue')
    axs[0].grid()

    axs[1].set_ylabel(r'$\dot x$ [m/s]')
    axs[1].plot(pred_x_dot - true_x_dot, color='blue')
    axs[1].grid()

    axs[2].set_ylabel(r'$\theta$ [rad]')
    axs[2].plot(pred_theta - true_theta, color='blue')
    axs[2].grid()

    axs[3].set_xlabel('Time Step [0.02 s]')
    axs[3].set_ylabel(r'$\dot \theta$ [rad/s]')
    axs[3].plot(pred_theta - true_theta_dot, label=r'True', color='blue')
    #axs[3].legend()
    axs[3].grid()
    fig.tight_layout()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'error_stack.png')
    plt.close()

    plt.plot(true_x, label='True x', color='green')
    plt.plot(pred_x, label='Predicted x', color='green', linestyle='dashed')
    plt.plot(true_x_dot, label=r'True $\dot x$', color='red')
    plt.plot(pred_x_dot, label=r'Predicted $\dot x$', color='red', linestyle='dashed')
    plt.plot(true_theta, label=r'True $\theta$', color='blue')
    plt.plot(pred_theta, label=r'Predicted $\theta$', color='blue', linestyle='dashed')
    plt.plot(true_theta_dot, label=r'True $\dot \theta$', color='magenta')
    plt.plot(pred_theta_dot, label=r'Predicted $\dot \theta$', color='magenta', linestyle='dashed')

    plt.legend()
    plt.grid()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'model_plot.png')
    plt.close()

def plot_in_motivation_info(info, save_dir='', show=True):
    pass

def plot_cartpole_run(true_traj, pred_traj, act_traj, vals, mean_vals, next_best_vals, save_dir='', show=True):
    true_traj = np.squeeze(true_traj)
    pred_traj = np.squeeze(pred_traj)
    act_traj = 30*np.squeeze(act_traj)
    true_x = true_traj[:,0]
    true_x_dot = true_traj[:,1]
    true_theta = true_traj[:,2]
    true_theta_dot = true_traj[:,3]
    vals = np.squeeze(vals)
    mean_vals = np.squeeze(mean_vals)
    next_best_vals = np.squeeze(next_best_vals)

    pred_x = pred_traj[:,0]
    pred_x_dot = pred_traj[:,1]
    pred_theta = pred_traj[:,2]
    pred_theta_dot = pred_traj[:,3]

    fig, axs = plt.subplots(5)
    axs[0].set_ylabel('x [m]')
    axs[0].plot(true_x, label='True', color='blue')
    axs[0].grid()

    axs[1].set_ylabel(r'$\dot x$ [m/s]')
    axs[1].plot(true_x_dot, label=r'True', color='blue')
    axs[1].grid()

    axs[2].set_ylabel(r'$\theta$ [rad]')
    axs[2].plot(true_theta, label=r'True', color='blue')
    axs[2].grid()

    axs[3].set_xlabel('time step [0.02 s]')
    axs[3].set_ylabel(r'$\dot \theta$ [rad/s]')
    axs[3].plot(true_theta_dot, label=r'true', color='blue')
    axs[3].grid()


    axs[4].set_xlabel('time step [0.02 s]')
    axs[4].set_ylabel(r'F [N]')
    axs[4].plot(act_traj, label=r'true', color='blue')
    axs[4].grid()
    fig.tight_layout()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'run.eps')
    plt.close()

    fig, axs = plt.subplots(3)

    axs[0].set_xlabel('time step [0.02 s]')
    axs[0].set_ylabel(r'Disc. Rew. Sum.')
    axs[0].plot(vals, label=r'true', color='blue')
    axs[0].grid()

    axs[1].set_xlabel('time step [0.02 s]')
    axs[1].set_ylabel(r'Mean Advantage')
    axs[1].plot(mean_vals, label=r'true', color='blue')
    axs[1].grid()

    axs[2].set_xlabel('time step [0.02 s]')
    axs[2].set_ylabel(r'Next Best Advantage')
    axs[2].plot(next_best_vals, label=r'true', color='blue')
    axs[2].grid()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'vals.png')
    plt.close()

    fig, axs = plt.subplots(4)
    axs[0].set_ylabel('x [m]')
    axs[0].plot(pred_x - true_x, color='blue')
    axs[0].grid()

    axs[1].set_ylabel(r'$\dot x$ [m/s]')
    axs[1].plot(pred_x_dot - true_x_dot, color='blue')
    axs[1].grid()

    axs[2].set_ylabel(r'$\theta$ [rad]')
    axs[2].plot(pred_theta - true_theta, color='blue')
    axs[2].grid()

    axs[3].set_xlabel('Time Step [0.02 s]')
    axs[3].set_ylabel(r'$\dot \theta$ [rad/s]')
    axs[3].plot(pred_theta - true_theta_dot, label=r'True', color='blue')
    #axs[3].legend()
    axs[3].grid()
    fig.tight_layout()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'error_stack.png')
    plt.close()

def plot_lti_run(true_traj, pred_traj, act_traj, vals, mean_vals, next_best_vals, save_dir='', show=True):
    true_traj = np.squeeze(true_traj)
    pred_traj = np.squeeze(pred_traj)
    act_traj = np.squeeze(act_traj)
    true_x1 = true_traj[:,0]
    if true_traj.shape[1]:
        true_x2 = true_traj[:,1]

    if true_traj.shape[1]:
        fig, axs = plt.subplots(2)
    axs[0].set_ylabel('x [m]')
    axs[0].plot(true_x, label='True', color='blue')
    axs[0].grid()

    axs[1].set_ylabel(r'$\dot x$ [m/s]')
    axs[1].plot(true_x_dot, label=r'True', color='blue')
    axs[1].grid()

    axs[2].set_ylabel(r'$\theta$ [rad]')
    axs[2].plot(true_theta, label=r'True', color='blue')
    axs[2].grid()

    axs[3].set_xlabel('time step [0.02 s]')
    axs[3].set_ylabel(r'$\dot \theta$ [rad/s]')
    axs[3].plot(true_theta_dot, label=r'true', color='blue')
    axs[3].grid()
    fig.tight_layout()

    axs[4].set_xlabel('time step [0.02 s]')
    axs[4].set_ylabel(r'F [N]')
    axs[4].plot(act_traj, label=r'true', color='blue')
    axs[4].grid()
    if show:
        plt.show()
    if save_dir:
        fig.savefig(save_dir + 'run.png')
    plt.close()

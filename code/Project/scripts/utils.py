import os, imageio
import numpy as np
from datetime import datetime


def generate_agent_dir(env_name, agent_type, suffix, timestamp_dir=False):
    dir = './agents/saved/' + env_name + '/' + agent_type + '/'
    dir = dir + suffix + '/' if suffix else dir
    dir = os.path.join(dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S/')) if timestamp_dir else dir
    os.makedirs(dir, exist_ok=True)
    return dir

def generate_results_dir(env_name, agent_type, suffix='', timestamp_dir=True):
    dir = './results/' + env_name + '/' + agent_type + '/'
    dir = dir + suffix + '/' if suffix else dir
    dir = os.path.join(dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S/')) if timestamp_dir else dir
    os.makedirs(dir, exist_ok=True)
    return dir

def save_gif(path, imgs):
    with imageio.get_writer(path, mode='I',fps=30) as writer:
        for img in imgs:
            writer.append_data(img)

def save_data(path, data):
    np.save(path, data)

def load_data(path):
    data = np.load(path)
    return data

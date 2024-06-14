import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import time
from stable_baselines3 import A2C, PPO, DQN
from stable_baselines3.common.env_util import make_vec_env

project_base_path = "/Users/computername/desiredOutputPath" #Change this to your desired output path
output_path = os.path.join(project_base_path, "dqn_a2c_ppo")
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
# #to save output to google drive on google collab
# drive.mount('/content/drive')
# project_base_path = "/content/drive/MyDrive/cartpole_project"
# output_path = os.path.join(project_base_path, "dqn_a2c_ppo")
# if not os.path.exists(output_path):
#     os.makedirs(output_path)


# #to save output to local on google collab
# project_base_path = "/content/cartpole_project"
# output_path = os.path.join(project_base_path, "dqn_a2c_ppo")
# if not os.path.exists(output_path):
#     os.makedirs(output_path)

env_id = 'CartPole-v1'
env = gym.make(env_id) #keep to prevent environment from rendering
# env = gym.make(env_id, render_mode = 'rgb_array') #uncomment to render environment

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#DQN Algorithm
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        hidden_layer = F.relu(self.fc1(state))
        hidden_layer = F.relu(self.fc2(hidden_layer))
        return self.fc3(hidden_layer)

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        experience_tuple = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience_tuple)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr = 5e-4)
        self.memory = ReplayBuffer(action_size, buffer_size = 10000, batch_size = 64, seed = seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, gamma = 0.99)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

def dqn(n_episodes = 2000, max_t = 1000, eps_start = 1.0, eps_end = 0.01, eps_decay = 0.995):
    dqn_metrics = []
    scores_window = deque(maxlen = 100)
    epsilon = eps_start
    dqn_convergence_episodes = []
    dqn_start_training_time = time.time()
    dqn_agent = DQNAgent(state_size=4, action_size = 2, seed = 0)

    for episode in range(1, n_episodes + 1):
        state, info = env.reset()
        episode_metrics = 0
        for timestep in range(max_t):
            action = dqn_agent.act(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            dqn_agent.step(state, action, reward, next_state, done)
            state = next_state
            episode_metrics += reward
            if done:
                break
        scores_window.append(episode_metrics)
        dqn_metrics.append(episode_metrics)
        epsilon = max(eps_end, eps_decay * epsilon)
        print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_window)}', end = "")
        if episode % 100 == 0:
            print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_window)}')
        if np.mean(scores_window) >= 200.0:
            print(f'\nEnvironment solved in {episode} episodes!\tAverage Score: {np.mean(scores_window)}')
            dqn_convergence_episodes.append(episode)
            torch.save(dqn_agent.qnetwork_local.state_dict(), os.path.join(output_path, 'dqn_checkpoint.pth'))
            break
        dqn_convergence_episodes.append(episode)

    dqn_total_training_time = time.time() - dqn_start_training_time
    dqn_stability_variance = np.var(dqn_metrics)

    np.savetxt(os.path.join(output_path, 'dqn_metrics.txt'), dqn_metrics, delimiter = ',')
    np.savetxt(os.path.join(output_path, 'dqn_convergence_rate.txt'), dqn_convergence_episodes, delimiter = ',')
    with open(os.path.join(output_path, 'dqn_othermetrics.txt'), 'w') as file:
        file.write(f"Convergence Time: {dqn_total_training_time}\n")
        file.write(f"Stability: {dqn_stability_variance}\n")

    plot_metrics(dqn_metrics, 'dqn_training_metrics.png')
    return dqn_metrics

#A2C Algorithm
def a2c():
    vec_env = make_vec_env(env_id, n_envs = 4)
    model = A2C("MlpPolicy", vec_env, verbose = 1)
    model.learn(total_timesteps = 25000)
    model.save(os.path.join(output_path, "a2c_cartpole"))
    del model
    model = A2C.load(os.path.join(output_path, "a2c_cartpole"))

    obs = vec_env.reset()
    a2c_metrics = []
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        a2c_metrics.append(rewards)
        if all(dones):
            break
    np.savetxt(os.path.join(output_path, 'a2c_metrics.txt'), a2c_metrics, delimiter = ',')

#PPO Algorithm
def ppo():
    vec_env = make_vec_env(env_id, n_envs = 4)
    model = PPO("MlpPolicy", vec_env, verbose = 1)
    model.learn(total_timesteps = 25000)
    model.save(os.path.join(output_path, "ppo_cartpole"))
    del model
    model = PPO.load(os.path.join(output_path, "ppo_cartpole"))

    obs = vec_env.reset()
    ppo_metrics = []
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        ppo_metrics.append(rewards)
        if all(dones):
            break
    np.savetxt(os.path.join(output_path, 'ppo_metrics.txt'), ppo_metrics, delimiter = ',')

def plot_metrics(metrics, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(metrics)), metrics)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    fig.savefig(os.path.join(output_path, filename))
    plt.close(fig)

dqn_metrics = dqn()
a2c()
ppo()

evaluation_comparison_metrics = []
for algorithm in ["dqn", "a2c", "ppo"]:
    score_file = os.path.join(output_path, f"{algorithm}_metrics.txt")
    if not os.path.exists(score_file):
        print(f"{algorithm} fiels not found.")
        continue

    metrics = np.loadtxt(score_file, delimiter = ',')
    average_reward = np.mean(metrics)
    evaluation_comparison_metrics.append({"Algorithms": algorithm.upper(),"Average Reward": average_reward,})

evaluation_dataframe = pd.DataFrame(evaluation_comparison_metrics)
print(evaluation_dataframe)

evaluation_csv_filename = os.path.join(output_path, "cartpole_algorithmsmetrics_evaluation.csv")
evaluation_dataframe.to_csv(evaluation_csv_filename, index = False)


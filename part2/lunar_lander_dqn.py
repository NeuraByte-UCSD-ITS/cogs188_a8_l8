import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import time
import pandas as pd

project_base_path = "/Users/computername/desiredOutputPath" #Change this to your desired output path
output_path = os.path.join(project_base_path, "dqn_qlearning_sarsa")
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
# Define the Lunar Lander environment with render_mode
env = gym.make('LunarLander-v2', render_mode='rgb_array')

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
class QNetwork(nn.Module):
    """Neural Network for approximating Q-values."""
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # TODO: Define the neural network layers
        # You can keep it fairly simple, e.g., with 3 linear layers with 64 units each
        # and use ReLU activations for the hidden layers.
        # However, make sure that the input size of the first layer matches the state size
        # and the output size of the last layer matches the action size
        # This is because the input to the network will be the state and the output will be the Q-values for each action
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        """Build a network that maps state -> action values.
        Params
        ======
            state (torch.Tensor): The state input
        Returns
        =======
            torch.Tensor: The predicted action values
        """
        # TODO: Define the forward pass
        # You're basically just passing the state through the network here (based on the layers you defined in __init__) and returning the output
        hidden_layer = F.relu(self.fc1(state))
        hidden_layer = F.relu(self.fc2(hidden_layer))
        return self.fc3(hidden_layer)        
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): Dimension of each action
            buffer_size (int): Maximum size of buffer
            batch_size (int): Size of each training batch
            seed (int): Random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # TODO: Implement this method
        # Use the namedtuple 'Experience' to create an experience tuple and append it to the memory
        experience_tuple = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience_tuple)        

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        # TODO: Complete this method
        # We first use random.sample to sample self.batch_size experiences from self.memory
        # Convert the sampled experiences to tensors and return them as a tuple
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # Similarly, convert the other components of the experiences to tensors
        # the `actions` tensor should be of type long
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class DQNAgent:
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        # TODO: Initialize Q-networks (local and target)
        # Hints: Use QNetwork to create both qnetwork_local and qnetwork_target, and move them to device
        # Use optim.Adam to create an optimizer for qnetwork_local
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=5e-4)


        # Replay memory
        # TODO: Initialize replay memory
        # Hint: Create a ReplayBuffer object with appropriate parameters (action_size, buffer_size, batch_size, seed)
        self.memory = ReplayBuffer(action_size, buffer_size=10000, batch_size=64, seed=seed)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # TODO: Save the experience in replay memory
        # Hint: Use the add method of ReplayBuffer
        self.memory.add(state, action, reward, next_state, done)
        self.time_step = (self.time_step + 1) % 4


        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, gamma=0.99)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (np.ndarray): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # TODO: Compute and minimize the loss
        # 1. Compute Q targets for current states (s')
        # Hint: Use the target network to get the next action values
        # 2. Compute Q expected for current states (s)
        # Hint: Use the local network to get the current action values
        # 3. Compute the loss between Q expected and Q target
        # 4. Perform a gradient descent step to update the local network
        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # TODO: Update target network
        # Hint: Use the soft_update method provided
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=1e-3)   
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

agent = DQNAgent(state_size=8, action_size=4, seed=0)

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    dqn_metrics = []
    scores_window = deque(maxlen=100)
    epsilon = eps_start
    dqn_convergence_episodes = []
    dqn_start_training_time = time.time()

    dqn_agent = DQNAgent(state_size=8, action_size=4, seed=0)

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
        print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_window)}', end="")
        if episode % 100 == 0:
            print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_window)}')
        if np.mean(scores_window) >= 200.0:
            print(f'\nEnvironment solved in {episode} episodes!\tAverage Score: {np.mean(scores_window)}')
            dqn_convergence_episodes.append(episode)
            torch.save(dqn_agent.qnetwork_local.state_dict(), os.path.join(output_path, 'checkpoint.pth'))
            break
        dqn_convergence_episodes.append(episode)

    dqn_total_training_time = time.time() - dqn_start_training_time
    dqn_stability_variance = np.var(dqn_metrics)

    np.savetxt(os.path.join(output_path, 'dqn_metrics.txt'), dqn_metrics, delimiter=',')
    np.savetxt(os.path.join(output_path, 'dqn_convergence_rate.txt'), dqn_convergence_episodes, delimiter=',')
    with open(os.path.join(output_path, 'dqn_othermetrics.txt'), 'w') as file:
        file.write(f"Convergence Time: {dqn_total_training_time}\n")
        file.write(f"Stability: {dqn_stability_variance}\n")

    plot_metrics(dqn_metrics, 'dqn_training_metrics.png')
    # create_video(dqn_agent, env, "lunar_lander.mp4")
    return dqn_metrics

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros(state_size + (action_size,))
        self.convergence_episodes = []

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        temporal_difference_target = reward + self.gamma * np.max(self.q_table[next_state])
        temporal_difference_error = temporal_difference_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * temporal_difference_error

def discretize_state(state):
    bins = 10
    discretized_state = tuple(np.clip(((state - env.observation_space.low) / (env.observation_space.high - env.observation_space.low) * (bins - 1)).astype(int), 0, bins-1))
    return discretized_state

def q_learning(num_episodes=2000, max_timesteps=1000):
    q_learning_agent = QLearningAgent(state_size=(10,) * env.observation_space.shape[0], action_size=env.action_space.n)
    q_learning_metrics = []
    scores_window = deque(maxlen=100)
    q_learning_start_training_time = time.time()

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        state = discretize_state(state)
        episode_metrics = 0
        for timestep in range(max_timesteps):
            action = q_learning_agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = discretize_state(next_state)
            q_learning_agent.learn(state, action, reward, next_state)
            state = next_state
            episode_metrics += reward
            if done:
                break
        scores_window.append(episode_metrics)
        q_learning_metrics.append(episode_metrics)
        print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_window)}', end="")
        if episode % 100 == 0:
            print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_window)}')
        q_learning_agent.convergence_episodes.append(episode)

    q_learning_total_training_time = time.time() - q_learning_start_training_time
    q_learning_stability_variance = np.var(q_learning_metrics)

    np.savetxt(os.path.join(output_path, 'q_learning_metrics.txt'), q_learning_metrics, delimiter=',')
    np.savetxt(os.path.join(output_path, 'q_learning_convergence_rate.txt'), q_learning_agent.convergence_episodes, delimiter=',')
    with open(os.path.join(output_path, 'q_learning_othermetrics.txt'), 'w') as file:
        file.write(f"Convergence Time: {q_learning_total_training_time}\n")
        file.write(f"Stability: {q_learning_stability_variance}\n")

    plot_metrics(q_learning_metrics, 'q_learning_training_metrics.png')
    return q_learning_metrics

class SARSAAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros(state_size + (action_size,))
        self.convergence_episodes = []

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, next_action):
        temporal_difference_target = reward + self.gamma * self.q_table[next_state][next_action]
        temporal_difference_error = temporal_difference_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * temporal_difference_error

def sarsa(num_episodes=2000, max_timesteps=1000):
    sarsa_agent = SARSAAgent(state_size=(10,) * env.observation_space.shape[0], action_size=env.action_space.n)
    sarsa_metrics = []
    scores_window = deque(maxlen=100)
    sarsa_start_training_time = time.time()

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        state = discretize_state(state)
        action = sarsa_agent.choose_action(state)
        episode_metrics = 0
        for timestep in range(max_timesteps):
            next_state, reward, done, truncated, info = env.step(action)
            next_state = discretize_state(next_state)
            next_action = sarsa_agent.choose_action(next_state)
            sarsa_agent.learn(state, action, reward, next_state, next_action)
            state = next_state
            action = next_action
            episode_metrics += reward
            if done:
                break
        scores_window.append(episode_metrics)
        sarsa_metrics.append(episode_metrics)
        print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_window)}', end="")
        if episode % 100 == 0:
            print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_window)}')
        sarsa_agent.convergence_episodes.append(episode)

    sarsa_total_training_time = time.time() - sarsa_start_training_time
    sarsa_stability_variance = np.var(sarsa_metrics)

    np.savetxt(os.path.join(output_path, 'sarsa_metrics.txt'), sarsa_metrics, delimiter=',')
    np.savetxt(os.path.join(output_path, 'sarsa_convergence_rate.txt'), sarsa_agent.convergence_episodes, delimiter=',')
    with open(os.path.join(output_path, 'sarsa_othermetrics.txt'), 'w') as file:
        file.write(f"Convergence Time: {sarsa_total_training_time}\n")
        file.write(f"Stability: {sarsa_stability_variance}\n")

    plot_metrics(sarsa_metrics, 'sarsa_training_metrics.png')
    return sarsa_metrics

dqn_metrics = dqn()
q_learning_metrics = q_learning()
sarsa_metrics = sarsa()

evaluation_comparison_metrics = []
for algorithm in ["dqn", "q_learning", "sarsa"]:
    score_file = os.path.join(output_path, f"{algorithm}_metrics.txt")
    convergence_file = os.path.join(output_path, f"{algorithm}_convergence_rate.txt")
    othermetrics_file = os.path.join(output_path, f"{algorithm}_othermetrics.txt")

    if not os.path.exists(score_file) or not os.path.exists(convergence_file) or not os.path.exists(othermetrics_file):
        print(f"Files for {algorithm} not found.")
        continue

    metrics = np.loadtxt(score_file, delimiter=',')
    convergence_rate = np.loadtxt(convergence_file, delimiter=',')

    average_reward = np.mean(metrics)
    convergence_episode = convergence_rate[-1]

    with open(othermetrics_file, 'r') as file:
        othermetrics_lines = file.readlines()
        total_training_time = float(othermetrics_lines[0].split(': ')[1])
        stability_variance = float(othermetrics_lines[1].split(': ')[1])

    evaluation_comparison_metrics.append({"Algorithms": algorithm.upper(),
                                          "Average Reward": average_reward,
                                          "Convergence Rate": convergence_episode,
                                          "Stability": stability_variance,
                                          "Computation Time": total_training_time})

evaluation_dataframe = pd.DataFrame(evaluation_comparison_metrics)
print(evaluation_dataframe)

evaluation_csv_filename = os.path.join(output_path, "algorithmsmetrics_evalution.csv")
evaluation_dataframe.to_csv(evaluation_csv_filename, index=False)

def plot_metrics(metrics, filename):
    """Plot the scores."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(metrics)), metrics)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    fig.savefig(os.path.join(output_path, filename))
    plt.close(fig)
    
def create_video(agent, env, filename="lunar_lander.mp4"):
    """Create a video of the agent's performance."""
    video_env = gym.wrappers.RecordVideo(env, './video', episode_trigger=lambda x: True, video_length=0)
    state, _ = video_env.reset()
    done = False
    while not done:
        action = agent.act(state, eps=0.0)  # Use greedy policy for evaluation
        state, reward, done, _, _ = video_env.step(action)
    video_env.close()

    # Rename the video file to the desired filename
    video_dir = './video'
    video_file = [f for f in os.listdir(video_dir) if f.endswith('.mp4')][0]
    os.rename(os.path.join(video_dir, video_file), filename)

#Loading trained model & creating video
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
create_video(agent, env)

import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    return np.loadtxt(filename, delimiter = ',')

def plot_comparison(dqn_scores, a2c_scores, ppo_scores):
    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(len(dqn_scores)), dqn_scores, label = 'DQN')
    plt.plot(np.arange(len(a2c_scores)), a2c_scores, label = 'A2C')
    plt.plot(np.arange(len(ppo_scores)), ppo_scores, label = 'PPO')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.title('Algorithm Performance Comparison')
    plt.legend()
    plt.show()
    plt.savefig('algorithm_performance_comparison.png')

dqn_scores = load_data('dqn_scores.txt')
a2c_scores = load_data('a2c_scores.txt')
ppo_scores = load_data('ppo_scores.txt')

plot_comparison(dqn_scores, a2c_scores, ppo_scores)

def plot_convergence(dqn_convergence, a2c_convergence, ppo_convergence):
    plt.figure(figsize=(8, 8))
    plt.plot(np.arange(len(dqn_convergence)), dqn_convergence, label = 'DQN')
    plt.plot(np.arange(len(a2c_convergence)), a2c_convergence, label = 'A2C')
    plt.plot(np.arange(len(ppo_convergence)), ppo_convergence, label = 'PPO')
    plt.xlabel('Episodes')
    plt.ylabel('Convergence Rate')
    plt.title('Convergence Comparisons')
    plt.legend()
    plt.show()
    plt.savefig('algorithm_convergence_comparison.png')

dqn_convergence = load_data('dqn_convergence_rate.txt')
a2c_convergence = load_data('a2c_convergence_rate.txt')
ppo_convergence = load_data('ppo_convergence_rate.txt')

plot_convergence(dqn_convergence, a2c_convergence, ppo_convergence)


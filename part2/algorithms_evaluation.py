# +
import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    return np.loadtxt(filename, delimiter = ',')

def plot_comparison(dqn_scores, sarsa_scores, q_learning_scores):
    plt.figure(figsize = (8, 8))
    plt.plot(np.arange(len(dqn_scores)), dqn_scores, label = 'DQN')
    plt.plot(np.arange(len(sarsa_scores)), sarsa_scores, label = 'SARSA')
    plt.plot(np.arange(len(q_learning_scores)), q_learning_scores, label = 'Q-learning')
    plt.xlabel('Episodes')
    plt.ylabel('Scores')
    plt.title('Algorithm Performance Comparison')
    plt.legend()
    plt.show()
    plt.savefig('algorithm_performance_comparison.png')

dqn_scores = load_data('dqn_scores.txt')
sarsa_scores = load_data('sarsa_scores.txt')
q_learning_scores = load_data('q_learning_scores.txt')

plot_comparison(dqn_scores, sarsa_scores, q_learning_scores)

def plot_convergence(dqn_convergence, sarsa_convergence, q_learning_convergence):
    plt.figure(figsize = (8, 8))
    plt.plot(np.arange(len(dqn_convergence)), dqn_convergence, label = 'DQN')
    plt.plot(np.arange(len(sarsa_convergence)), sarsa_convergence, label = 'SARSA')
    plt.plot(np.arange(len(q_learning_convergence)), q_learning_convergence, label = 'Q-learning')
    plt.xlabel('Episodes')
    plt.ylabel('Convergence Rate')
    plt.title('Algorithm Convergence Comparison')
    plt.legend()
    plt.show()
    plt.savefig('algorithm_convergence_comparison.png')

dqn_convergence = load_data('dqn_convergence_rate.txt')
sarsa_convergence = load_data('sarsa_convergence_rate.txt')
q_learning_convergence = load_data('q_learning_convergence_rate.txt')

plot_convergence(dqn_convergence, sarsa_convergence, q_learning_convergence)


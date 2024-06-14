import pygame
import numpy as np
import random
import argparse
from tqdm import tqdm
import pickle

# Add a required argument that must be either "forward" or "backward"
parser = argparse.ArgumentParser()
parser.add_argument("algorithm", help="Algorithm to solve the maze", choices=["forward", "backward"])
args = parser.parse_args()

# Check the value of the argument
if args.algorithm == "forward":
    print("Using Forward Eligibility Traces")
elif args.algorithm == "backward":
    print("Using Backward Eligibility Traces")
else:
    print(f"Invalid algorithm. Please run the script using 'python eligibility_traces_maze.py <algorithm>' where <algorithm> is either 'forward' or 'backward'")

# Constants
SCREEN_SIZE = 600
GRID_SIZE = 25
BLOCK_SIZE = SCREEN_SIZE // GRID_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

actions = ['up', 'down', 'left', 'right']

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Maze Solver")

# Maze representation (0 = open, 1 = wall)
maze = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.8, 0.2])
maze[0][0] = 0  # Start
maze[GRID_SIZE-1][GRID_SIZE-1] = 0  # Goal

# Function to draw the maze
def draw_maze(maze):
    """Draws the maze on the screen."""
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(screen, WHITE if maze[y][x] == 0 else BLACK, rect)
    start_rect = pygame.Rect(0, 0, BLOCK_SIZE, BLOCK_SIZE)
    goal_rect = pygame.Rect((GRID_SIZE-1) * BLOCK_SIZE, (GRID_SIZE-1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(screen, GREEN, start_rect)
    pygame.draw.rect(screen, RED, goal_rect)

# Function to draw the solution path
def draw_path(path):
    """Draws the solution path on the screen."""
    for position in path:
        rect = pygame.Rect(position[1] * BLOCK_SIZE, position[0] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(screen, BLUE, rect)

    start_rect = pygame.Rect(0, 0, BLOCK_SIZE, BLOCK_SIZE)
    goal_rect = pygame.Rect((GRID_SIZE-1) * BLOCK_SIZE, (GRID_SIZE-1) * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(screen, GREEN, start_rect)
    pygame.draw.rect(screen, RED, goal_rect)

# Helper functions for the eligibility traces algorithms
def choose_action(state, q_values, epsilon=0.1):
    """Chooses an action based on epsilon-greedy policy."""
    # TODO: Implement the epsilon-greedy policy to choose an action from the action space
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        return actions[np.argmax(q_values[state])]

def get_next_state(state, action):
    """Returns the next state given the current state and action."""
    if action == 'up' and state[0] > 0:
        return (state[0] - 1, state[1])
    elif action == 'down' and state[0] < GRID_SIZE - 1:
        return (state[0] + 1, state[1])
    elif action == 'left' and state[1] > 0:
        return (state[0], state[1] - 1)
    elif action == 'right' and state[1] < GRID_SIZE - 1:
        return (state[0], state[1] + 1)
    return state

def get_reward(state):
    """Returns the reward for reaching the given state."""
    # TODO: Implement the reward function for the maze
    return 1.0 if state == (GRID_SIZE - 1, GRID_SIZE - 1) else -0.05

# Forward Eligibility Traces
class ForwardEligibilityTraces:
    def __init__(self, maze, start, end, actions, alpha=0.1, gamma=0.9, lambda_=0.8):
        self.maze = np.array(maze)
        self.start = start
        self.end = end
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.q_values = np.zeros((*self.maze.shape, len(actions)))
        self.eligibility_traces = np.zeros_like(self.q_values)

    def choose_action(self, state):
        """Chooses an action based on epsilon-greedy policy."""
        return choose_action(state, self.q_values)

    def update(self, state, action, reward, next_state, next_action):
        """
        Updates the Q-values and eligibility traces.
        TODO: Implement the update rule for forward eligibility traces.
        """
        action_index = self.actions.index(action)
        next_action_index = self.actions.index(next_action)
        # TODO: Compute the temporal difference error (delta)
        # TODO: Update the eligibility trace for the current state-action pair
        # TODO: Update all Q-values and eligibility traces
        delta = reward + self.gamma * self.q_values[next_state][next_action_index] - self.q_values[state][action_index]
        self.eligibility_traces[state][action_index] += 1
        self.q_values += self.alpha * delta * self.eligibility_traces
        self.eligibility_traces *= self.gamma * self.lambda_    
        
    def train(self, episodes=100):
        """Trains the agent using forward eligibility traces."""
        # TODO: Implement the training loop for forward eligibility traces
        # You may use tqdm to display a progress bar for the training loop
        # This function doesn't return anything, but it should update the Q-values
        for episode_index in tqdm(range(episodes)):
            state = self.start
            self.eligibility_traces = np.zeros_like(self.q_values)  
            action = self.choose_action(state)
            
            while state != self.end:
                next_state = get_next_state(state, action)
                reward = get_reward(next_state)
                next_action = self.choose_action(next_state)
                self.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action      
    
    #pickle - save and load
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.q_values, self.eligibility_traces), f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.q_values, self.eligibility_traces = pickle.load(f)

# Backward Eligibility Traces
class BackwardEligibilityTraces:
    def __init__(self, maze, start, end, actions, alpha=0.1, gamma=0.9, lambda_=0.8):
        self.maze = np.array(maze)
        self.start = start
        self.end = end
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.q_values = np.zeros((*self.maze.shape, len(actions)))
        self.eligibility_traces = np.zeros_like(self.q_values)

    def choose_action(self, state):
        """Chooses an action based on epsilon-greedy policy."""
        return choose_action(state, self.q_values)

    def update(self, state, action, reward, next_state):
        """
        Updates the Q-values and eligibility traces.
        TODO: Implement the update rule for backward eligibility traces.
        """
        action_index = self.actions.index(action)
        # TODO: Compute the temporal difference error (delta)
        # TODO: Update the eligibility trace for the current state-action pair
        # TODO: Update all Q-values and eligibility traces
        delta = reward + self.gamma * np.max(self.q_values[next_state]) - self.q_values[state][action_index]
        self.eligibility_traces[state][action_index] += 1
        self.q_values += self.alpha * delta * self.eligibility_traces
        self.eligibility_traces *= self.gamma * self.lambda_  
        
    def train(self, episodes=100):
        """Trains the agent using backward eligibility traces."""
        # TODO: Implement the training loop for backward eligibility traces
        # You may use tqdm to display a progress bar for the training loop
        # This function doesn't return anything, but it should update the Q-values
        for episode_index in tqdm(range(episodes)):
            state = self.start
            self.eligibility_traces = np.zeros_like(self.q_values)  
            while state != self.end:
                action = self.choose_action(state)
                next_state = get_next_state(state, action)
                reward = get_reward(next_state)
                self.update(state, action, reward, next_state)
                state = next_state
                
    #pickle - save and load
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump((self.q_values, self.eligibility_traces), f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.q_values, self.eligibility_traces = pickle.load(f)   

# Solve the maze using the selected algorithm
def solve_maze(maze, algorithm):
    """Solves the maze using the selected eligibility traces algorithm."""
    start = (0, 0)
    goal = (GRID_SIZE - 1, GRID_SIZE - 1)
    if algorithm == "forward":
        agent = ForwardEligibilityTraces(maze, start, goal, actions)
    elif algorithm == "backward":
        agent = BackwardEligibilityTraces(maze, start, goal, actions)
    agent.train(episodes=100)
    # Extract the path from Q-values
    state = start
    path = [state]
    while state != goal:
        action = agent.choose_action(state)
        state = get_next_state(state, action)
        path.append(state)
    return path, agent

# Main loop
running = True
maze_solved = False

if __name__ == "__main__":
    
    #pickle - load existing model or generate model otherwise
    start = (0, 0)
    goal = (GRID_SIZE - 1, GRID_SIZE - 1)

    if args.algorithm == "forward":
        agent = ForwardEligibilityTraces(maze, start, goal, actions)
        try:
            agent.load_model("forward_trained_model.pkl")
        except FileNotFoundError:
            print("Generating new model because one does not exist")
    elif args.algorithm == "backward":
        agent = BackwardEligibilityTraces(maze, start, goal, actions)
        try:
            agent.load_model("backward_trained_model.pkl")
        except FileNotFoundError:
            print("Generating new model because one does not exist")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)
        draw_maze(maze)

        # Solve the maze and draw the solution path
        if not maze_solved:
            path, agent = solve_maze(maze, args.algorithm)
            if path:
                maze_solved = True

        if maze_solved:
            draw_path(path)

        new_map_rect = pygame.Rect(SCREEN_SIZE - 150, 0, 150, 50)
        pygame.draw.rect(screen, RED, new_map_rect)
        font = pygame.font.Font(None, 36)
        text = font.render("New Map", True, WHITE)
        text_rect = text.get_rect(center=new_map_rect.center)
        screen.blit(text, text_rect)

        # Check if the user clicked the new_map button
        if event.type == pygame.MOUSEBUTTONDOWN:
            if new_map_rect.collidepoint(event.pos):
                maze_solved = False
                maze = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.8, 0.2])
                maze[0][0] = 0  # Start
                maze[GRID_SIZE-1][GRID_SIZE-1] = 0  # Goal

        pygame.display.flip()

    #pickle - saving model
    if args.algorithm == "forward":
        agent.save_model("forward_trained_model.pkl")
    elif args.algorithm == "backward":
        agent.save_model("backward_trained_model.pkl")

    pygame.quit()

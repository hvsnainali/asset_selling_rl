import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from models.DQNetwork import DQNetwork

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, learning_rate=0.001, memory_size=2000):
        """
        Initialize the DQN agent.

        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Number of possible actions.
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial exploration rate.
            epsilon_min (float): Minimum exploration rate.
            epsilon_decay (float): Decay rate for epsilon.
            learning_rate (float): Learning rate for the optimizer.
            memory_size (int): Size of the replay memory.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        # Replay memory
        self.memory = deque(maxlen=memory_size)

        # Device for PyTorch (use GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Network and Target Network
        self.q_network = DQNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Sync weights

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay memory.

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state after the action.
            done (bool): Whether the episode has ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action based on the epsilon-greedy policy.

        Args:
            state (np.ndarray): Current state.

        Returns:
            int: Action index (0: Buy, 1: Hold, 2: Sell).
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Random action
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Add batch dimension
        with torch.no_grad():
            q_values = self.q_network(state)
        return np.argmax(q_values.cpu().numpy())  # Exploit learned Q-values

    def replay(self, batch_size):
        """
        Train the Q-network using a batch of experiences from the replay memory.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            float: Training loss for the batch.
        """
        if len(self.memory) < batch_size:
            return  # Not enough samples for training

        # Sample a batch of experiences
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)  # Actions as indices
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q-value predictions for current states
        q_values = self.q_network(states).gather(1, actions)

        # Target Q-values for next states
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Compute loss (Mean Squared Error)
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        Update the target network with weights from the Q-network.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """
        Decay the exploration rate (epsilon).
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from .DQNetwork import DQNetwork


class DQNAgent:
    def __init__(self, state_size, action_size, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.998, gamma=0.9, lr=0.001):
        """
        Initialize a DQN Agent.

        Args:
            state_size (int): Dimension of the state space.
            action_size (int): Number of possible actions.
            epsilon (float): Initial exploration rate.
            epsilon_min (float): Minimum exploration rate.
            epsilon_decay (float): Decay rate for exploration.
            gamma (float): Discount factor for future rewards.
            lr (float): Learning rate for the optimizer.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Replay memory
        self.memory = deque(maxlen=20000)

        # Neural networks for Q-learning
        self.model = DQNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNetwork(state_size, action_size).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Initialize target model weights
        self.update_target_model()

    def update_target_model(self):
        """Copy weights from the current model to the target model."""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """Store experiences in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()  # Set to evaluation mode

    def act(self, state, train=True):
        """
        Choose an action using an epsilon-greedy policy.
        """
        # If in training mode and random number < epsilon, pick random action:
        if train and (np.random.rand() <= self.epsilon):
            return np.random.randint(self.action_size)
        else:
            # Exploitation
            self.model.eval()
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                act_values = self.model(state)
            return np.argmax(act_values.cpu().numpy())
        

    def replay(self, batch_size):
        """Train the model using a random batch from replay memory."""
        self.model.train() 
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.model(states).gather(1, actions)

        # Target Q values
        q_next = self.target_model(next_states).max(1)[0].detach()
        q_targets = rewards + (self.gamma * q_next * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(q_values.squeeze(), q_targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

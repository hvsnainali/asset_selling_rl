import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        """
        Initialize the deep Q-network.

        Args:
            state_size (int): Number of features in the state.
            action_size (int): Number of possible actions.
            hidden_size (int): Number of units in the hidden layers.
        """
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        """
        Perform a forward pass through the network.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: Q-values for each action.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation on the output layer (Q-values)

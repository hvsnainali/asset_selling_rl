import torch
import torch.nn as nn

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, dropout_rate=0.2):
        super(DQNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate), 
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.fc(x)

    def _initialize_weights(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

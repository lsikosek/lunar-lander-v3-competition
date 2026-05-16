import torch
import torch.nn as nn


class DQNModel(nn.Module):
    """Dueling Q-network for discrete LunarLander actions."""

    def __init__(self, obs_dim=8, num_actions=4, hidden=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden = hidden

        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, observations):
        features = self.features(observations)
        advantage = self.advantage(features)
        value = self.value(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def save(self, path="dqn_model.pt"):
        torch.save(self.state_dict(), path)

    def load(self, path="dqn_model.pt"):
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))

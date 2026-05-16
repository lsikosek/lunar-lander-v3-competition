import json
import os
import numpy as np
import torch

from dqn_model import DQNModel
from rl_model import RLModel


class Agent:
    def __init__(self, env=None, player_name=None):
        config = {}
        if os.path.exists("agent_config.json"):
            with open("agent_config.json", "r", encoding="utf-8") as f:
                config = json.load(f)

        dqn_path = "dqn_model.pt"
        if config.get("model") == "dqn" and os.path.exists(dqn_path):
            self.kind = "dqn"
            self.model = DQNModel(obs_dim=8, num_actions=4)
            self.model.load(dqn_path)
        else:
            self.kind = "a2c"
            path = "model.pt"
            self.model = RLModel(obs_dim=8, num_actions=4)
            if os.path.exists(path):
                self.model.load(path)
        self.model.eval()

    @torch.inference_mode()
    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False,
                      info=None, action_mask=None):
        obs = torch.from_numpy(np.asarray(observation, dtype=np.float32))[None, :]
        if self.kind == "dqn":
            scores = self.model(obs)
        else:
            scores, _ = self.model(obs)
        action_index = int(scores.argmax(dim=-1).item())

        return action_index

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor

from config import config


class EmbeddingModel(nn.Module):
    def __init__(self, obs_size, num_outputs):
        super(EmbeddingModel, self).__init__()
        self.obs_size = obs_size
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(obs_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.last = nn.Linear(32 * 2, num_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, x1, x2):  # for classifier. for training
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x = torch.cat([x1, x2], dim=2)
        x = self.last(x)
        return nn.Softmax(dim=2)(x)

    def embedding(self, x):  # controllable state 뽑기
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def train_model(self, batch):
        batch_size = torch.stack(batch.state).size()[0]
        # last 5 in sequence
        states = torch.stack(batch.state).view(batch_size, config.sequence_length, self.obs_size)[:, -5:, :]
        next_states = torch.stack(batch.next_state).view(batch_size, config.sequence_length, self.obs_size)[:, -5:, :]
        actions = torch.stack(batch.action).view(batch_size, config.sequence_length, -1).long()[:, -5:, :]

        self.optimizer.zero_grad()
        net_out = self.forward(states, next_states)
        actions_one_hot = torch.squeeze(F.one_hot(actions, self.num_outputs)).float()
        loss = nn.MSELoss()(net_out, actions_one_hot)
        loss.backward()
        self.optimizer.step()
        return loss.item()

class GFunction(nn.Module):
    def __init__(self, obs_size, num_outputs=128):
        super().__init__()
        self.obs_size = obs_size
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(obs_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.last = nn.Linear(32, num_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.last(x)
        return x

    def train_model(self, c_out, next_state):
        loss = nn.MSELoss()(c_out, self.forward(next_state))
        loss.backward()
        self.optimizer.step()
        return loss.item()

def compute_intrinsic_reward(
    episodic_memory: List,
    current_c_state: Tensor,
    k=10,
    kernel_cluster_distance=0.008,
    kernel_epsilon=0.0001,
    c=0.001,
    sm=8,
    alpha=None,
    episode_steps=-1,
    MA=None
) -> float:
    state_dist = [(c_state, torch.dist(c_state, current_c_state)) for c_state in episodic_memory]
    state_dist.sort(key=lambda x: x[1])
    state_dist = state_dist[:k]
    dist = [d[1].item() for d in state_dist]
    dist = np.array(dist)

    # TODO: moving average
    dist = dist / (np.mean(dist) + kernel_epsilon)

    dist = np.max(dist - kernel_cluster_distance, 0)
    kernel = kernel_epsilon / (dist / MA + kernel_epsilon)
    s = np.sqrt(np.sum(kernel)) + c

    MA = (MA * episode_steps + dist) / (episode_steps + 1)

    if np.isnan(s) or s > sm:
        return 0, MA
    return (1 / s) * min(max(alpha, 1), config.L), MA# return : repisodict·min{max{αt,1},L}

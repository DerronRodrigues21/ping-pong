"""
DQN Agent — PyTorch implementation with replay buffer and target network.
Architecture: 6 → 256(ReLU) → 256(ReLU) → 128(ReLU) → 3
Must match the Keras/TF.js export exactly.
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_device():
    """Pick the training device, with an env override for quick experiments."""
    requested = os.getenv("DQN_DEVICE", "").strip().lower()
    if requested:
        if requested == "mps" and not torch.backends.mps.is_available():
            print("[DQNAgent] DQN_DEVICE=mps requested, but MPS is unavailable; using CPU")
            return torch.device("cpu")
        return torch.device(requested)

    # This model and batch size are small enough that MPS dispatch/sync overhead
    # is usually slower than CPU on Apple Silicon.
    return torch.device("cpu")


class QNetwork(nn.Module):
    """
    Q-value network.
    Input:  6 (state)
    Output: 3 (Q-values for each action)
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size circular replay buffer."""

    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent with experience replay and target network.
    """

    def __init__(
        self,
        lr=1e-4,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.9999,
        epsilon_min=0.01,
        buffer_size=100_000,
        batch_size=64,
        target_update=1000,
    ):
        self.device = get_device()
        print(f"[DQNAgent] Using device: {self.device}")

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, 2)

        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(t)
            return q.argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def train_step(self, return_loss=False):
        """Sample a batch and perform one gradient step. Optionally return loss."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
        next_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device).unsqueeze(1)

        # Current Q-values for chosen actions
        q_values = self.policy_net(states_t).gather(1, actions_t)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_t).max(dim=1, keepdim=True)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Target network update
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        if return_loss:
            return loss.item()
        return None

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def avg_q(self, states):
        """Compute average Q-value for a batch of states (for logging)."""
        with torch.no_grad():
            t = torch.tensor(states, dtype=torch.float32, device=self.device)
            return self.policy_net(t).max(dim=1)[0].mean().item()

    def save(self, path, episode=None):
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "episode": episode,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self.steps = ckpt["steps"]
        return ckpt

"""
DQN Trainer with Double DQN, soft updates, and Bellman details.

Handles experience replay, target network updates, and optimization.

Fixes:
- M6: Use deque for loss history (no copy on every step)
- Use per-sample discount from replay buffer
- Remove non_blocking=True (CPU default)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

from .config import AgentConfig
from .network import QNet
from .replay import ReplayBuffer


class QTrainer:
    """
    Trainer for Q-Learning using Bellman Equation.

    Features:
    - Double DQN (online net chooses action, target net evaluates)
    - Vectorized soft target updates
    - Huber loss
    - Gradient clipping
    - Bellman details for visualization
    - Per-sample discount (gamma^k) from replay buffer
    """

    def __init__(
        self,
        model: QNet,
        config: AgentConfig | None = None,
    ):
        self.model = model
        self.config = config or AgentConfig()

        # Device setup - default to CPU, CUDA opt-in
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.model.to(self.device)

        # Target network (for stable Q-targets)
        self.target_model = QNet(self.model.config).to(self.device)
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=self.config.lr)

        # Loss function
        self.criterion = nn.SmoothL1Loss()  # Huber loss, more stable than MSE

        # Training stats - use deque for O(1) bounded history
        self.training_step = 0
        self.loss_history: deque[float] = deque(maxlen=10_000)

        # For Bellman details
        self._last_bellman_details = None

    def soft_update_target(self) -> None:
        """Soft update target network parameters using vectorized lerp."""
        with torch.no_grad():
            # Use _foreach_lerp_ for vectorized update (faster than Python loop)
            target_params = list(self.target_model.parameters())
            online_params = list(self.model.parameters())
            torch._foreach_lerp_(target_params, online_params, self.config.tau)

    def hard_update_target(self) -> None:
        """Hard update target network (copy weights)."""
        self.target_model.load_state_dict(self.model.state_dict())

    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        discounts: torch.Tensor,  # Per-sample gamma^k from replay buffer
        capture_details: bool = False,
    ) -> float | tuple[float, dict]:
        """
        Perform one training step using Bellman equation.

        Args:
            states: Batch of current states (batch_size, state_dim)
            actions: Batch of actions taken (batch_size,)
            rewards: Batch of rewards received (batch_size,)
            next_states: Batch of next states (batch_size, state_dim)
            terminated: Batch of terminated flags (batch_size,)
                       True = game over (suppress bootstrap)
                       False = continue or truncated (keep bootstrap)
            discounts: Per-sample discount factors (gamma^k) from replay buffer
            capture_details: If True, return Bellman update details for first transition

        Returns:
            Loss value, or (loss, details_dict) if capture_details=True
        """
        self.model.train()

        # Move to device (non_blocking has no effect on CPU)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminated = terminated.to(self.device)
        discounts = discounts.to(self.device)

        # Current Q-values: Q(s, a)
        all_q_values = self.model(states)
        current_q_values = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + discount * Q'(s', a') * (1 - terminated)
        # Double DQN: choose action with online net, evaluate with target net
        with torch.no_grad():
            if self.config.double_dqn:
                # Online net chooses action
                next_q_online = self.model(next_states)
                greedy_actions = next_q_online.argmax(1, keepdim=True)
                # Target net evaluates
                next_q_target = self.target_model(next_states)
                max_next_q_values = next_q_target.gather(1, greedy_actions).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_model(next_states)
                max_next_q_values = next_q_values.max(1)[0]
                greedy_actions = next_q_values.argmax(1, keepdim=True)

            # Use per-sample discount from replay buffer
            target_q_values = rewards + (discounts * max_next_q_values * (1 - terminated.float()))

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft update target network
        self.soft_update_target()

        self.training_step += 1
        loss_val: float = loss.item()
        self.loss_history.append(loss_val)

        if capture_details:
            # Capture details for the first transition in batch
            details = {
                "state": states[0].detach().cpu().numpy(),
                "action": int(actions[0].item()),
                "reward": float(rewards[0].item()),
                "next_state": next_states[0].detach().cpu().numpy(),
                "terminated": bool(terminated[0].item()),
                "discount": float(discounts[0].item()),
                "current_q": all_q_values[0].detach().cpu().numpy(),
                "current_q_taken": float(current_q_values[0].item()),
                "target_q": float(target_q_values[0].item()),
                "td_error": float((target_q_values[0] - current_q_values[0]).item()),
                "target_q_max": float(max_next_q_values[0].item()),
                "greedy_action": int(greedy_actions[0].item()),
                "gamma": self.config.gamma,
                "n_step": self.config.n_step,
            }
            self._last_bellman_details = details
            return loss_val, details

        return loss_val

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a single state (for inference)."""
        self.model.eval()
        with torch.no_grad():
            # Use torch.from_numpy for zero-copy conversion
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values: torch.Tensor = self.model(state_tensor)
            result: np.ndarray = q_values.cpu().numpy().squeeze()
            return result.astype(np.float32)

    def get_action_probabilities(self, state: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Get action probabilities using softmax over Q-values."""
        q_values = self.get_q_values(state)
        scaled_q = q_values / temperature
        probs_t: torch.Tensor = F.softmax(torch.from_numpy(scaled_q), dim=0)
        probs: np.ndarray = probs_t.numpy()
        return probs.astype(np.float32)

    @staticmethod
    def probabilities_from_q_values(q_values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute action probabilities from existing Q-values (no forward pass)."""
        scaled_q = np.asarray(q_values, dtype=np.float64) / temperature
        scaled_q -= scaled_q.max()  # Numerical stability
        weights: np.ndarray = np.exp(scaled_q)
        result: np.ndarray = weights / weights.sum()
        return result.astype(np.float32)

    def get_last_bellman_details(self) -> dict | None:
        """Get Bellman update details from last optimization step for visualization."""
        return self._last_bellman_details


if __name__ == "__main__":
    # Test the trainer
    from .config import NetworkConfig, AgentConfig

    config = AgentConfig(network=NetworkConfig(input_size=16, hidden_size=64))
    model = QNet(config.network)
    print(f"Model architecture:\n{model}")

    trainer = QTrainer(model, config)

    # Test forward pass
    states = torch.randn(64, 16)
    actions = torch.randint(0, 3, (64,))
    rewards = torch.randn(64)
    next_states = torch.randn(64, 16)
    dones = torch.randint(0, 2, (64,)).bool()
    discounts = torch.full((64,), 0.99**3)  # gamma^3

    loss = trainer.train_step(states, actions, rewards, next_states, dones, discounts)
    print(f"Training loss: {loss:.4f}")

    # Test inference
    test_state = np.random.randn(16).astype(np.float32)
    q_vals = trainer.get_q_values(test_state)
    print(f"Q-values: {q_vals}")

    probs = trainer.get_action_probabilities(test_state)
    print(f"Action probs: {probs}")

    print("All tests passed!")
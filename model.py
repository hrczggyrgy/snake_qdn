"""
Neural Network Models for Snake DQN Agent.

Contains the Q-Network architecture and trainer logic.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Linear_QNet(nn.Module):
    """
    Feedforward Neural Network for Q-Learning.

    Architecture:
    - Input: 11 features (state vector)
    - Hidden 1: 256 units, ReLU
    - Hidden 2: 256 units, ReLU
    - Output: 3 Q-values (Straight, Right, Left)
    """

    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 256,
        output_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Network layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def get_activations(self, x: torch.Tensor) -> dict:
        """
        Get intermediate activations for visualization.
        Returns dict with layer outputs.
        """
        activations = {}
        x = x.clone()

        x = F.relu(self.fc1(x))
        activations["fc1"] = x.detach().cpu().numpy()

        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        activations["fc2"] = x.detach().cpu().numpy()

        x = self.dropout(x)
        x = self.fc3(x)
        activations["fc3"] = x.detach().cpu().numpy()

        return activations

    def save(self, file_path: str) -> None:
        """Save model weights to file."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "output_size": self.output_size,
            },
            file_path,
        )

    @classmethod
    def load(cls, file_path: str, device: str = "cpu") -> "Linear_QNet":
        """Load model from file."""
        checkpoint = torch.load(file_path, map_location=device)
        model = cls(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            output_size=checkpoint["output_size"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model


class QTrainer:
    """
    Trainer for Q-Learning using Bellman Equation.

    Handles experience replay, target network updates, and optimization.
    """

    def __init__(
        self,
        model: Linear_QNet,
        lr: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.005,  # Soft update parameter
        device: str = "auto",
    ):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.tau = tau

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Target network (for stable Q-targets)
        self.target_model = Linear_QNet(model.input_size, model.hidden_size, model.output_size).to(
            self.device
        )
        self.target_model.load_state_dict(model.state_dict())
        self.target_model.eval()

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        # Loss function
        self.criterion = nn.SmoothL1Loss()  # Huber loss, more stable than MSE

        # Training stats
        self.training_step = 0
        self.loss_history = []

    def soft_update_target(self) -> None:
        """Soft update target network parameters."""
        for target_param, param in zip(
            self.target_model.parameters(), self.model.parameters(), strict=True
        ):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

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
    ) -> float:
        """
        Perform one training step using Bellman equation.

        Args:
            states: Batch of current states (batch_size, 11)
            actions: Batch of actions taken (batch_size,)
            rewards: Batch of rewards received (batch_size,)
            next_states: Batch of next states (batch_size, 11)
            terminated: Batch of terminated flags (batch_size,)
                       True = game over (suppress bootstrap)
                       False = continue or truncated (keep bootstrap)

        Returns:
            Loss value
        """
        self.model.train()

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminated = terminated.to(self.device)

        # Current Q-values: Q(s, a)
        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values: r + gamma * max_a' Q'(s', a') * (1 - terminated)
        # Only suppress bootstrap on actual termination, not truncation
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - terminated.float()))

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
        loss_val = loss.item()
        self.loss_history.append(loss_val)

        return loss_val

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a single state (for inference)."""
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state_tensor)
            return q_values.cpu().numpy().squeeze()

    def get_action_probabilities(self, state: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Get action probabilities using softmax over Q-values."""
        q_values = self.get_q_values(state)
        # Apply temperature
        scaled_q = q_values / temperature
        probs = F.softmax(torch.FloatTensor(scaled_q), dim=0).numpy()
        return probs

    @staticmethod
    def probabilities_from_q_values(q_values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Compute action probabilities from existing Q-values (no forward pass).

        Args:
            q_values: Q-values array of shape (3,)
            temperature: Softmax temperature

        Returns:
            Action probabilities (3,)
        """
        scaled_q = np.asarray(q_values, dtype=np.float64) / temperature
        scaled_q -= scaled_q.max()  # Numerical stability
        weights = np.exp(scaled_q)
        return weights / weights.sum()

    def save_checkpoint(
        self, file_path: str, episode: int, epsilon: float, high_score: int, avg_score: float
    ) -> None:
        """Save full training checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_step": self.training_step,
                "episode": episode,
                "epsilon": epsilon,
                "high_score": high_score,
                "avg_score": avg_score,
                "loss_history": self.loss_history[-1000:],  # Keep last 1000
                "model_config": {
                    "input_size": self.model.input_size,
                    "hidden_size": self.model.hidden_size,
                    "output_size": self.model.output_size,
                },
            },
            file_path,
        )

    def load_checkpoint(self, file_path: str) -> dict:
        """Load full training checkpoint."""
        checkpoint = torch.load(file_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.loss_history = checkpoint.get("loss_history", [])

        return {
            "episode": checkpoint.get("episode", 0),
            "epsilon": checkpoint.get("epsilon", 1.0),
            "high_score": checkpoint.get("high_score", 0),
            "avg_score": checkpoint.get("avg_score", 0.0),
        }


class Dueling_QNet(nn.Module):
    """
    Dueling DQN Architecture (optional enhancement).
    Separates value and advantage streams.
    """

    def __init__(self, input_size: int = 11, hidden_size: int = 256, output_size: int = 3):
        super().__init__()

        # Shared feature layers
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


if __name__ == "__main__":
    # Test the model
    model = Linear_QNet()
    print(f"Model architecture:\n{model}")

    # Test forward pass
    x = torch.randn(32, 11)  # Batch of 32
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (32, 3)

    # Test activations
    acts = model.get_activations(x[0])
    for k, v in acts.items():
        print(f"{k}: {v.shape}")

    # Test trainer
    trainer = QTrainer(model)
    states = torch.randn(64, 11)
    actions = torch.randint(0, 3, (64,))
    rewards = torch.randn(64)
    next_states = torch.randn(64, 11)
    dones = torch.randint(0, 2, (64,)).bool()

    loss = trainer.train_step(states, actions, rewards, next_states, dones)
    print(f"Training loss: {loss:.4f}")

    print("All tests passed!")

"""
Neural Network Models for Snake QDN Agent.

Contains the Q-Network architecture.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NetworkConfig


class QNet(nn.Module):
    """
    Feedforward Neural Network for Q-Learning.

    Architecture:
    - Input: 14 features (extended state vector)
    - Hidden 1: 64 units, ReLU
    - Hidden 2: 64 units, ReLU
    - Output: 3 Q-values (Straight, Right, Left)

    No dropout (anti-pattern in DQN - target net runs in eval mode)
    """

    def __init__(
        self,
        config: NetworkConfig | None = None,
        # Backward compatibility: accept flat kwargs
        input_size: int | None = None,
        hidden_size: int | None = None,
        output_size: int | None = None,
        dropout: float | None = None,
    ):
        super().__init__()
        if config is not None:
            self.config = config
        else:
            # Build config from flat kwargs for backward compatibility
            kwargs = {}
            if input_size is not None: kwargs["input_size"] = input_size
            if hidden_size is not None: kwargs["hidden_size"] = hidden_size
            if output_size is not None: kwargs["output_size"] = output_size
            if dropout is not None: kwargs["dropout"] = dropout
            self.config = NetworkConfig(**kwargs) if kwargs else NetworkConfig()
        self.input_size = self.config.input_size
        self.hidden_size = self.config.hidden_size
        self.output_size = self.config.output_size
        self.dropout_p = self.config.dropout

        # Network layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

        # Dropout (only used if dropout > 0)
        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else None

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
        if self.dropout is not None:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc3(x)
        return x

    def get_activations(self, x: torch.Tensor) -> dict[str, np.ndarray]:
        """
        Get intermediate activations for visualization.
        Returns dict with layer outputs.
        """
        activations = {}
        x = x.clone()

        x = F.relu(self.fc1(x))
        activations["fc1"] = x.detach().cpu().numpy()

        if self.dropout is not None:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        activations["fc2"] = x.detach().cpu().numpy()

        if self.dropout is not None:
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
                "dropout": self.dropout_p,
            },
            file_path,
        )

    @classmethod
    def load(cls, file_path: str, device: str = "cpu") -> "QNet":
        """Load model from file."""
        checkpoint = torch.load(file_path, map_location=device, weights_only=True)
        config = NetworkConfig(
            input_size=checkpoint["input_size"],
            hidden_size=checkpoint["hidden_size"],
            output_size=checkpoint["output_size"],
            dropout=checkpoint.get("dropout", 0.0),
        )
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model


if __name__ == "__main__":
    # Test the model
    model = QNet()
    print(f"Model architecture:\n{model}")

    # Test forward pass
    x = torch.randn(32, 14)  # Batch of 32
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (32, 3)

    # Test activations
    acts = model.get_activations(x[0])
    for k, v in acts.items():
        print(f"{k}: {v.shape}")

    print("All tests passed!")
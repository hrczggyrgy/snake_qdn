"""
Configuration classes for Snake QDN.

Single source of truth for all hyperparameters and environment settings.
"""

from dataclasses import dataclass, field
from typing import Any


# Single source of truth for observation dimension
OBS_DIM = 17


@dataclass(frozen=True)
class EnvConfig:
    """Validated environment configuration."""

    grid_size: int = 8  # Start with curriculum stage 0
    step_penalty: float = -0.01
    food_reward: float = 10.0
    death_penalty: float = -10.0
    render_mode: str | None = "rgb_array"
    seed: int | None = None
    allow_tail_entry: bool = True
    shaping_coef: float = 0.1  # Potential-based reward shaping coefficient
    # Curriculum settings
    curriculum_enabled: bool = True
    curriculum_stages: list[int] = field(default_factory=lambda: [8, 12, 20])
    curriculum_threshold: float = 10.0  # Rolling avg score to advance

    def __post_init__(self):
        if self.grid_size < 3:
            raise ValueError(f"grid_size must be >= 3, got {self.grid_size}")
        if not -1.0 <= self.step_penalty <= 0.0:
            raise ValueError(f"step_penalty must be in [-1.0, 0.0], got {self.step_penalty}")
        if self.food_reward < 0.0:
            raise ValueError(f"food_reward must be >= 0, got {self.food_reward}")
        if self.death_penalty > 0.0:
            raise ValueError(f"death_penalty must be <= 0, got {self.death_penalty}")
        if not self.curriculum_stages:
            raise ValueError("curriculum_stages must not be empty")
        if any(s < 3 for s in self.curriculum_stages):
            raise ValueError("All curriculum stages must be >= 3")


@dataclass(frozen=True)
class NetworkConfig:
    """Neural network architecture configuration."""

    input_size: int = OBS_DIM
    hidden_size: int = 64  # Shrunk from 256
    output_size: int = 3
    dropout: float = 0.0  # Removed dropout (anti-pattern in DQN)

    def __post_init__(self):
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {self.hidden_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class AgentConfig:
    """Validated agent configuration."""

    network: NetworkConfig
    lr: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    memory_size: int
    batch_size: int
    tau: float
    device: str
    learning_starts: int
    train_every: int
    gradient_steps: int
    epsilon_decay_steps: int
    shaping_coef: float
    n_step: int
    double_dqn: bool

    def __init__(
        self,
        # New structured config
        network: NetworkConfig | None = None,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 100_000,
        batch_size: int = 64,
        tau: float = 0.005,
        device: str = "cpu",
        learning_starts: int = 1000,
        train_every: int = 2,
        gradient_steps: int = 1,
        epsilon_decay_steps: int = 50_000,
        shaping_coef: float = 0.1,
        n_step: int = 3,
        double_dqn: bool = True,
        # Backward compatibility: accept flat kwargs
        hidden_size: int | None = None,
        dropout: float | None = None,
        input_size: int | None = None,
        output_size: int | None = None,
    ):
        # Handle network config
        if network is not None:
            self.network = network
        else:
            # Build network config from flat kwargs
            net_kwargs = {}
            if input_size is not None: net_kwargs["input_size"] = input_size
            if hidden_size is not None: net_kwargs["hidden_size"] = hidden_size
            if output_size is not None: net_kwargs["output_size"] = output_size
            if dropout is not None: net_kwargs["dropout"] = dropout
            self.network = NetworkConfig(**net_kwargs) if net_kwargs else NetworkConfig(input_size=16, hidden_size=64)

        # Set all other fields
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.device = device
        self.learning_starts = learning_starts
        self.train_every = train_every
        self.gradient_steps = gradient_steps
        self.epsilon_decay_steps = epsilon_decay_steps
        self.shaping_coef = shaping_coef
        self.n_step = n_step
        self.double_dqn = double_dqn

        self.__post_init__()

    def __post_init__(self):
        if not 0.0 < self.lr <= 1.0:
            raise ValueError(f"lr must be in (0, 1], got {self.lr}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 <= self.epsilon_start <= 1.0:
            raise ValueError(f"epsilon_start must be in [0, 1], got {self.epsilon_start}")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start:
            raise ValueError(f"epsilon_end must be in [0, epsilon_start], got {self.epsilon_end}")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError(f"epsilon_decay must be in (0, 1], got {self.epsilon_decay}")
        if self.memory_size <= 0:
            raise ValueError(f"memory_size must be > 0, got {self.memory_size}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.batch_size > self.memory_size:
            raise ValueError(f"batch_size ({self.batch_size}) must be <= memory_size ({self.memory_size})")
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if self.learning_starts < 0:
            raise ValueError(f"learning_starts must be >= 0, got {self.learning_starts}")
        if self.train_every <= 0:
            raise ValueError(f"train_every must be > 0, got {self.train_every}")
        if self.gradient_steps <= 0:
            raise ValueError(f"gradient_steps must be > 0, got {self.gradient_steps}")
        if self.epsilon_decay_steps < 0:
            raise ValueError(f"epsilon_decay_steps must be >= 0, got {self.epsilon_decay_steps}")
        if not 0.0 <= self.shaping_coef <= 1.0:
            raise ValueError(f"shaping_coef must be in [0, 1], got {self.shaping_coef}")
        if self.n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {self.n_step}")

    def to_agent_config(self) -> "AgentConfig":
        """Return self (for backward compatibility with LegacyAgentConfig)."""
        return self


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    max_env_transitions: int = 500_000
    max_wall_time_seconds: float | None = None
    max_optimizer_steps: int | None = None
    max_episodes: int | None = None
    eval_every: int = 5000
    eval_episodes: int = 5
    save_every: int = 100
    save_path: str = "checkpoints/snake_dqn.pth"


# Legacy config for backward compatibility
@dataclass
class LegacyAgentConfig:
    """Legacy flat config matching old Agent.__init__ signature."""

    input_size: int = 11
    hidden_size: int = 256
    output_size: int = 3
    lr: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 100_000
    batch_size: int = 256
    tau: float = 0.005
    dropout: float = 0.1
    device: str = "auto"
    learning_starts: int = 1000
    train_every: int = 1
    gradient_steps: int = 1
    epsilon_decay_steps: int = 0

    def to_agent_config(self) -> AgentConfig:
        """Convert to new structured config."""
        return AgentConfig(
            network=NetworkConfig(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                dropout=self.dropout,
            ),
            lr=self.lr,
            gamma=self.gamma,
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay=self.epsilon_decay,
            memory_size=self.memory_size,
            batch_size=self.batch_size,
            tau=self.tau,
            device=self.device,
            learning_starts=self.learning_starts,
            train_every=self.train_every,
            gradient_steps=self.gradient_steps,
            epsilon_decay_steps=self.epsilon_decay_steps,
        )
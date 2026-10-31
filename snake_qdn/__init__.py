"""
Snake QDN - Deep Q-Network for Snake Game.

A clean, modular implementation of DQN with modern improvements:
- Double DQN
- n-step returns
- Potential-based reward shaping
- Step-based epsilon decay
- Grid curriculum
- Vectorized operations for speed
"""

from .agent import Agent, evaluate_agent
from .config import (
    AgentConfig,
    EnvConfig,
    NetworkConfig,
    TrainingConfig,
    LegacyAgentConfig,
)
from .controller import TrainingController
from .env import SnakeEnv, make_snake_env
from .events import (
    ActivityMode,
    BellmanDetails,
    DecisionRecord,
    EventType,
    PhaseInfo,
    TrainingEvent,
    TrainingSnapshot,
)
from .headless import train_headless, train_from_config, evaluate_policy
from .network import QNet
from .persistence import (
    CheckpointManager,
    ModelConfig,
    CHECKPOINT_VERSION,
    TrainingState,
    capture_rng_state,
    restore_rng_state,
    save_checkpoint,
    load_checkpoint,
    save_model_only,
    load_model_only,
)
from .replay import ReplayBuffer
from .trainer import QTrainer

__version__ = "2.0.0"

__all__ = [
    # Config
    "AgentConfig",
    "EnvConfig",
    "NetworkConfig",
    "TrainingConfig",
    "LegacyAgentConfig",
    # Core components
    "SnakeEnv",
    "make_snake_env",
    "QNet",
    "ReplayBuffer",
    "QTrainer",
    "Agent",
    "TrainingController",
    # Training
    "train_headless",
    "train_from_config",
    "evaluate_policy",
    "evaluate_agent",
    # Persistence
    "CheckpointManager",
    "ModelConfig",
    "CHECKPOINT_VERSION",
    "TrainingState",
    "capture_rng_state",
    "restore_rng_state",
    "save_checkpoint",
    "load_checkpoint",
    "save_model_only",
    "load_model_only",
    # Events & Snapshots
    "ActivityMode",
    "EventType",
    "TrainingEvent",
    "TrainingSnapshot",
    "DecisionRecord",
    "BellmanDetails",
    "PhaseInfo",
]
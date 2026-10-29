"""
Checkpoint persistence module for Snake DQN.

Handles saving/loading of training checkpoints with versioning,
validation, and backward compatibility.
"""

import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Checkpoint schema version
CHECKPOINT_VERSION = 2


@dataclass
class ModelConfig:
    """Model architecture configuration for checkpoint validation."""

    input_size: int
    hidden_size: int
    output_size: int
    dropout: float = 0.1

    def matches(self, other: "ModelConfig") -> bool:
        return (
            self.input_size == other.input_size
            and self.hidden_size == other.hidden_size
            and self.output_size == other.output_size
            and self.dropout == other.dropout
        )


@dataclass
class TrainingState:
    """Training state for exact resume capability."""

    episode: int
    epsilon: float
    high_score: int
    total_steps: int
    training_step: int
    scores_history: list
    mean_scores_history: list
    epsilon_history: list
    loss_history: list
    memory: list  # Replay buffer contents
    memory_position: int
    rng_state_python: Any
    rng_state_numpy: Any
    rng_state_torch: Any
    rng_state_cuda: Any | None = None
    termination_history: dict[str, int] | None = None


def get_model_config(model: torch.nn.Module) -> ModelConfig:
    """Extract model configuration from a Linear_QNet."""
    from model import Linear_QNet

    if isinstance(model, Linear_QNet):
        return ModelConfig(
            input_size=model.input_size,
            hidden_size=model.hidden_size,
            output_size=model.output_size,
            dropout=model.dropout.p if hasattr(model.dropout, "p") else 0.1,
        )
    raise ValueError(f"Unknown model type: {type(model)}")


def save_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    target_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    training_state: TrainingState,
    model_config: ModelConfig | None = None,
    atomic: bool = True,
) -> None:
    """
    Save full training checkpoint with versioning.

    Args:
        filepath: Path to save checkpoint
        model: Policy network
        target_model: Target network
        optimizer: Optimizer state
        training_state: Complete training state for exact resume
        model_config: Model architecture config (extracted from model if None)
        atomic: If True, write to temp file then rename (prevents corruption)
    """
    if model_config is None:
        model_config = get_model_config(model)

    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "model_config": asdict(model_config),
        "model_state_dict": model.state_dict(),
        "target_model_state_dict": target_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": asdict(training_state),
    }

    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    if atomic:
        # Atomic write: write to temp file then rename
        temp_path = filepath + ".tmp"
        torch.save(checkpoint, temp_path)
        os.replace(temp_path, filepath)
    else:
        torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    target_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    strict: bool = True,
) -> tuple[TrainingState, ModelConfig]:
    """
    Load training checkpoint with validation.

    Args:
        filepath: Path to checkpoint file
        model: Policy network (will be loaded in-place)
        target_model: Target network (will be loaded in-place)
        optimizer: Optimizer (will be loaded in-place)
        device: Target device for loading
        strict: If True, raise on architecture mismatch

    Returns:
        Tuple of (training_state, model_config)

    Raises:
        ValueError: If checkpoint version incompatible or architecture mismatch
        FileNotFoundError: If checkpoint file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # Use weights_only=False since we trust our own checkpoints and need numpy arrays
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Version handling
    version = checkpoint.get("version", 1)
    if version > CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version {version} is newer than supported version {CHECKPOINT_VERSION}. "
            "Please update the code to load this checkpoint."
        )

    # Model config validation
    saved_config_dict = checkpoint.get("model_config", {})
    if not saved_config_dict:
        # Legacy v1 checkpoint - try to infer from model
        saved_config_dict = {
            "input_size": model.input_size if hasattr(model, "input_size") else 11,
            "hidden_size": model.hidden_size if hasattr(model, "hidden_size") else 256,
            "output_size": model.output_size if hasattr(model, "output_size") else 3,
            "dropout": 0.1,
        }

    saved_config = ModelConfig(**saved_config_dict)
    current_config = get_model_config(model)

    if not saved_config.matches(current_config):
        msg = (
            f"Model architecture mismatch: "
            f"checkpoint has {saved_config}, current model has {current_config}"
        )
        if strict:
            raise ValueError(msg)
        else:
            print(f"Warning: {msg}. Loading weights with strict=False.")
            # Allow partial loading for transfer learning
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            target_model.load_state_dict(checkpoint["target_model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        target_model.load_state_dict(checkpoint["target_model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load training state
    training_state_dict = checkpoint.get("training_state", {})

    # Handle legacy checkpoints without full training state
    if not training_state_dict:
        training_state_dict = {
            "episode": checkpoint.get("episode", 0),
            "epsilon": checkpoint.get("epsilon", 1.0),
            "high_score": checkpoint.get("high_score", 0),
            "total_steps": 0,
            "training_step": checkpoint.get("training_step", 0),
            "scores_history": [],
            "mean_scores_history": [],
            "epsilon_history": [],
            "loss_history": checkpoint.get("loss_history", [])[-1000:],
            "memory": [],
            "memory_position": 0,
            "rng_state_python": None,
            "rng_state_numpy": None,
            "rng_state_torch": None,
            "rng_state_cuda": None,
            "termination_history": {"wall": 0, "self": 0, "timeout": 0, "victory": 0},
        }

    training_state = TrainingState(**training_state_dict)

    return training_state, saved_config


def save_model_only(filepath: str, model: torch.nn.Module, model_config: ModelConfig) -> None:
    """Save only model weights (for inference/deployment)."""
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "model_config": asdict(model_config),
        "model_state_dict": model.state_dict(),
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)


def load_model_only(filepath: str, model: torch.nn.Module, device: torch.device) -> ModelConfig:
    """Load only model weights (for inference/deployment)."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    version = checkpoint.get("version", 1)
    if version > CHECKPOINT_VERSION:
        raise ValueError(f"Checkpoint version {version} > supported {CHECKPOINT_VERSION}")

    saved_config_dict = checkpoint.get("model_config", {})
    saved_config = ModelConfig(**saved_config_dict)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return saved_config


def capture_rng_state() -> tuple[Any, Any, Any, Any | None]:
    """Capture current RNG state for all libraries."""
    import random

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return python_state, numpy_state, torch_state, cuda_state


def restore_rng_state(python_state, numpy_state, torch_state, cuda_state) -> None:
    """Restore RNG state for all libraries."""
    import random

    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


class CheckpointManager:
    """
    High-level checkpoint management for Agent.

    Handles both full training state (for exact resume) and
    model-only checkpoints (for deployment/inference).
    """

    def __init__(self, agent, device: torch.device):
        self.agent = agent
        self.device = device

    def save_full(self, filepath: str, atomic: bool = True) -> None:
        """Save complete training state for exact resume."""
        # Capture RNG state
        rng_python, rng_numpy, rng_torch, rng_cuda = capture_rng_state()

        # Capture replay buffer
        memory_data = list(self.agent.memory.memory)

        training_state = TrainingState(
            episode=self.agent.n_games,
            epsilon=self.agent.epsilon,
            high_score=self.agent.high_score,
            total_steps=self.agent.total_steps,
            training_step=self.agent.trainer.training_step,
            scores_history=self.agent.scores_history.copy(),
            mean_scores_history=self.agent.mean_scores_history.copy(),
            epsilon_history=self.agent.epsilon_history.copy(),
            loss_history=self.agent.trainer.loss_history.copy(),
            memory=memory_data,
            memory_position=self.agent.memory.position,
            rng_state_python=rng_python,
            rng_state_numpy=rng_numpy,
            rng_state_torch=rng_torch,
            rng_state_cuda=rng_cuda,
            termination_history=self.agent.termination_history.copy(),
        )

        model_config = get_model_config(self.agent.model)

        save_checkpoint(
            filepath=filepath,
            model=self.agent.model,
            target_model=self.agent.trainer.target_model,
            optimizer=self.agent.trainer.optimizer,
            training_state=training_state,
            model_config=model_config,
            atomic=atomic,
        )

    def load_full(self, filepath: str, strict: bool = True) -> dict[str, Any]:
        """Load complete training state for exact resume."""
        training_state, saved_config = load_checkpoint(
            filepath=filepath,
            model=self.agent.model,
            target_model=self.agent.trainer.target_model,
            optimizer=self.agent.trainer.optimizer,
            device=self.device,
            strict=strict,
        )

        # Restore agent state
        self.agent.n_games = training_state.episode
        self.agent.epsilon = training_state.epsilon
        self.agent.high_score = training_state.high_score
        self.agent.total_steps = training_state.total_steps
        self.agent.trainer.training_step = training_state.training_step
        self.agent.scores_history = training_state.scores_history
        self.agent.mean_scores_history = training_state.mean_scores_history
        self.agent.epsilon_history = training_state.epsilon_history
        self.agent.trainer.loss_history = training_state.loss_history
        self.agent.termination_history = training_state.termination_history or {"wall": 0, "self": 0, "timeout": 0, "victory": 0}

        # Restore replay buffer
        self.agent.memory.memory = deque(training_state.memory, maxlen=self.agent.memory.capacity)
        self.agent.memory.position = training_state.memory_position

        # Restore RNG state
        restore_rng_state(
            training_state.rng_state_python,
            training_state.rng_state_numpy,
            training_state.rng_state_torch,
            training_state.rng_state_cuda,
        )

        return {
            "episode": training_state.episode,
            "epsilon": training_state.epsilon,
            "high_score": training_state.high_score,
            "avg_score": training_state.mean_scores_history[-1]
            if training_state.mean_scores_history
            else 0.0,
        }

    def save_weights_only(self, filepath: str) -> None:
        """Save only model weights (for inference)."""
        model_config = get_model_config(self.agent.model)
        save_model_only(filepath, self.agent.model, model_config)

    def load_weights_only(self, filepath: str) -> ModelConfig:
        """Load only model weights (for inference)."""
        return load_model_only(filepath, self.agent.model, self.device)


# Backward compatibility functions for existing Agent.save_checkpoint/load_checkpoint
def save_agent_checkpoint(agent, filepath: str, **kwargs) -> None:
    """Backward compatible save for Agent.checkpoint()."""
    manager = CheckpointManager(agent, agent.trainer.device)
    manager.save_full(filepath)


def load_agent_checkpoint(agent, filepath: str) -> dict[str, Any]:
    """Backward compatible load for Agent.checkpoint()."""
    manager = CheckpointManager(agent, agent.trainer.device)
    return manager.load_full(filepath, strict=False)


if __name__ == "__main__":
    # Quick test
    import tempfile

    from agent import Agent

    print("Testing persistence module...")

    agent = Agent(hidden_size=128)
    agent.n_games = 42
    agent.high_score = 15
    agent.epsilon = 0.3

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_checkpoint.pth")

        # Test full save/load
        manager = CheckpointManager(agent, agent.trainer.device)
        manager.save_full(path)
        print("  ✓ Full checkpoint saved")

        # Create new agent and load
        new_agent = Agent(hidden_size=128)
        new_manager = CheckpointManager(new_agent, new_agent.trainer.device)
        info = new_manager.load_full(path)
        print(f"  ✓ Full checkpoint loaded: {info}")

        # Verify state restored
        assert new_agent.n_games == 42
        assert new_agent.high_score == 15
        assert abs(new_agent.epsilon - 0.3) < 0.01
        print("  ✓ Agent state restored")

        # Verify Q-values match
        test_state = np.random.randn(11)
        orig_q = agent.get_q_values(test_state)
        loaded_q = new_agent.get_q_values(test_state)
        assert np.allclose(orig_q, loaded_q)
        print("  ✓ Q-values match")

        # Test model-only save/load
        weights_path = os.path.join(tmpdir, "weights_only.pth")
        manager.save_weights_only(weights_path)
        print("  ✓ Weights-only saved")

        inference_agent = Agent(hidden_size=128)
        inf_manager = CheckpointManager(inference_agent, inference_agent.trainer.device)
        inf_manager.load_weights_only(weights_path)
        print("  ✓ Weights-only loaded")

        # Verify Q-values match
        inf_q = inference_agent.get_q_values(test_state)
        assert np.allclose(orig_q, inf_q)
        print("  ✓ Inference Q-values match")

    print("Persistence module test passed!")

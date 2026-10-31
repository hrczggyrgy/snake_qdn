"""
Checkpoint persistence module for Snake QDN.

Handles saving/loading of training checkpoints with versioning,
validation, and backward compatibility.
"""

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .agent import Agent
from .config import NetworkConfig
from .env import SnakeEnv
from .replay import ReplayBuffer

# Checkpoint schema version
CHECKPOINT_VERSION = 3
OBSERVATION_SCHEMA_VERSION = 1
REWARD_SCHEMA_VERSION = 1


@dataclass
class ModelConfig:
    """Model architecture configuration for checkpoint validation."""

    input_size: int
    hidden_size: int
    output_size: int
    dropout: float = 0.0

    def matches(self, other: "ModelConfig") -> bool:
        return (
            self.input_size == other.input_size
            and self.hidden_size == other.hidden_size
            and self.output_size == other.output_size
            and self.dropout == other.dropout
        )


@dataclass
class EnvironmentState:
    """Environment state for exact mid-episode resume."""

    snake: list[tuple[int, int]]
    food: tuple[int, int] | None
    direction: int
    score: int
    steps_since_food: int
    max_steps_without_food: int
    grid_size: int
    game_over: bool
    # Instance-local RNG states
    rng_state_python: Any
    rng_state_numpy: Any


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
    memory: dict  # Replay buffer contents
    rng_state_python: Any
    rng_state_numpy: Any
    rng_state_torch: Any
    rng_state_cuda: Any | None = None
    termination_history: dict[str, int] | None = None
    # Schema versions for compatibility
    observation_schema_version: int = OBSERVATION_SCHEMA_VERSION
    reward_schema_version: int = REWARD_SCHEMA_VERSION
    # Environment state for mid-episode resume
    environment_state: EnvironmentState | None = None
    # Controller state
    run_id: int = 0
    episode_id: int = 0
    step_id: int = 0
    replay_warmed_up: bool = False
    # Agent hyperparameters (for exact resume validation)
    gamma: float = 0.99
    tau: float = 0.005
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    epsilon_decay_steps: int = 0
    learning_starts: int = 1000
    train_every: int = 1
    gradient_steps: int = 1
    # Internal agent state
    transitions_since_train: int = 0
    # Recent episode outcomes (for exact resume of termination order)
    recent_terminations: list[str] | None = None
    # Environment reward/rule settings
    env_grid_size: int = 20
    env_step_penalty: float = -0.01
    env_food_reward: float = 10.0
    env_death_penalty: float = -10.0
    env_allow_tail_entry: bool = True


def get_model_config(model: torch.nn.Module) -> ModelConfig:
    """Extract model configuration from a QNet."""
    from .network import QNet

    if isinstance(model, QNet):
        return ModelConfig(
            input_size=model.input_size,
            hidden_size=model.hidden_size,
            output_size=model.output_size,
            dropout=model.dropout_p,
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
    """Save full training checkpoint with versioning."""
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

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    if atomic:
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
    """Load training checkpoint with validation."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # Use weights_only=False since we trust our own checkpoints and need numpy arrays
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    version = checkpoint.get("version", 1)
    if version > CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version {version} is newer than supported version {CHECKPOINT_VERSION}. "
            "Please update the code to load this checkpoint."
        )

    # Model config validation
    saved_config_dict = checkpoint.get("model_config", {})
    if not saved_config_dict:
        # Legacy v1 checkpoint
        saved_config_dict = {
            "input_size": model.input_size if hasattr(model, "input_size") else 14,
            "hidden_size": model.hidden_size if hasattr(model, "hidden_size") else 64,
            "output_size": model.output_size if hasattr(model, "output_size") else 3,
            "dropout": 0.0,
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
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            target_model.load_state_dict(checkpoint["target_model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        target_model.load_state_dict(checkpoint["target_model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load training state
    training_state_dict = checkpoint.get("training_state", {})

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
            "memory": {},
            "rng_state_python": None,
            "rng_state_numpy": None,
            "rng_state_torch": None,
            "rng_state_cuda": None,
            "termination_history": {"wall": 0, "self": 0, "timeout": 0, "victory": 0},
        }

    # Reconstruct nested EnvironmentState if it was serialized as a dict
    env_data = training_state_dict.get("environment_state")
    if isinstance(env_data, dict):
        training_state_dict["environment_state"] = EnvironmentState(**env_data)

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
    """Load only model weights (for inference/deployment).

    Uses weights_only=True for security since model-only checkpoints
    contain only tensors and primitives, not arbitrary Python objects.
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=True)

    version = checkpoint.get("version", 1)
    if version > CHECKPOINT_VERSION:
        raise ValueError(f"Checkpoint version {version} > supported {CHECKPOINT_VERSION}")

    saved_config_dict = checkpoint.get("model_config", {})
    saved_config = ModelConfig(**saved_config_dict)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return saved_config


def capture_rng_state() -> tuple[Any, Any, Any, Any | None]:
    """Capture current global RNG state for all libraries."""
    import random

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return python_state, numpy_state, torch_state, cuda_state


def restore_rng_state(python_state, numpy_state, torch_state, cuda_state) -> None:
    """Restore global RNG state for all libraries."""
    import random

    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def capture_env_rng_state(env: SnakeEnv) -> tuple[Any, Any]:
    """Capture environment-local RNG state."""
    return env._rng.getstate(), env._np_rng.bit_generator.state


def restore_env_rng_state(env: SnakeEnv, python_state, numpy_state) -> None:
    """Restore environment-local RNG state."""
    env._rng.setstate(python_state)
    env._np_rng.bit_generator.state = numpy_state


class CheckpointManager:
    """High-level checkpoint management for Agent."""

    def __init__(self, agent: Agent, device: torch.device):
        self.agent = agent
        self.device = device

    def save_full(
        self,
        filepath: str,
        env: SnakeEnv | None = None,
        atomic: bool = True,
        run_id: int = 0,
        episode_id: int = 0,
        step_id: int = 0,
        replay_warmed_up: bool = False,
    ) -> None:
        """Save complete training state for exact resume."""
        rng_python, rng_numpy, rng_torch, rng_cuda = capture_rng_state()

        # Capture replay buffer exactly
        memory_data = self.agent.memory.get_state_dict()

        # Capture environment state if provided
        environment_state = None
        if env is not None:
            env_rng_python, env_rng_numpy = capture_env_rng_state(env)
            environment_state = EnvironmentState(
                snake=list(env.snake),
                food=env.food,
                direction=env.direction,
                score=env.score,
                steps_since_food=env.steps_since_food,
                max_steps_without_food=env.max_steps_without_food,
                grid_size=env.config.grid_size,
                game_over=env.game_over,
                rng_state_python=env_rng_python,
                rng_state_numpy=env_rng_numpy,
            )

        training_state = TrainingState(
            episode=self.agent.n_games,
            epsilon=self.agent.epsilon,
            high_score=self.agent.high_score,
            total_steps=self.agent.total_steps,
            training_step=self.agent.trainer.training_step,
            scores_history=self.agent.scores_history.copy(),
            mean_scores_history=self.agent.mean_scores_history.copy(),
            epsilon_history=self.agent.epsilon_history.copy(),
            loss_history=self.agent.trainer.loss_history.copy()[-1000:],
            memory=memory_data,
            rng_state_python=rng_python,
            rng_state_numpy=rng_numpy,
            rng_state_torch=rng_torch,
            rng_state_cuda=rng_cuda,
            termination_history=self.agent.termination_history.copy(),
            environment_state=environment_state,
            run_id=run_id,
            episode_id=episode_id,
            step_id=step_id,
            replay_warmed_up=replay_warmed_up,
            gamma=self.agent.gamma,
            tau=self.agent.tau,
            epsilon_start=self.agent.epsilon_start,
            epsilon_end=self.agent.epsilon_end,
            epsilon_decay=self.agent.epsilon_decay,
            epsilon_decay_steps=self.agent.epsilon_decay_steps,
            learning_starts=self.agent.learning_starts,
            train_every=self.agent.train_every,
            gradient_steps=self.agent.gradient_steps,
            transitions_since_train=self.agent._transitions_since_train,
            recent_terminations=list(self.agent.recent_terminations),
            env_grid_size=env.config.grid_size if env is not None else 20,
            env_step_penalty=env.config.step_penalty if env is not None else -0.01,
            env_food_reward=env.config.food_reward if env is not None else 10.0,
            env_death_penalty=env.config.death_penalty if env is not None else -10.0,
            env_allow_tail_entry=env.config.allow_tail_entry if env is not None else True,
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

    def load_full(
        self, filepath: str, strict: bool = True, env: SnakeEnv | None = None
    ) -> dict[str, Any]:
        """Load complete training state for exact resume."""
        training_state, saved_config = load_checkpoint(
            filepath=filepath,
            model=self.agent.model,
            target_model=self.agent.trainer.target_model,
            optimizer=self.agent.trainer.optimizer,
            device=self.device,
            strict=strict,
        )

        # Validate schema versions
        if training_state.observation_schema_version != OBSERVATION_SCHEMA_VERSION:
            msg = (
                f"Observation schema mismatch: checkpoint has v{training_state.observation_schema_version}, "
                f"current is v{OBSERVATION_SCHEMA_VERSION}. Cannot resume."
            )
            if strict:
                raise ValueError(msg)
            else:
                print(f"Warning: {msg}")

        if training_state.reward_schema_version != REWARD_SCHEMA_VERSION:
            msg = (
                f"Reward schema mismatch: checkpoint has v{training_state.reward_schema_version}, "
                f"current is v{REWARD_SCHEMA_VERSION}. Cannot resume."
            )
            if strict:
                raise ValueError(msg)
            else:
                print(f"Warning: {msg}")

        # Validate hyperparameters match exactly for exact resume
        if strict:
            self._validate_hyperparameters(training_state)

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
        self.agent.termination_history = training_state.termination_history or {
            "wall": 0,
            "self": 0,
            "timeout": 0,
            "victory": 0,
        }
        self.agent._transitions_since_train = getattr(training_state, "transitions_since_train", 0)
        recent_terms = getattr(training_state, "recent_terminations", None)
        if recent_terms is not None:
            from collections import deque
            self.agent.recent_terminations = deque(recent_terms, maxlen=50)
        else:
            from collections import deque
            self.agent.recent_terminations = deque(
                [k for k, v in self.agent.termination_history.items() for _ in range(v)],
                maxlen=50,
            )

        # Restore hyperparameters from checkpoint (exact resume)
        self.agent.gamma = training_state.gamma
        self.agent.tau = training_state.tau
        self.agent.epsilon_start = training_state.epsilon_start
        self.agent.epsilon_end = training_state.epsilon_end
        self.agent.epsilon_decay = training_state.epsilon_decay
        self.agent.epsilon_decay_steps = training_state.epsilon_decay_steps
        self.agent.learning_starts = training_state.learning_starts
        self.agent.train_every = training_state.train_every
        self.agent.gradient_steps = training_state.gradient_steps
        self.agent._transitions_since_train = getattr(training_state, "transitions_since_train", 0)

        # Restore replay buffer
        self.agent.memory.load_state_dict(training_state.memory)

        # Restore global RNG state
        if training_state.rng_state_python is not None:
            restore_rng_state(
                training_state.rng_state_python,
                training_state.rng_state_numpy,
                training_state.rng_state_torch,
                training_state.rng_state_cuda,
            )

        # Restore environment state if provided and available
        if env is not None and training_state.environment_state is not None:
            env_state = training_state.environment_state
            env.snake = deque(env_state.snake)
            env.food = env_state.food
            env.direction = env_state.direction
            env.score = env_state.score
            env.steps_since_food = env_state.steps_since_food
            env.max_steps_without_food = env_state.max_steps_without_food
            env.config.grid_size = env_state.grid_size
            env.game_over = env_state.game_over
            if env_state.rng_state_python is not None:
                restore_env_rng_state(env, env_state.rng_state_python, env_state.rng_state_numpy)
            env._rebuild_empty_cells()

            # Validate environment settings match
            if env.config.grid_size != training_state.env_grid_size:
                msg = f"env grid_size mismatch: checkpoint={training_state.env_grid_size}, current={env.config.grid_size}"
                if strict:
                    raise ValueError(msg)
                else:
                    print(f"Warning: {msg}")
            if abs(env.config.step_penalty - training_state.env_step_penalty) > 1e-6:
                msg = f"env step_penalty mismatch: checkpoint={training_state.env_step_penalty}, current={env.config.step_penalty}"
                if strict:
                    raise ValueError(msg)
                else:
                    print(f"Warning: {msg}")
            if abs(env.config.food_reward - training_state.env_food_reward) > 1e-6:
                msg = f"env food_reward mismatch: checkpoint={training_state.env_food_reward}, current={env.config.food_reward}"
                if strict:
                    raise ValueError(msg)
                else:
                    print(f"Warning: {msg}")
            if abs(env.config.death_penalty - training_state.env_death_penalty) > 1e-6:
                msg = f"env death_penalty mismatch: checkpoint={training_state.env_death_penalty}, current={env.config.death_penalty}"
                if strict:
                    raise ValueError(msg)
                else:
                    print(f"Warning: {msg}")
            if env.config.allow_tail_entry != training_state.env_allow_tail_entry:
                msg = f"env allow_tail_entry mismatch: checkpoint={training_state.env_allow_tail_entry}, current={env.config.allow_tail_entry}"
                if strict:
                    raise ValueError(msg)
                else:
                    print(f"Warning: {msg}")

        return {
            "episode": training_state.episode,
            "epsilon": training_state.epsilon,
            "high_score": training_state.high_score,
            "avg_score": training_state.mean_scores_history[-1]
            if training_state.mean_scores_history
            else 0.0,
            "run_id": training_state.run_id,
            "episode_id": training_state.episode_id,
            "step_id": training_state.step_id,
            "replay_warmed_up": training_state.replay_warmed_up,
            "observation_schema_version": training_state.observation_schema_version,
            "reward_schema_version": training_state.reward_schema_version,
        }

    def _validate_hyperparameters(self, training_state: TrainingState) -> None:
        """Validate that agent hyperparameters match checkpoint exactly for strict resume."""
        checks = [
            ("gamma", self.agent.gamma, training_state.gamma),
            ("tau", self.agent.tau, training_state.tau),
            ("epsilon_start", self.agent.epsilon_start, training_state.epsilon_start),
            ("epsilon_end", self.agent.epsilon_end, training_state.epsilon_end),
            ("epsilon_decay", self.agent.epsilon_decay, training_state.epsilon_decay),
            ("epsilon_decay_steps", self.agent.epsilon_decay_steps, training_state.epsilon_decay_steps),
            ("learning_starts", self.agent.learning_starts, training_state.learning_starts),
            ("train_every", self.agent.train_every, training_state.train_every),
            ("gradient_steps", self.agent.gradient_steps, training_state.gradient_steps),
        ]
        for name, current, saved in checks:
            if abs(current - saved) > 1e-6:
                raise ValueError(
                    f"Hyperparameter mismatch for {name}: "
                    f"checkpoint has {saved}, current agent has {current}. "
                    "Use import_weights_only() for transfer learning."
                )

        if self.agent.memory.capacity != training_state.memory.get("capacity", self.agent.memory.capacity):
            raise ValueError(
                f"Replay buffer capacity mismatch: checkpoint has {training_state.memory.get('capacity')}, "
                f"current agent has {self.agent.memory.capacity}. "
                "Cannot resume with different replay buffer capacity."
            )

    def import_weights_only(self, filepath: str) -> ModelConfig:
        """Load only model weights for transfer learning or inference."""
        return load_model_only(filepath, self.agent.model, self.device)

    def save_weights_only(self, filepath: str) -> None:
        """Save only model weights (for inference)."""
        model_config = get_model_config(self.agent.model)
        save_model_only(filepath, self.agent.model, model_config)

    def load_weights_only(self, filepath: str) -> ModelConfig:
        """Load only model weights (for inference)."""
        return load_model_only(filepath, self.agent.model, self.device)


if __name__ == "__main__":
    import tempfile

    from .config import AgentConfig

    print("Testing persistence module...")

    agent = Agent(AgentConfig(hidden_size=64))
    agent.n_games = 42
    agent.high_score = 15
    agent.epsilon = 0.3

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_checkpoint.pth")

        manager = CheckpointManager(agent, agent.trainer.device)
        manager.save_full(path)
        print("  ✓ Full checkpoint saved")

        new_agent = Agent(AgentConfig(hidden_size=64))
        new_manager = CheckpointManager(new_agent, new_agent.trainer.device)
        info = new_manager.load_full(path)
        print(f"  ✓ Full checkpoint loaded: {info}")

        assert new_agent.n_games == 42
        assert new_agent.high_score == 15
        assert abs(new_agent.epsilon - 0.3) < 0.01
        print("  ✓ Agent state restored")

        test_state = np.random.randn(14).astype(np.float32)
        orig_q = agent.get_q_values(test_state)
        loaded_q = new_agent.get_q_values(test_state)
        assert np.allclose(orig_q, loaded_q)
        print("  ✓ Q-values match")

        weights_path = os.path.join(tmpdir, "weights_only.pth")
        manager.save_weights_only(weights_path)
        print("  ✓ Weights-only saved")

        inference_agent = Agent(AgentConfig(hidden_size=64))
        inf_manager = CheckpointManager(inference_agent, inference_agent.trainer.device)
        inf_manager.load_weights_only(weights_path)
        print("  ✓ Weights-only loaded")

        inf_q = inference_agent.get_q_values(test_state)
        assert np.allclose(orig_q, inf_q)
        print("  ✓ Inference Q-values match")

    print("Persistence module test passed!")
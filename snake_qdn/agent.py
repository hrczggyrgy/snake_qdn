"""
DQN Agent for Snake Game.

Implements epsilon-greedy exploration, training loop management,
and unified replay schedule.
"""

import random
from typing import Any

import numpy as np
import torch

from .config import AgentConfig, LegacyAgentConfig
from .env import SnakeEnv
from .network import QNet
from .replay import ReplayBuffer
from .trainer import QTrainer


class Agent:
    """
    DQN Agent for Snake.

    Features:
    - Experience Replay with n-step returns
    - Epsilon-Greedy Exploration (step-based decay by default)
    - Target Network with soft updates
    - Double DQN
    - Reward shaping (potential-based)
    - Unified replay schedule
    """

    def __init__(
        self,
        config: AgentConfig | LegacyAgentConfig | None = None,
        # Backward compatibility: accept flat kwargs
        input_size: int | None = None,
        hidden_size: int | None = None,
        output_size: int | None = None,
        lr: float | None = None,
        gamma: float | None = None,
        epsilon_start: float | None = None,
        epsilon_end: float | None = None,
        epsilon_decay: float | None = None,
        memory_size: int | None = None,
        batch_size: int | None = None,
        tau: float | None = None,
        dropout: float | None = None,
        device: str | None = None,
        learning_starts: int | None = None,
        train_every: int | None = None,
        gradient_steps: int | None = None,
        epsilon_decay_steps: int | None = None,
    ):
        # Handle both new structured config and legacy flat config
        if isinstance(config, LegacyAgentConfig):
            config = config.to_agent_config()
        elif config is None:
            # Build config from flat kwargs for backward compatibility
            kwargs = {}
            if input_size is not None: kwargs["input_size"] = input_size
            if hidden_size is not None: kwargs["hidden_size"] = hidden_size
            if output_size is not None: kwargs["output_size"] = output_size
            if lr is not None: kwargs["lr"] = lr
            if gamma is not None: kwargs["gamma"] = gamma
            if epsilon_start is not None: kwargs["epsilon_start"] = epsilon_start
            if epsilon_end is not None: kwargs["epsilon_end"] = epsilon_end
            if epsilon_decay is not None: kwargs["epsilon_decay"] = epsilon_decay
            if memory_size is not None: kwargs["memory_size"] = memory_size
            if batch_size is not None: kwargs["batch_size"] = batch_size
            if tau is not None: kwargs["tau"] = tau
            if dropout is not None: kwargs["dropout"] = dropout
            if device is not None: kwargs["device"] = device
            if learning_starts is not None: kwargs["learning_starts"] = learning_starts
            if train_every is not None: kwargs["train_every"] = train_every
            if gradient_steps is not None: kwargs["gradient_steps"] = gradient_steps
            if epsilon_decay_steps is not None: kwargs["epsilon_decay_steps"] = epsilon_decay_steps
            config = LegacyAgentConfig(**kwargs).to_agent_config() if kwargs else AgentConfig()
        elif not isinstance(config, AgentConfig):
            config = AgentConfig()

        self.config = config

        # Model and trainer
        self.model = QNet(config.network)
        self.trainer = QTrainer(self.model, config)

        # Hyperparameters (exposed for backward compat)
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.batch_size = config.batch_size
        self.tau = config.tau
        self.dropout = config.network.dropout

        # Replay schedule (unified)
        self.learning_starts = config.learning_starts
        self.train_every = config.train_every
        self.gradient_steps = config.gradient_steps
        self.epsilon_decay_steps = config.epsilon_decay_steps
        self._transitions_since_train = 0

        # Memory
        self.memory = ReplayBuffer(
            capacity=config.memory_size,
            state_dim=config.network.input_size,
            n_step=config.n_step,
            gamma=config.gamma,
        )

        # Training stats
        self.n_games = 0
        self.total_steps = 0
        self.high_score = 0
        self.scores_history: list[int] = []
        self.mean_scores_history: list[float] = []
        self.epsilon_history: list[float] = []
        self.loss_history: list[float] = []
        # Lifetime termination totals
        self.termination_history: dict[str, int] = {
            "wall": 0,
            "self": 0,
            "timeout": 0,
            "victory": 0,
        }
        # Recent episode outcomes for display (bounded window)
        from collections import deque

        self.recent_terminations: deque[str] = deque(maxlen=50)

        # Bellman update details for visualization (last sampled replay transition)
        self._last_bellman_details: dict | None = None

        # For step-by-step mode
        self.last_state: np.ndarray | None = None
        self.last_action: int | None = None

    def _select_action(
        self, state: np.ndarray, training: bool = True, compute_q_values: bool = True
    ) -> tuple[int, str, np.ndarray | None]:
        """
        Core epsilon-greedy action selection.

        Args:
            state: Current observation (state_dim,)
            training: If True, use epsilon-greedy. If False, greedy.
            compute_q_values: If False, skip Q-value computation (for fast training).
                             Returns None for q_values.

        Returns:
            Tuple of (action_index, reason, q_values_or_None)
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            if compute_q_values:
                q_values = self.trainer.get_q_values(state)
            else:
                q_values = None
            return random.randint(0, 2), "exploring", q_values

        # Exploitation: greedy action (always needs Q-values)
        q_values = self.trainer.get_q_values(state)
        return int(np.argmax(q_values)), "exploiting", q_values

    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: Current observation (state_dim,)
            training: If True, use epsilon-greedy. If False, greedy.

        Returns:
            Action index (0, 1, 2)
        """
        action, _, _ = self._select_action(state, training)
        return action

    def get_action_with_reason(self, state: np.ndarray, training: bool = True) -> tuple[int, str]:
        """
        Epsilon-greedy action selection with reason.

        Args:
            state: Current observation (state_dim,)
            training: If True, use epsilon-greedy. If False, greedy.

        Returns:
            Tuple of (action_index, reason) where reason is "exploring", "exploiting", or "manual"
        """
        action, reason, _ = self._select_action(state, training)
        return action, reason

    def get_action_with_q_values(
        self, state: np.ndarray, training: bool = True, compute_q_values: bool = True
    ) -> tuple[int, str, np.ndarray | None]:
        """
        Epsilon-greedy action selection with reason and Q-values.

        Avoids duplicate forward pass when both action and Q-values are needed
        (e.g., for dashboard visualization).

        Args:
            state: Current observation (state_dim,)
            training: If True, use epsilon-greedy. If False, greedy.
            compute_q_values: If False, skip Q-value computation during exploration.

        Returns:
            Tuple of (action_index, reason, q_values_or_None)
        """
        return self._select_action(state, training, compute_q_values)

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities for visualization."""
        # Use probabilities_from_q_values to avoid duplicate forward pass
        q_values = self.trainer.get_q_values(state)
        return self.trainer.probabilities_from_q_values(q_values)

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get raw Q-values for visualization."""
        return self.trainer.get_q_values(state)

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        """Store experience in replay memory with n-step aggregation.

        Args:
            terminated: Episode ended due to game-over (suppresses bootstrap)
            truncated: Episode ended due to external limit (keeps bootstrap)
        """
        self.memory.push(state, action, reward, next_state, terminated, truncated)

    def maybe_optimize(self, capture_bellman: bool = False) -> list[float]:
        """
        Unified optimization step following replay schedule.

        Args:
            capture_bellman: If True, capture Bellman details on the last gradient step

        Returns:
            List of loss values from each gradient step (empty if no training occurred)
        """
        losses: list[float] = []
        self._last_bellman_details = None

        # Check if we should train this transition
        self._transitions_since_train += 1
        if self._transitions_since_train < self.train_every:
            return losses

        self._transitions_since_train = 0

        # Check if replay has enough samples
        min_samples = max(self.learning_starts, self.batch_size)
        if len(self.memory) < min_samples:
            return losses

        # Perform gradient_steps optimizer updates
        for i in range(self.gradient_steps):
            states, actions, rewards, next_states, terminated, truncated, discounts = self.memory.sample(
                self.batch_size
            )

            # Use torch.from_numpy for zero-copy conversion
            states_t = torch.from_numpy(states)
            actions_t = torch.from_numpy(actions.astype(np.int64))
            rewards_t = torch.from_numpy(rewards)
            next_states_t = torch.from_numpy(next_states)
            terminated_t = torch.from_numpy(terminated)
            discounts_t = torch.from_numpy(discounts)

            # Capture Bellman details on the last gradient step only if requested
            if i == self.gradient_steps - 1 and capture_bellman:
                loss, details = self.trainer.train_step(
                    states_t, actions_t, rewards_t, next_states_t, terminated_t, discounts_t,
                    capture_details=True
                )
                self._last_bellman_details = details
            else:
                loss = self.trainer.train_step(
                    states_t, actions_t, rewards_t, next_states_t, terminated_t, discounts_t,
                    capture_details=False
                )
            losses.append(loss)

        return losses

    def get_last_bellman_details(self) -> dict | None:
        """Get Bellman update details from last optimization step for visualization."""
        return self._last_bellman_details

    def decay_epsilon(self) -> None:
        """Decay exploration rate (per-episode decay - legacy)."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def decay_epsilon_by_steps(self, training_transitions: int, decay_steps: int) -> None:
        """
        Decay exploration rate based on training transition count.

        Uses linear decay from epsilon_start to epsilon_end over decay_steps.

        Args:
            training_transitions: Number of training transitions completed
            decay_steps: Total transitions over which to decay epsilon
        """
        if decay_steps <= 0:
            return
        progress = min(1.0, training_transitions / decay_steps)
        self.epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def update_stats(
        self, score: int, loss: float | None = None, termination_reason: str | None = None
    ) -> None:
        """Update training statistics.

        Args:
            score: Final score of the episode
            loss: Average loss for the episode (optional)
            termination_reason: One of 'wall', 'self', 'timeout', 'victory' (optional)
        """
        self.n_games += 1
        self.scores_history.append(score)
        self.epsilon_history.append(self.epsilon)

        # Rolling average (last 50 games)
        window = min(50, len(self.scores_history))
        avg_score = np.mean(self.scores_history[-window:])
        self.mean_scores_history.append(avg_score)

        if score > self.high_score:
            self.high_score = score

        if loss is not None:
            self.loss_history.append(loss)

        # Track termination reason - both lifetime and recent window
        if termination_reason and termination_reason in self.termination_history:
            self.termination_history[termination_reason] += 1
            self.recent_terminations.append(termination_reason)

    def get_training_stats(self) -> dict[str, Any]:
        """Get current training statistics."""
        return {
            "n_games": self.n_games,
            "total_steps": self.total_steps,
            "high_score": self.high_score,
            "current_score": self.scores_history[-1] if self.scores_history else 0,
            "epsilon": self.epsilon,
            "avg_score_50": self.mean_scores_history[-1] if self.mean_scores_history else 0,
            "avg_score_10": np.mean(self.scores_history[-10:])
            if len(self.scores_history) >= 10
            else 0,
            "memory_size": len(self.memory),
            "loss": self.loss_history[-1] if self.loss_history else 0,
        }

    def reset_stats(self) -> None:
        """Reset display statistics only (keep lifetime counters, model, replay, optimizer, epsilon).

        Clears:
        - scores_history, mean_scores_history, epsilon_history, loss_history
        - recent_terminations (recent window)

        Preserves:
        - n_games (lifetime episode count)
        - total_steps (lifetime transition count)
        - high_score (lifetime best)
        - epsilon (current exploration rate)
        - termination_history (lifetime totals)
        - trainer.loss_history, trainer.training_step
        - replay memory
        """
        self.scores_history = []
        self.mean_scores_history = []
        self.epsilon_history = []
        self.loss_history = []
        self.recent_terminations.clear()
        # Note: n_games, total_steps, high_score, epsilon are NOT reset
        # Note: termination_history is NOT reset
        # Note: trainer.loss_history and trainer.training_step are NOT reset
        # Note: replay memory is NOT cleared

    # Checkpointing delegated to persistence.CheckpointManager
    # See persistence.py for save/load methods

    def save_checkpoint(self, filepath: str, env=None, run_id: int = 0, episode_id: int = 0,
                        step_id: int = 0, replay_warmed_up: bool = False) -> None:
        """Save agent checkpoint with optional environment state for exact resume."""
        from .persistence import CheckpointManager
        manager = CheckpointManager(self, self.trainer.device)
        manager.save_full(
            filepath,
            env=env,
            run_id=run_id,
            episode_id=episode_id,
            step_id=step_id,
            replay_warmed_up=replay_warmed_up,
        )

    def load_checkpoint(self, filepath: str, env=None) -> dict[str, Any]:
        """Load agent checkpoint (backward compatible - loads training state)."""
        from .persistence import CheckpointManager
        manager = CheckpointManager(self, self.trainer.device)
        return manager.load_full(filepath, strict=False, env=env)

    def save_weights_only(self, filepath: str) -> None:
        """Save only model weights (for inference/deployment)."""
        from .persistence import CheckpointManager
        manager = CheckpointManager(self, self.trainer.device)
        manager.save_weights_only(filepath)

    def load_weights_only(self, filepath: str):
        """Load only model weights (for inference/deployment)."""
        from .persistence import CheckpointManager, ModelConfig
        manager = CheckpointManager(self, self.trainer.device)
        return manager.load_weights_only(filepath)


def evaluate_agent(
    agent: Agent, env: SnakeEnv, n_episodes: int = 10, render: bool = False
) -> dict[str, float]:
    """
    Evaluate agent performance (no exploration).

    Returns:
        Dictionary with evaluation metrics
    """
    scores = []
    steps_list = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        steps = 0

        while True:
            action = agent.get_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                scores.append(info["score"])
                steps_list.append(steps)
                break

    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "max_score": np.max(scores),
        "min_score": np.min(scores),
        "mean_steps": np.mean(steps_list),
        "scores": scores,
    }


if __name__ == "__main__":
    # Quick test
    from .config import EnvConfig

    config = AgentConfig()
    agent = Agent(config)
    env = SnakeEnv(EnvConfig(grid_size=10))

    state, _ = env.reset(seed=42)
    for _ in range(10):
        action = agent.get_action(state, training=True)
        state, reward, terminated, truncated, info = env.step(action)
        agent.remember(state, action, reward, state, terminated, truncated)
        if terminated or truncated:
            break

    print("Agent test passed!")
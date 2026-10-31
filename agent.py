"""
DQN Agent for Snake Game.

Implements experience replay, epsilon-greedy exploration,
and training loop management.
"""

import os
import random
from collections import deque
from typing import Any

import numpy as np
import torch

from model import Linear_QNet, QTrainer
from persistence import (
    CheckpointManager,
    ModelConfig,
)
from snake_env import SnakeGameAI
from observation_schema import OBSERVATION_SIZE


class ReplayMemory:
    """Compact Experience Replay Buffer using preallocated arrays.

    Stores terminated flag separately from truncated for correct
    bootstrapping in Q-learning (terminated suppresses bootstrap).
    
    Uses circular buffer with preallocated numpy arrays for efficient
    storage and sampling.
    """

    def __init__(self, capacity: int = 100_000, state_dim: int = OBSERVATION_SIZE):
        self.capacity = capacity
        self.state_dim = state_dim
        self.size = 0
        self.position = 0

        # Preallocate arrays
        self.states: np.ndarray = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self.rewards: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.next_states: np.ndarray = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminated: np.ndarray = np.zeros(capacity, dtype=bool)
        self.truncated: np.ndarray = np.zeros(capacity, dtype=bool)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Add experience to memory.

        Args:
            terminated: True if episode ended due to game-over (wall/self collision)
            truncated: True if episode ended due to external limit (timeout)
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.terminated[self.position] = terminated
        self.truncated[self.position] = truncated

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Sample random batch from memory.

        Returns:
            states, actions, rewards, next_states, terminated, truncated
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminated[indices],
            self.truncated[indices],
        )

    def __len__(self) -> int:
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


class Agent:
    """
    DQN Agent for Snake.

    Features:
    - Experience Replay
    - Epsilon-Greedy Exploration
    - Target Network
    - Training metrics tracking
    """

    def __init__(
        self,
        input_size: int = OBSERVATION_SIZE,
        hidden_size: int = 256,
        output_size: int = 3,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 100_000,
        batch_size: int = 256,
        tau: float = 0.005,
        dropout: float = 0.1,
        device: str = "auto",
        learning_starts: int = 1000,
        train_every: int = 1,
        gradient_steps: int = 1,
        epsilon_decay_steps: int = 0,  # If > 0, use step-based epsilon decay
    ):
        # Model and trainer
        self.model = Linear_QNet(input_size, hidden_size, output_size, dropout=dropout)
        self.trainer = QTrainer(self.model, lr=lr, gamma=gamma, tau=tau, device=device)

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.dropout = dropout

        # Replay schedule (unified)
        self.learning_starts = learning_starts
        self.train_every = train_every
        self.gradient_steps = gradient_steps
        self.epsilon_decay_steps = epsilon_decay_steps
        self._transitions_since_train = 0

        # Memory
        self.memory = ReplayMemory(memory_size, state_dim=input_size)

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
            state: Current observation (11,)
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
            state: Current observation (11,)
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
            state: Current observation (11,)
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
            state: Current observation (11,)
            training: If True, use epsilon-greedy. If False, greedy.
            compute_q_values: If False, skip Q-value computation during exploration.

        Returns:
            Tuple of (action_index, reason, q_values_or_None)
        """
        return self._select_action(state, training, compute_q_values)

    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities for visualization."""
        return self.trainer.get_action_probabilities(state)  # type: ignore

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get raw Q-values for visualization."""
        return self.trainer.get_q_values(state)  # type: ignore

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool = False,
    ) -> None:
        """Store experience in replay memory.

        Args:
            terminated: Episode ended due to game-over (suppresses bootstrap)
            truncated: Episode ended due to external limit (keeps bootstrap)
        """
        self.memory.push(state, action, reward, next_state, terminated, truncated)

    def train_short_memory(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> float:
        """Train on single step (for immediate feedback).

        Args:
            terminated: True if episode ended due to game-over (suppresses bootstrap)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action_t = torch.LongTensor([action])
        reward_t = torch.FloatTensor([reward])
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        terminated_t = torch.BoolTensor([terminated])

        loss = self.trainer.train_step(state_t, action_t, reward_t, next_state_t, terminated_t)
        return loss  # type: ignore

    def train_long_memory(self) -> float | None:
        """Train on batch from replay memory (legacy method, use maybe_optimize for unified schedule)."""
        if not self.memory.is_ready(self.batch_size):
            return None

        states, actions, rewards, next_states, terminated, truncated = self.memory.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        terminated_t = torch.BoolTensor(terminated)

        loss = self.trainer.train_step(states_t, actions_t, rewards_t, next_states_t, terminated_t)
        return loss  # type: ignore

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

        # Check if replay has enough samples (need at least max(learning_starts, batch_size))
        min_samples = max(self.learning_starts, self.batch_size)
        if len(self.memory) < min_samples:
            return losses

        # Perform gradient_steps optimizer updates
        for i in range(self.gradient_steps):
            states, actions, rewards, next_states, terminated, truncated = self.memory.sample(
                self.batch_size
            )

            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards)
            next_states_t = torch.FloatTensor(next_states)
            terminated_t = torch.BoolTensor(terminated)

            # Capture Bellman details on the last gradient step only if requested
            if i == self.gradient_steps - 1 and capture_bellman:
                loss, details = self.trainer.train_step(
                    states_t, actions_t, rewards_t, next_states_t, terminated_t,
                    capture_details=True
                )
                self._last_bellman_details = details
            else:
                loss = self.trainer.train_step(
                    states_t, actions_t, rewards_t, next_states_t, terminated_t,
                    capture_details=False
                )
            losses.append(loss)  # type: ignore

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

    def save_checkpoint(self, filepath: str, env=None, run_id: int = 0, episode_id: int = 0,
                       step_id: int = 0, replay_warmed_up: bool = False) -> None:
        """Save agent checkpoint with optional environment state for exact resume.

        Args:
            filepath: Path to save checkpoint
            env: Optional environment instance for mid-episode resume support.
                 If provided, captures environment state and local RNG.
            run_id: Current run ID from controller
            episode_id: Current episode ID from controller
            step_id: Current step ID from controller
            replay_warmed_up: Whether replay buffer is ready
        """
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
        """Load agent checkpoint (strict - validates architecture and schema)."""
        manager = CheckpointManager(self, self.trainer.device)
        return manager.load_full(filepath, strict=True, env=env)  # type: ignore

    def save_weights_only(self, filepath: str) -> None:
        """Save only model weights (for inference/deployment)."""
        manager = CheckpointManager(self, self.trainer.device)
        manager.save_weights_only(filepath)

    def load_weights_only(self, filepath: str) -> ModelConfig:
        """Load only model weights (for inference/deployment)."""
        manager = CheckpointManager(self, self.trainer.device)
        return manager.load_weights_only(filepath)

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

    def reset_display_stats(self) -> None:
        """Alias for reset_stats for clarity."""
        self.reset_stats()


def train_agent(
    agent: Agent,
    env: SnakeGameAI,
    n_episodes: int = 1000,
    max_steps_per_episode: int = 2000,
    save_every: int = 100,
    save_path: str = "checkpoints/snake_dqn.pth",
    callback=None,
) -> dict[str, list]:
    """
    Main training loop (backward-compatible wrapper using TrainingController).

    Args:
        agent: DQN Agent
        env: Snake Environment
        n_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        save_every: Save checkpoint every N episodes
        save_path: Path to save checkpoints
        callback: Optional callback function(episode, stats) for progress tracking

    Returns:
        Dictionary with training history
    """
    from training import TrainingController, train_headless, ActivityMode

    # Create controller with agent's config
    controller = TrainingController()
    controller._agent = agent
    controller._env = env
    controller.set_mode(ActivityMode.TRAIN)
    controller.set_paused(False)

    return train_headless(
        controller=controller,
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps_per_episode,
        save_every=save_every,
        save_path=save_path,
        callback=callback,
    )


def evaluate_agent(
    agent: Agent, env: SnakeGameAI, n_episodes: int = 10, render: bool = False
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
    print("Test passed!")

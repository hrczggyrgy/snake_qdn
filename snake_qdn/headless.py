"""
Headless training loop for non-interactive runs.

Provides a simple training function that can be used from scripts
or the command line without any UI dependencies.
"""

from typing import Any, Callable

import numpy as np

from .agent import Agent
from .config import AgentConfig, EnvConfig, TrainingConfig
from .controller import TrainingController
from .events import ActivityMode
from .env import SnakeEnv


def train_headless(
    controller: TrainingController,
    n_episodes: int = 1000,
    max_steps_per_episode: int = 2000,
    save_every: int = 100,
    save_path: str = "checkpoints/snake_dqn.pth",
    callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, list]:
    """
    Main training loop (backward-compatible wrapper using TrainingController).

    Args:
        controller: TrainingController with agent and env configured
        n_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        save_every: Save checkpoint every N episodes
        save_path: Path to save checkpoints
        callback: Optional callback function(episode, stats) for progress tracking

    Returns:
        Dictionary with training history
    """
    controller.set_mode(ActivityMode.TRAIN)
    controller.set_paused(False)

    scores = []
    mean_scores = []
    epsilons = []
    losses = []

    # Ensure we start with a fresh board
    controller.restart_board()

    for episode in range(n_episodes):
        episode_steps = 0
        episode_losses = []

        while True:
            snap = controller.advance_tick(execution_policy="FastTraining")

            episode_steps += 1
            if snap.bellman_details is not None:
                episode_losses.append(snap.bellman_details.get("td_error", 0))

            if snap.game_over or episode_steps >= max_steps_per_episode:
                break

        # Collect stats
        agent = controller.agent
        scores.append(agent.scores_history[-1] if agent.scores_history else 0)
        mean_scores.append(agent.mean_scores_history[-1] if agent.mean_scores_history else 0)
        epsilons.append(agent.epsilon)
        losses.append(np.mean(episode_losses) if episode_losses else 0)

        # Callback
        if callback:
            callback(episode, controller.get_metrics())

        # Save checkpoint
        if save_every > 0 and (episode + 1) % save_every == 0:
            controller.save_checkpoint(save_path)
            print(f"Checkpoint saved at episode {episode + 1}")

        # Auto-restart handled by controller in TRAIN mode

    # Final save
    controller.save_checkpoint(save_path)

    return {
        "scores": scores,
        "mean_scores": mean_scores,
        "epsilons": epsilons,
        "losses": losses,
    }


def train_from_config(
    env_config: EnvConfig | None = None,
    agent_config: AgentConfig | None = None,
    training_config: TrainingConfig | None = None,
    n_episodes: int = 1000,
    max_steps_per_episode: int = 2000,
    save_every: int = 100,
    save_path: str = "checkpoints/snake_dqn.pth",
    callback: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, list]:
    """
    Train from config objects (creates controller internally).

    Args:
        env_config: Environment configuration
        agent_config: Agent configuration
        training_config: Training configuration
        n_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        save_every: Save checkpoint every N episodes
        save_path: Path to save checkpoints
        callback: Optional callback function(episode, stats)

    Returns:
        Dictionary with training history
    """
    controller = TrainingController(
        env_config=env_config,
        agent_config=agent_config,
    )
    return train_headless(
        controller=controller,
        n_episodes=n_episodes,
        max_steps_per_episode=max_steps_per_episode,
        save_every=save_every,
        save_path=save_path,
        callback=callback,
    )


def evaluate_policy(
    agent: Agent,
    env: SnakeEnv,
    n_episodes: int = 10,
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
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "max_score": int(np.max(scores)),
        "min_score": int(np.min(scores)),
        "mean_steps": float(np.mean(steps_list)),
        "scores": scores,
    }


if __name__ == "__main__":
    # Quick test
    from .config import EnvConfig, AgentConfig

    print("Testing headless training...")

    controller = TrainingController(
        env_config=EnvConfig(grid_size=10),
        agent_config=AgentConfig(),
    )

    history = train_headless(
        controller=controller,
        n_episodes=10,
        max_steps_per_episode=200,
        save_every=0,
    )

    print(f"Scores: {history['scores']}")
    print(f"Mean scores: {history['mean_scores']}")
    print("Headless training test passed!")
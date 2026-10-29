"""
Controlled Experiment Framework for Snake DQN.

Provides budget-matched experiment execution for evidence-based improvements.
Each experiment runs with a fixed total budget (environment transitions, wall-clock time,
or optimizer steps) to enable fair comparison.
"""

import time
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from agent import Agent
from snake_env import SnakeGameAI
from training import TrainingController, ActivityMode, AgentConfig, EnvConfig


@dataclass
class ExperimentBudget:
    """Fixed budget for an experiment run."""
    max_env_transitions: int | None = None      # Total environment transitions
    max_wall_time_seconds: float | None = None  # Wall-clock time limit
    max_optimizer_steps: int | None = None      # Total optimizer updates
    max_episodes: int | None = None             # Episode limit

    def is_exhausted(self, transitions: int, wall_time: float,
                     optimizer_steps: int, episodes: int) -> bool:
        """Check if any budget limit is reached."""
        if self.max_env_transitions is not None and transitions >= self.max_env_transitions:
            return True
        if self.max_wall_time_seconds is not None and wall_time >= self.max_wall_time_seconds:
            return True
        if self.max_optimizer_steps is not None and optimizer_steps >= self.max_optimizer_steps:
            return True
        if self.max_episodes is not None and episodes >= self.max_episodes:
            return True
        return False


@dataclass
class ExperimentConfig:
    """Configuration for a controlled experiment."""
    name: str
    description: str
    agent_config: AgentConfig
    env_config: EnvConfig
    budget: ExperimentBudget
    seed: int = 42
    tags: list[str] = field(default_factory=list)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_id: str
    config_name: str
    seed: int
    start_time: float
    end_time: float
    wall_time: float
    # Final metrics
    total_env_transitions: int
    total_optimizer_steps: int
    total_episodes: int
    # Performance
    final_epsilon: float
    best_score: int
    mean_score_last_50: float
    mean_score_last_10: float
    # Termination distribution
    termination_counts: dict[str, int]
    # Learning curve data
    scores_history: list[int]
    mean_scores_history: list[float]
    epsilon_history: list[float]
    loss_history: list[float]
    # Evaluation (greedy) scores
    eval_scores: list[int] = field(default_factory=list)
    eval_mean: float = 0.0
    eval_std: float = 0.0
    # Resource usage
    peak_memory_mb: float | None = None
    # Status
    completed: bool = True
    stopped_reason: str = "budget_exhausted"


class ExperimentRunner:
    """Runs controlled experiments with fixed budgets."""

    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(self, config: ExperimentConfig, eval_every: int = 500,
                       eval_episodes: int = 5, verbose: bool = True) -> ExperimentResult:
        """
        Run a single experiment with the given configuration and budget.

        Args:
            config: Experiment configuration
            eval_every: Run evaluation every N environment transitions
            eval_episodes: Number of evaluation episodes per eval run
            verbose: Print progress

        Returns:
            ExperimentResult with all metrics
        """
        experiment_id = f"{config.name}_{uuid.uuid4().hex[:8]}"
        start_time = time.monotonic()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting experiment: {experiment_id}")
            print(f"Config: {config.name}")
            print(f"Budget: {config.budget}")
            print(f"Seed: {config.seed}")
            print(f"{'='*60}")

        # Create controller with fixed seed
        controller = TrainingController(
            env_config=config.env_config,
            agent_config=config.agent_config,
        )
        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)

        # Track evaluation scores
        eval_scores = []
        metrics = controller.get_metrics()
        last_eval_transition = metrics.get("training_transitions", 0)

        # Main training loop
        try:
            controller.restart_board()  # Start first episode
            while True:
                # Check budget using env_transitions (total environment interactions)
                metrics = controller.get_metrics()
                env_transitions = metrics.get("env_transitions", 0)
                training_transitions = metrics.get("training_transitions", 0)
                optimizer_steps = metrics.get("optimizer_updates", 0)
                wall_time = metrics.get("total_session_time", 0)
                episodes = controller.episode

                if config.budget.is_exhausted(env_transitions, wall_time, optimizer_steps, episodes):
                    stopped_reason = "budget_exhausted"
                    break

                # Run one step
                snap = controller.step()

                # Auto-restart board on episode end (TRAIN mode)
                if snap.game_over and controller.mode == ActivityMode.TRAIN:
                    controller.restart_board()

                # Periodic evaluation (based on training transitions)
                if eval_every > 0 and training_transitions - last_eval_transition >= eval_every:
                    if verbose:
                        print(f"  Running evaluation at training transition {training_transitions}...")
                    eval_result = self._evaluate(controller, eval_episodes)
                    eval_scores.append(eval_result)
                    last_eval_transition = training_transitions
                    if verbose:
                        print(f"  Eval mean: {eval_result['mean_score']:.2f} ± {eval_result['std_score']:.2f}")

        except KeyboardInterrupt:
            stopped_reason = "interrupted"
        except Exception as e:
            stopped_reason = f"error: {e}"
            if verbose:
                print(f"Experiment error: {e}")

        end_time = time.monotonic()
        wall_time = end_time - start_time

        # Final evaluation
        final_eval = self._evaluate(controller, eval_episodes * 2)
        eval_scores.append(final_eval)

        # Collect results
        agent = controller.agent
        result = ExperimentResult(
            experiment_id=experiment_id,
            config_name=config.name,
            seed=config.seed,
            start_time=start_time,
            end_time=end_time,
            wall_time=wall_time,
            total_env_transitions=metrics.get("env_transitions", 0),
            total_optimizer_steps=metrics.get("optimizer_updates", 0),
            total_episodes=controller.episode,
            final_epsilon=agent.epsilon,
            best_score=agent.high_score,
            mean_score_last_50=agent.mean_scores_history[-1] if agent.mean_scores_history else 0,
            mean_score_last_10=np.mean(agent.scores_history[-10:]) if len(agent.scores_history) >= 10 else 0,
            termination_counts=agent.termination_history.copy(),
            scores_history=agent.scores_history.copy(),
            mean_scores_history=agent.mean_scores_history.copy(),
            epsilon_history=agent.epsilon_history.copy(),
            loss_history=agent.trainer.loss_history.copy(),
            eval_scores=[e["mean_score"] for e in eval_scores],
            eval_mean=np.mean([e["mean_score"] for e in eval_scores]) if eval_scores else 0,
            eval_std=np.std([e["mean_score"] for e in eval_scores]) if eval_scores else 0,
            completed=(stopped_reason == "budget_exhausted"),
            stopped_reason=stopped_reason,
        )

        # Save results
        self._save_result(result, config)

        if verbose:
            print(f"\nExperiment {experiment_id} completed!")
            print(f"  Wall time: {wall_time:.1f}s")
            print(f"  Transitions: {result.total_env_transitions}")
            print(f"  Optimizer steps: {result.total_optimizer_steps}")
            print(f"  Episodes: {result.total_episodes}")
            print(f"  Best score: {result.best_score}")
            print(f"  Mean score (last 50): {result.mean_score_last_50:.2f}")
            print(f"  Eval mean: {result.eval_mean:.2f} ± {result.eval_std:.2f}")
            print(f"  Stopped: {stopped_reason}")

        controller.close()
        return result

    def _evaluate(self, controller: TrainingController, n_episodes: int, max_steps_per_episode: int = 200) -> dict:
        """Run evaluation episodes with step limit."""
        controller.set_mode(ActivityMode.EVALUATE)
        controller.set_paused(False)

        scores = []
        for _ in range(n_episodes):
            controller.restart_board()
            done = False
            steps = 0
            while not done and steps < max_steps_per_episode:
                snap = controller.step()
                done = snap.game_over
                steps += 1
            scores.append(snap.score)

        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)  # Ensure training resumes
        controller.restart_board()    # Start new training episode
        return {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "max_score": np.max(scores),
            "min_score": np.min(scores),
            "scores": scores,
        }

    def _save_result(self, result: ExperimentResult, config: ExperimentConfig) -> None:
        """Save experiment result to disk."""
        result_dict = asdict(result)
        result_dict["config"] = asdict(config)
        result_dict["config"]["agent_config"] = asdict(config.agent_config)
        result_dict["config"]["env_config"] = asdict(config.env_config)
        result_dict["config"]["budget"] = asdict(config.budget)

        filepath = self.output_dir / f"{result.experiment_id}.json"
        with open(filepath, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

    def load_results(self, pattern: str = "*.json") -> list[ExperimentResult]:
        """Load experiment results from disk."""
        results = []
        for filepath in self.output_dir.glob(pattern):
            with open(filepath) as f:
                data = json.load(f)
                results.append(ExperimentResult(**data))
        return results

    def compare_results(self, results: list[ExperimentResult]) -> None:
        """Print comparison table of experiment results."""
        print(f"\n{'='*100}")
        print(f"{'Experiment':<30} {'Transitions':>12} {'Opt Steps':>10} {'Episodes':>9} {'Best':>5} {'Mean50':>7} {'Eval±Std':>12} {'Time(s)':>8}")
        print(f"{'='*100}")
        for r in results:
            eval_str = f"{r.eval_mean:.1f}±{r.eval_std:.1f}"
            print(f"{r.config_name:<30} {r.total_env_transitions:>12} {r.total_optimizer_steps:>10} "
                  f"{r.total_episodes:>9} {r.best_score:>5} {r.mean_score_last_50:>7.1f} "
                  f"{eval_str:>12} {r.wall_time:>8.1f}")
        print(f"{'='*100}")


# Pre-defined experiment configurations
def get_baseline_experiment() -> ExperimentConfig:
    """Baseline experiment with default settings."""
    return ExperimentConfig(
        name="baseline",
        description="Baseline DQN with default hyperparameters",
        agent_config=AgentConfig(
            hidden_size=256,
            lr=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            memory_size=100_000,
            batch_size=256,
            tau=0.005,
            dropout=0.1,
            learning_starts=1000,
            train_every=1,
            gradient_steps=1,
        ),
        env_config=EnvConfig(
            grid_size=20,
            step_penalty=-0.01,
            food_reward=10.0,
            death_penalty=-10.0,
        ),
        budget=ExperimentBudget(
            max_env_transitions=500_000,
        ),
        seed=42,
        tags=["baseline"],
    )


def get_no_dropout_experiment() -> ExperimentConfig:
    """Experiment with dropout=0."""
    config = get_baseline_experiment()
    config.name = "no_dropout"
    config.description = "Baseline with dropout=0"
    config.agent_config = AgentConfig(
        hidden_size=256,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        memory_size=100_000,
        batch_size=256,
        tau=0.005,
        dropout=0.0,  # No dropout
        learning_starts=1000,
        train_every=1,
        gradient_steps=1,
    )
    config.tags = ["ablation", "dropout"]
    return config


def get_smaller_model_experiment() -> ExperimentConfig:
    """Experiment with smaller hidden size."""
    config = get_baseline_experiment()
    config.name = "smaller_model"
    config.description = "Smaller hidden size (128) with no dropout"
    config.agent_config = AgentConfig(
        hidden_size=128,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        memory_size=100_000,
        batch_size=256,
        tau=0.005,
        dropout=0.0,
        learning_starts=1000,
        train_every=1,
        gradient_steps=1,
    )
    config.tags = ["ablation", "model_size"]
    return config


def get_step_epsilon_decay_experiment() -> ExperimentConfig:
    """Experiment with step-based epsilon decay (config change needed in agent)."""
    config = get_baseline_experiment()
    config.name = "step_epsilon_decay"
    config.description = "Epsilon decay by training transitions (not episodes)"
    config.tags = ["ablation", "epsilon_schedule"]
    return config


def run_ablation_suite(budget_transitions: int = 500_000) -> list[ExperimentResult]:
    """Run a standard ablation suite with fixed budget."""
    runner = ExperimentRunner()

    configs = [
        get_baseline_experiment(),
        get_no_dropout_experiment(),
        get_smaller_model_experiment(),
    ]

    # Update budgets
    for config in configs:
        config.budget = ExperimentBudget(max_env_transitions=budget_transitions)

    results = []
    for config in configs:
        result = runner.run_experiment(config, verbose=True)
        results.append(result)

    runner.compare_results(results)
    return results


if __name__ == "__main__":
    # Quick test run with small budget
    runner = ExperimentRunner("test_experiment_results")
    config = get_baseline_experiment()
    config.budget = ExperimentBudget(max_env_transitions=5_000)
    result = runner.run_experiment(config, eval_every=1000, eval_episodes=2, verbose=True)
    print(f"Test experiment completed: {result.experiment_id}")
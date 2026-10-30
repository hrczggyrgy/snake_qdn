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
import torch

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
class EvaluationRecord:
    """Single evaluation record at a training checkpoint."""
    training_transition: int
    episode_seeds: list[int]
    episode_scores: list[int]
    mean_score: float
    std_score: float


@dataclass
class FinalEvaluation:
    """Final evaluation of the trained policy."""
    episode_seeds: list[int]
    episode_scores: list[int]
    mean_score: float
    std_score: float
    completed_episodes: int
    cutoff_reason: str = "completed"


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
    total_training_transitions: int
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
    # Evaluation history (periodic evaluations during training)
    evaluation_history: list[EvaluationRecord] = field(default_factory=list)
    # Final evaluation (on final policy)
    final_evaluation: FinalEvaluation | None = None
    # Resource usage
    peak_memory_mb: float | None = None
    # Timing (with defaults)
    training_time: float = 0.0
    evaluation_time: float = 0.0
    # Status
    completed: bool = True
    stopped_reason: str = "budget_exhausted"
    
    # Backward compatibility properties
    @property
    def eval_scores(self) -> list[int]:
        """Backward compat: list of periodic evaluation means."""
        return [e.mean_score for e in self.evaluation_history]
    
    @property
    def eval_mean(self) -> float:
        """Backward compat: mean of periodic evaluation means."""
        if not self.evaluation_history:
            return 0.0
        return np.mean([e.mean_score for e in self.evaluation_history])
    
    @property
    def eval_std(self) -> float:
        """Backward compat: std of periodic evaluation means."""
        if not self.evaluation_history:
            return 0.0
        return np.std([e.mean_score for e in self.evaluation_history])
    
    @property
    def final_eval_mean(self) -> float:
        """Backward compat: final evaluation mean."""
        return self.final_evaluation.mean_score if self.final_evaluation else 0.0
    
    @property
    def final_eval_std(self) -> float:
        """Backward compat: final evaluation std."""
        return self.final_evaluation.std_score if self.final_evaluation else 0.0


class ExperimentRunner:
    """Runs controlled experiments with fixed budgets."""

    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _derive_seeds(self, base_seed: int, n_training: int = 1, n_eval_episodes: int = 10) -> tuple[int, list[int]]:
        """Derive training and evaluation seeds from base experiment seed.
        
        Uses a deterministic derivation so the same base seed always produces
        the same training seed and evaluation seed list.
        """
        # Create a local RNG from the base seed
        rng = np.random.default_rng(base_seed)
        
        # Training environment seed
        train_seed = int(rng.integers(0, 2**31 - 1))
        
        # Evaluation episode seeds (for reproducible final evaluation)
        eval_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_eval_episodes)]
        
        return train_seed, eval_seeds

    def evaluate_policy(
        self,
        agent: Agent,
        env_config: EnvConfig,
        n_episodes: int = 5,
        max_steps_per_episode: int = 200,
        episode_seeds: list[int] | None = None,
    ) -> EvaluationRecord:
        """
        Evaluate a policy in isolation without affecting training state.
        
        Args:
            agent: The agent to evaluate (uses greedy policy)
            env_config: Environment configuration
            n_episodes: Number of evaluation episodes
            max_steps_per_episode: Step limit per episode
            episode_seeds: Optional list of seeds for each episode
            
        Returns:
            EvaluationRecord with scores and statistics
        """
        # Use provided seeds or generate deterministic ones
        if episode_seeds is None:
            episode_seeds = [None] * n_episodes
        
        # Create evaluation environment (no rendering)
        # EnvConfig doesn't have allow_tail_entry, use default (True)
        eval_env = SnakeGameAI(
            grid_size=env_config.grid_size,
            step_penalty=env_config.step_penalty,
            food_reward=env_config.food_reward,
            death_penalty=env_config.death_penalty,
            render_mode=None,  # No rendering for evaluation
            seed=None,  # We'll seed each episode explicitly
        )
        
        # Save model mode and set to eval
        was_training = agent.model.training
        agent.model.eval()
        
        scores = []
        try:
            for i in range(n_episodes):
                # Seed this episode's environment
                ep_seed = episode_seeds[i] if i < len(episode_seeds) else None
                if ep_seed is not None:
                    eval_env.seed(ep_seed)
                
                state, _ = eval_env.reset()
                done = False
                steps = 0
                
                while not done and steps < max_steps_per_episode:
                    action = agent.get_action(state, training=False)
                    state, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    steps += 1
                
                scores.append(info["score"])
        finally:
            # Always restore model mode and close environment
            agent.model.train(was_training)
            eval_env.close()
        
        return EvaluationRecord(
            training_transition=0,  # Set by caller
            episode_seeds=episode_seeds[:n_episodes],
            episode_scores=scores,
            mean_score=float(np.mean(scores)) if scores else 0.0,
            std_score=float(np.std(scores)) if scores else 0.0,
        )

    def run_experiment(self, config: ExperimentConfig, eval_every: int = 500,
                       eval_episodes: int = 5, verbose: bool = True) -> ExperimentResult:
        """
        Run a single experiment with the given configuration and budget.

        Args:
            config: Experiment configuration
            eval_every: Run evaluation every N training transitions
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

        # Derive seeds for training and evaluation
        train_seed, final_eval_seeds = self._derive_seeds(config.seed, n_eval_episodes=eval_episodes * 2)
        
        # Apply seed for reproducibility (global RNGs)
        self._apply_seed(config.seed)

        # Create controller with seeded training environment
        env_config = config.env_config
        # Set the training environment seed
        env_config = EnvConfig(
            grid_size=env_config.grid_size,
            step_penalty=env_config.step_penalty,
            food_reward=env_config.food_reward,
            death_penalty=env_config.death_penalty,
            render_mode=env_config.render_mode,
            seed=train_seed,
        )
        
        controller = TrainingController(
            env_config=env_config,
            agent_config=config.agent_config,
        )
        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)

        # Track evaluations
        evaluation_history: list[EvaluationRecord] = []
        metrics = controller.get_metrics()
        last_eval_transition = metrics.get("training_transitions", 0)
        
        # For periodic evaluation seeds, derive from experiment seed
        periodic_eval_seeds = self._derive_seeds(config.seed + 1, n_eval_episodes=eval_episodes * 10)[1]

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
                    
                    eval_start = time.monotonic()
                    # Get seeds for this periodic evaluation
                    ep_seeds = periodic_eval_seeds[:eval_episodes]
                    periodic_eval_seeds = periodic_eval_seeds[eval_episodes:]
                    
                    eval_record = self.evaluate_policy(
                        agent=controller.agent,
                        env_config=config.env_config,  # Use same config but eval env will be separate
                        n_episodes=eval_episodes,
                        episode_seeds=ep_seeds,
                    )
                    eval_record.training_transition = training_transitions
                    evaluation_history.append(eval_record)
                    last_eval_transition = training_transitions
                    
                    eval_time = time.monotonic() - eval_start
                    
                    if verbose:
                        print(f"  Eval mean: {eval_record.mean_score:.2f} ± {eval_record.std_score:.2f} "
                              f"(time: {eval_time:.2f}s)")

        except KeyboardInterrupt:
            stopped_reason = "interrupted"
        except Exception as e:
            stopped_reason = f"error: {e}"
            if verbose:
                print(f"Experiment error: {e}")

        end_time = time.monotonic()
        wall_time = end_time - start_time

        # Final evaluation
        if verbose:
            print(f"  Running final evaluation...")
        eval_start = time.monotonic()
        final_eval_record = self.evaluate_policy(
            agent=controller.agent,
            env_config=config.env_config,
            n_episodes=eval_episodes * 2,
            episode_seeds=final_eval_seeds,
        )
        final_eval_time = time.monotonic() - eval_start
        
        final_evaluation = FinalEvaluation(
            episode_seeds=final_eval_record.episode_seeds,
            episode_scores=final_eval_record.episode_scores,
            mean_score=final_eval_record.mean_score,
            std_score=final_eval_record.std_score,
            completed_episodes=len(final_eval_record.episode_scores),
            cutoff_reason="completed",
        )
        
        if verbose:
            print(f"  Final eval: {final_evaluation.mean_score:.2f} ± {final_evaluation.std_score:.2f} "
                  f"(time: {final_eval_time:.2f}s)")

        # Collect results
        agent = controller.agent
        metrics = controller.get_metrics()
        training_time = metrics.get("total_session_time", 0.0)
        # Evaluation time = total - training (approximate)
        evaluation_time = wall_time - training_time

        result = ExperimentResult(
            experiment_id=experiment_id,
            config_name=config.name,
            seed=config.seed,
            start_time=start_time,
            end_time=end_time,
            wall_time=wall_time,
            training_time=training_time,
            evaluation_time=evaluation_time,
            total_env_transitions=metrics.get("env_transitions", 0),
            total_training_transitions=metrics.get("training_transitions", 0),
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
            evaluation_history=evaluation_history,
            final_evaluation=final_evaluation,
            completed=(stopped_reason == "budget_exhausted"),
            stopped_reason=stopped_reason,
        )

        # Save results
        self._save_result(result, config)

        if verbose:
            print(f"\nExperiment {experiment_id} completed!")
            print(f"  Wall time: {wall_time:.1f}s (training: {training_time:.1f}s, eval: {evaluation_time:.1f}s)")
            print(f"  Env transitions: {result.total_env_transitions}")
            print(f"  Training transitions: {result.total_training_transitions}")
            print(f"  Optimizer steps: {result.total_optimizer_steps}")
            print(f"  Episodes: {result.total_episodes}")
            print(f"  Best score: {result.best_score}")
            print(f"  Mean score (last 50): {result.mean_score_last_50:.2f}")
            if evaluation_history:
                print(f"  Periodic eval: {result.eval_mean:.2f} ± {result.eval_std:.2f}")
            print(f"  Final eval: {result.final_eval_mean:.2f} ± {result.final_eval_std:.2f}")
            print(f"  Stopped: {stopped_reason}")

        controller.close()
        return result
    
    def _apply_seed(self, seed: int) -> None:
        """Apply seed to all random number generators."""
        import random
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _save_result(self, result: ExperimentResult, config: ExperimentConfig) -> None:
        """Save experiment result to disk using versioned envelope."""
        envelope = {
            "schema_version": 1,
            "config": {
                **asdict(config),
                "agent_config": asdict(config.agent_config),
                "env_config": asdict(config.env_config),
                "budget": asdict(config.budget),
            },
            "result": asdict(result),
        }

        filepath = self.output_dir / f"{result.experiment_id}.json"
        with open(filepath, "w") as f:
            json.dump(envelope, f, indent=2, default=str)

    def load_results(self, pattern: str = "*.json") -> list[ExperimentResult]:
        """Load experiment results from disk (supports v1 envelope and legacy flat format)."""
        results = []
        for filepath in self.output_dir.glob(pattern):
            with open(filepath) as f:
                data = json.load(f)
            
            # Handle versioned envelope format
            if "schema_version" in data and "result" in data:
                result_data = data["result"]
            else:
                # Legacy flat format (backward compatibility)
                result_data = data
            
            # Reconstruct nested dataclasses
            if "evaluation_history" in result_data:
                result_data["evaluation_history"] = [
                    EvaluationRecord(**e) if isinstance(e, dict) else e
                    for e in result_data["evaluation_history"]
                ]
            
            if "final_evaluation" in result_data and result_data["final_evaluation"]:
                fe = result_data["final_evaluation"]
                if isinstance(fe, dict):
                    result_data["final_evaluation"] = FinalEvaluation(**fe)
            
            # Filter out unexpected fields (e.g., "config" from legacy saves)
            result_fields = {f.name for f in ExperimentResult.__dataclass_fields__.values()}
            filtered_data = {k: v for k, v in result_data.items() if k in result_fields}
            
            results.append(ExperimentResult(**filtered_data))
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
    """Experiment with step-based epsilon decay over training transitions."""
    config = get_baseline_experiment()
    config.name = "step_epsilon_decay"
    config.description = "Epsilon decay by training transitions (not episodes)"
    config.agent_config = AgentConfig(
        hidden_size=256,
        lr=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,  # Not used when epsilon_decay_steps > 0
        memory_size=100_000,
        batch_size=256,
        tau=0.005,
        dropout=0.1,
        learning_starts=1000,
        train_every=1,
        gradient_steps=1,
        epsilon_decay_steps=500_000,  # Decay over 500k training transitions
    )
    config.tags = ["ablation", "epsilon_schedule"]
    return config


def run_ablation_suite(budget_transitions: int = 500_000) -> list[ExperimentResult]:
    """Run a standard ablation suite with fixed budget."""
    runner = ExperimentRunner()

    configs = [
        get_baseline_experiment(),
        get_no_dropout_experiment(),
        get_smaller_model_experiment(),
        get_step_epsilon_decay_experiment(),
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
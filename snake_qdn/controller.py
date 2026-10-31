"""
TrainingController - Framework-independent training lifecycle management.

Owns the environment, agent, observation, mode, and episode lifecycle,
with no UI dependencies. Provides explicit operations for stepping,
restarting, clearing history, and creating new runs.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .agent import Agent
from .config import AgentConfig, EnvConfig, TrainingConfig
from .env import SnakeEnv
from .events import (
    ActivityMode,
    BellmanDetails,
    DecisionRecord,
    EventType,
    PhaseInfo,
    TrainingEvent,
    TrainingSnapshot,
)

# Module logger
logger = logging.getLogger(__name__)


class TrainingController:
    """
    Framework-independent training controller.

    Owns the complete training lifecycle: environment, agent, observations,
    mode, episode management, and metrics. Provides explicit operations for
    stepping, restarting, clearing history, and creating new runs.

    Contains NO UI imports or rendering logic.
    """

    def __init__(
        self,
        env_config: EnvConfig | None = None,
        agent_config: AgentConfig | None = None,
        training_config: TrainingConfig | None = None,
        # Backward compatibility: accept flat kwargs
        grid_size: int | None = None,
        step_penalty: float | None = None,
        food_reward: float | None = None,
        death_penalty: float | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
        allow_tail_entry: bool | None = None,
        hidden_size: int | None = None,
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
        if env_config is not None:
            self.env_config = env_config
        else:
            kwargs = {}
            if grid_size is not None: kwargs["grid_size"] = grid_size
            if step_penalty is not None: kwargs["step_penalty"] = step_penalty
            if food_reward is not None: kwargs["food_reward"] = food_reward
            if death_penalty is not None: kwargs["death_penalty"] = death_penalty
            if render_mode is not None: kwargs["render_mode"] = render_mode
            if seed is not None: kwargs["seed"] = seed
            if allow_tail_entry is not None: kwargs["allow_tail_entry"] = allow_tail_entry
            self.env_config = EnvConfig(**kwargs) if kwargs else EnvConfig()

        if agent_config is not None:
            self.agent_config = agent_config
        else:
            kwargs = {}
            if hidden_size is not None: kwargs["hidden_size"] = hidden_size
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
            self.agent_config = AgentConfig(**kwargs) if kwargs else AgentConfig()

        self.training_config = training_config or TrainingConfig()

        # Core components (lazy initialization)
        self._env: SnakeEnv | None = None
        self._agent: Agent | None = None

        # Current step state
        self._observation: np.ndarray | None = None
        self._game_state: dict[str, Any] | None = None
        self._q_values: np.ndarray | None = None
        self._action_probs: np.ndarray | None = None
        self._action: int | None = None
        self._selection_reason: str | None = None
        self._game_over: bool = True

        # Mode and control
        self._mode: ActivityMode = ActivityMode.TRAIN
        self._paused: bool = True

        # Run/episode/step identifiers for cache invalidation
        self._run_id: int = 0
        self._episode_id: int = 0
        self._step_id: int = 0

        # History revision counter for cache invalidation
        self._history_revision: int = 0

        # History windows for UI (bounded)
        self._max_history = 500
        self._loss_max_history = 1000

        # Performance metrics
        self._metrics: dict[str, float | int | bool | None] = {
            "env_transitions": 0,
            "training_transitions": 0,
            "optimizer_updates": 0,
            "episodes_completed": 0,
            "total_session_time": 0.0,
            "active_execution_time": 0.0,
            "replay_buffer_size": 0,
            "replay_active": False,
            "time_action_selection": 0.0,
            "time_env_step": 0.0,
            "time_optimization": 0.0,
            "time_diagnostics": 0.0,
            "time_display_prep": 0.0,
            "session_start_time": None,
            "last_transition_time": None,
            "_prev_optimizer_step": 0,
        }

        # Event system
        self._event_counter = 0
        self._events: list[TrainingEvent] = []
        self._max_events = 200
        self._replay_warmed_up = False

        # Execution lock: serializes ALL model/optimizer mutations
        self._lock = threading.RLock()

        # Fault state
        self._faulted: bool = False
        self._fault_error: Exception | None = None
        self._fault_traceback: str | None = None
        self._fault_tick_id: int | None = None

        # Execution tracing
        self._tick_counter: int = 0
        self._session_id: str = uuid.uuid4().hex[:8]

        # Curriculum state
        self._curriculum_stage_idx: int = 0
        self._phase: str = "warmup"

    # -------------------------------------------------------------------------
    # Properties (read-only access to internal state)
    # -------------------------------------------------------------------------

    @property
    def faulted(self) -> bool:
        """Whether the controller has faulted and needs manual recovery."""
        return self._faulted

    @property
    def fault_error(self) -> Exception | None:
        """The exception that caused the fault, if any."""
        return self._fault_error

    @property
    def session_id(self) -> str:
        """Session correlation ID for tracing."""
        return self._session_id

    @property
    def env(self) -> SnakeEnv:
        if self._env is None:
            self._env = self._create_env()
        return self._env

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    @property
    def observation(self) -> np.ndarray | None:
        return self._observation

    @property
    def game_state(self) -> dict[str, Any] | None:
        return self._game_state

    @property
    def q_values(self) -> np.ndarray | None:
        return self._q_values

    @property
    def action_probs(self) -> np.ndarray | None:
        return self._action_probs

    @property
    def action(self) -> int | None:
        return self._action

    @property
    def selection_reason(self) -> str | None:
        return self._selection_reason

    @property
    def game_over(self) -> bool:
        return self._game_over

    @property
    def mode(self) -> ActivityMode:
        return self._mode

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def epsilon(self) -> float:
        return self.agent.epsilon if self._agent else self.agent_config.epsilon_start

    @property
    def episode(self) -> int:
        """Total completed episodes (training + evaluation)."""
        return self._episode_id

    @property
    def training_episodes(self) -> int:
        """Completed training episodes (where learning occurred)."""
        return self.agent.n_games if self._agent else 0

    @property
    def evaluation_episodes(self) -> int:
        """Completed evaluation episodes (no learning)."""
        train_eps = self.agent.n_games if self._agent else 0
        return max(0, self._episode_id - train_eps)

    @property
    def total_env_steps(self) -> int:
        return self.agent.total_steps if self._agent else 0

    @property
    def optimizer_steps(self) -> int:
        return self.agent.trainer.training_step if self._agent else 0

    @property
    def score(self) -> int:
        return self._game_state["score"] if self._game_state else 0

    @property
    def high_score(self) -> int:
        return self.agent.high_score if self._agent else 0

    @property
    def run_id(self) -> int:
        return self._run_id

    @property
    def episode_id(self) -> int:
        return self._episode_id

    @property
    def step_id(self) -> int:
        return self._step_id

    @property
    def curriculum_stage(self) -> int:
        """Current curriculum stage index."""
        return self._curriculum_stage_idx

    @property
    def phase(self) -> str:
        """Current training phase."""
        return self._phase

    def get_metrics(self) -> dict[str, float | int | bool]:
        """Get current performance metrics (copy for isolation)."""
        metrics = self._metrics.copy()
        # Remove internal tracking field
        metrics.pop("_prev_optimizer_step", None)
        # Add derived rates
        elapsed_session = metrics.get("total_session_time", 0.0)
        elapsed_active = metrics.get("active_execution_time", 0.0)
        if isinstance(elapsed_session, (int, float)) and elapsed_session > 0:
            env_transitions = metrics.get("env_transitions", 0)
            training_transitions = metrics.get("training_transitions", 0)
            optimizer_updates = metrics.get("optimizer_updates", 0)
            if isinstance(env_transitions, (int, float)):
                metrics["env_transitions_per_sec_session"] = env_transitions / elapsed_session
            if isinstance(training_transitions, (int, float)):
                metrics["training_transitions_per_sec_session"] = training_transitions / elapsed_session
            if isinstance(optimizer_updates, (int, float)):
                metrics["optimizer_updates_per_sec_session"] = optimizer_updates / elapsed_session
        else:
            metrics["env_transitions_per_sec_session"] = 0.0
            metrics["training_transitions_per_sec_session"] = 0.0
            metrics["optimizer_updates_per_sec_session"] = 0.0

        if isinstance(elapsed_active, (int, float)) and elapsed_active > 0:
            env_transitions = metrics.get("env_transitions", 0)
            training_transitions = metrics.get("training_transitions", 0)
            optimizer_updates = metrics.get("optimizer_updates", 0)
            if isinstance(env_transitions, (int, float)):
                metrics["env_transitions_per_sec_active"] = env_transitions / elapsed_active
            if isinstance(training_transitions, (int, float)):
                metrics["training_transitions_per_sec_active"] = training_transitions / elapsed_active
            if isinstance(optimizer_updates, (int, float)):
                metrics["optimizer_updates_per_sec_active"] = optimizer_updates / elapsed_active
        else:
            metrics["env_transitions_per_sec_active"] = 0.0
            metrics["training_transitions_per_sec_active"] = 0.0
            metrics["optimizer_updates_per_sec_active"] = 0.0

        # Backward compatibility aliases
        metrics["transitions"] = metrics.get("env_transitions", 0)
        metrics["total_training_time"] = metrics.get("total_session_time", 0.0)
        metrics["transitions_per_sec"] = metrics.get("env_transitions_per_sec_session", 0.0)
        metrics["optimizer_updates_per_sec"] = metrics.get("optimizer_updates_per_sec_session", 0.0)

        # Remove None values
        return {k: v for k, v in metrics.items() if v is not None}

    # -------------------------------------------------------------------------
    # Event System
    # -------------------------------------------------------------------------

    def _emit_event(
        self,
        event_type: EventType,
        message: str,
        episode: int | None = None,
        step: int | None = None,
        score: int | None = None,
        termination_reason: str | None = None,
    ) -> TrainingEvent:
        """Emit an explicit training event."""
        self._event_counter += 1
        event = TrainingEvent(
            id=self._event_counter,
            type=event_type,
            message=message,
            timestamp=time.monotonic(),
            episode=episode if episode is not None else self.episode,
            step=step if step is not None else self._step_id,
            score=score,
            termination_reason=termination_reason,
            run_id=self._run_id,
            episode_id=self._episode_id,
        )
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]
        return event

    def get_events(self, since_id: int = 0) -> list[TrainingEvent]:
        """Get events newer than since_id."""
        return [e for e in self._events if e.id > since_id]

    def get_all_events(self) -> list[TrainingEvent]:
        """Get all events (copy)."""
        return list(self._events)

    def clear_events(self) -> None:
        """Clear event history."""
        self._events.clear()
        self._event_counter = 0

    # -------------------------------------------------------------------------
    # Component Creation
    # -------------------------------------------------------------------------

    def _create_env(self) -> SnakeEnv:
        return SnakeEnv(config=self.env_config)

    def _create_agent(self) -> Agent:
        return Agent(config=self.agent_config)

    # -------------------------------------------------------------------------
    # Public API - Lifecycle Operations
    # -------------------------------------------------------------------------

    def set_mode(self, mode: ActivityMode) -> None:
        """Set activity mode (TRAIN/EVALUATE/MANUAL)."""
        with self._lock:
            old_mode = self._mode
            self._mode = mode
            if old_mode != mode:
                self._emit_event(
                    EventType.MODE_CHANGE,
                    f"Mode changed: {old_mode.value} → {mode.value}",
                )

    def set_paused(self, paused: bool) -> None:
        """Set paused state."""
        with self._lock:
            self._paused = paused

    def restart_board(self) -> None:
        """Restart current game episode, keep learned weights and training history."""
        with self._lock:
            _ = self.agent  # Ensure agent exists
            self._reset_episode()
            self._emit_event(EventType.BOARD_RESTART, "Board restarted")

    def clear_history(self) -> None:
        """Clear displayed charts and statistics (display only)."""
        with self._lock:
            if self._agent:
                self._agent.reset_stats()
            self._episode_id = 0
            self._step_id = 0
            self.clear_events()
            self._emit_event(EventType.HISTORY_CLEAR, "Display history cleared (training state preserved)")

    def _reset_metrics(self) -> None:
        """Reset all run-scoped metrics."""
        self._metrics = {
            "env_transitions": 0,
            "training_transitions": 0,
            "optimizer_updates": 0,
            "episodes_completed": 0,
            "total_session_time": 0.0,
            "active_execution_time": 0.0,
            "replay_buffer_size": 0,
            "replay_active": False,
            "time_action_selection": 0.0,
            "time_env_step": 0.0,
            "time_optimization": 0.0,
            "time_diagnostics": 0.0,
            "time_display_prep": 0.0,
            "session_start_time": None,
            "last_transition_time": None,
            "_prev_optimizer_step": 0,
        }

    def new_run(self) -> None:
        """Create fresh agent and replay buffer, discard current model."""
        with self._lock:
            self._agent = self._create_agent()
            self._env = self._create_env()
            self._run_id += 1
            self._episode_id = 0
            self._step_id = 0
            self._curriculum_stage_idx = 0
            self._phase = "warmup"
            self._replay_warmed_up = False
            self._reset_metrics()
            self.clear_events()
            self._reset_episode()
            self._emit_event(EventType.NEW_RUN, "New training run started")

    def update_env_config(self, config: EnvConfig) -> None:
        """Update environment configuration and recreate env/agent."""
        with self._lock:
            self.env_config = config
            self._env = self._create_env()
            self._agent = self._create_agent()
            self._run_id += 1
            self._episode_id = 0
            self._step_id = 0
            self._curriculum_stage_idx = 0
            self._phase = "warmup"
            self._replay_warmed_up = False
            self._reset_metrics()
            self.clear_events()
            self._reset_episode()
            self._emit_event(EventType.MODE_CHANGE, "Environment config updated — new run")

    def update_agent_config(self, config: AgentConfig) -> None:
        """Update agent configuration and recreate agent."""
        with self._lock:
            self.agent_config = config
            self._agent = self._create_agent()
            self._reset_metrics()
            self._reset_episode()
            self.clear_events()
            self._emit_event(EventType.MODE_CHANGE, "Agent config updated — new agent created")

    # -------------------------------------------------------------------------
    # Curriculum
    # -------------------------------------------------------------------------

    def _check_curriculum_advance(self) -> bool:
        """Check if curriculum should advance. Returns True if advanced."""
        if not self.env_config.curriculum_enabled:
            return False

        agent = self._agent
        if agent is None:
            return False

        # Check rolling average score
        if not agent.mean_scores_history:
            return False

        current_avg = agent.mean_scores_history[-1]
        threshold = self.env_config.curriculum_threshold
        stages = self.env_config.curriculum_stages

        if current_avg >= threshold and self._curriculum_stage_idx < len(stages) - 1:
            # Advance to next stage
            self._curriculum_stage_idx += 1
            new_grid_size = stages[self._curriculum_stage_idx]
            self.env.set_grid_size(new_grid_size)
            self.env_config.grid_size = new_grid_size
            self._reset_episode()
            self._emit_event(
                EventType.CURRICULUM_ADVANCE,
                f"Curriculum advanced to {new_grid_size}×{new_grid_size} (avg score: {current_avg:.1f})",
            )
            return True
        return False

    def _update_phase(self) -> None:
        """Update training phase based on progress."""
        metrics = self.get_metrics()
        training_transitions = metrics.get("training_transitions", 0)
        learning_starts = self.agent.learning_starts if self._agent else 1000
        epsilon_decay_steps = self.agent.epsilon_decay_steps if self._agent else 50_000

        if training_transitions < learning_starts:
            self._phase = "warmup"
        elif training_transitions < learning_starts + epsilon_decay_steps:
            self._phase = "exploring"
        else:
            self._phase = "exploiting"

    # -------------------------------------------------------------------------
    # Public API - Step Operations
    # -------------------------------------------------------------------------

    def step(self, action: int | None = None, execution_policy: str = "Watch") -> TrainingSnapshot:
        """
        Execute one environment step.

        Args:
            action: Optional action for MANUAL mode. Ignored in TRAIN/EVALUATE.
            execution_policy: "Watch" or "FastTraining" - controls diagnostic level.

        Returns:
            TrainingSnapshot with pre-action decision data and post-step state.
        """
        with self._lock:
            self._execution_policy = execution_policy
            return self._step_impl(action)

    def _step_impl(self, action: int | None = None, skip_diagnostics: bool = False) -> TrainingSnapshot:
        """Internal step implementation (must hold self._lock)."""
        if self._faulted:
            raise RuntimeError(
                f"Controller faulted: {self._fault_error}. "
                "Manual recovery required (new_run or load_checkpoint)."
            )
        if self._game_over or self._env is None:
            return self._make_snapshot()

        now = time.monotonic()
        if self._metrics["session_start_time"] is None:
            self._metrics["session_start_time"] = now
            self._metrics["last_transition_time"] = now

        step_start = time.monotonic()

        # --- Get action (pre-step) ---
        t_action_start = time.monotonic()
        if action is None:
            _ = self.agent
            action, self._selection_reason, pre_q_values = self._select_action()
        elif action is not None:
            self._selection_reason = "manual"
            pre_q_values = (
                self.agent.get_q_values(self._observation)
                if self._observation is not None
                else None
            )
        else:
            action = 0
            self._selection_reason = "exploiting"
            pre_q_values = (
                self.agent.get_q_values(self._observation)
                if self._observation is not None
                else None
            )
        self._metrics["time_action_selection"] = (
            (self._metrics.get("time_action_selection") or 0.0) + time.monotonic() - t_action_start
        )

        # --- Capture PRE-ACTION decision snapshot ---
        pre_observation = self._observation.copy() if self._observation is not None else None
        pre_action_probs = None

        if pre_q_values is not None:
            pre_action_probs = self.agent.trainer.probabilities_from_q_values(pre_q_values)

        # --- Step environment ---
        t_env_start = time.monotonic()
        next_obs, reward, terminated, truncated, info = self._env.step(action)
        self._metrics["time_env_step"] = (
            (self._metrics.get("time_env_step") or 0.0) + time.monotonic() - t_env_start
        )
        done = terminated or truncated

        # --- Emit FOOD_EATEN event ---
        if reward > 0 and self._mode == ActivityMode.TRAIN:
            if reward == self.env_config.food_reward or reward > self.env_config.food_reward * 0.5:
                self._emit_event(
                    EventType.FOOD_EATEN,
                    f"Food eaten! Score: {info['score']}",
                    score=info["score"],
                )

        # --- Learning (only in TRAIN mode) ---
        replay_loss = None
        if self._mode == ActivityMode.TRAIN and self._agent is not None:
            assert self._observation is not None, "Observation should be set before training"
            self._agent.remember(self._observation, action, reward, next_obs, terminated, truncated)

            self._metrics["training_transitions"] = (
                self._metrics.get("training_transitions") or 0
            ) + 1

            # Step-based epsilon decay
            if self._agent.epsilon_decay_steps > 0:
                self._agent.decay_epsilon_by_steps(
                    self._metrics["training_transitions"],
                    self._agent.epsilon_decay_steps
                )

            # Check replay readiness
            replay_ready = self._agent.memory.is_ready(self._agent.batch_size)
            self._metrics["replay_buffer_size"] = len(self._agent.memory)
            self._metrics["replay_active"] = replay_ready

            if replay_ready and not self._replay_warmed_up:
                self._replay_warmed_up = True
                self._emit_event(EventType.REPLAY_ACTIVE, "Replay buffer ready — replay training active")

            # Capture Bellman details periodically
            is_fast_training = (
                self._mode == ActivityMode.TRAIN
                and hasattr(self, '_execution_policy')
                and self._execution_policy == "FastTraining"
            )
            capture_bellman = (
                not is_fast_training
                and (self._metrics.get("training_transitions", 0) % 10 == 0)
                and replay_ready
            )

            t_opt_start = time.monotonic()
            optimizer_steps_before = self._agent.trainer.training_step
            losses = self._agent.maybe_optimize(capture_bellman=capture_bellman)
            self._metrics["time_optimization"] = (
                (self._metrics.get("time_optimization") or 0.0) + time.monotonic() - t_opt_start
            )

            optimizer_steps_after = self._agent.trainer.training_step
            delta = optimizer_steps_after - optimizer_steps_before
            self._metrics["optimizer_updates"] = (
                (self._metrics.get("optimizer_updates") or 0) + delta
            )
            self._metrics["_prev_optimizer_step"] = optimizer_steps_after

            # Update replay metrics
            self._metrics["replay_buffer_size"] = len(self._agent.memory)
            replay_ready = self._agent.memory.is_ready(self._agent.batch_size)
            self._metrics["replay_active"] = replay_ready

            if replay_ready and not self._replay_warmed_up:
                self._replay_warmed_up = True
                self._emit_event(EventType.REPLAY_ACTIVE, "Replay buffer ready — replay training active")

            if losses:
                replay_loss = losses[-1]

            self._agent.total_steps += 1

        # --- Update state ---
        self._observation = next_obs
        self._game_state = self._env.get_state_for_display()
        self._game_over = done
        self._action = action
        self._step_id += 1
        self._metrics["env_transitions"] = (self._metrics.get("env_transitions") or 0) + 1

        # --- Update Q-values for NEXT state visualization ---
        if self._agent is not None and not skip_diagnostics:
            t_diag_start = time.monotonic()
            self._q_values = self._agent.get_q_values(next_obs)
            self._action_probs = self.agent.trainer.probabilities_from_q_values(self._q_values)
            self._metrics["time_diagnostics"] = (
                (self._metrics.get("time_diagnostics") or 0.0) + time.monotonic() - t_diag_start
            )
        elif skip_diagnostics:
            self._q_values = None
            self._action_probs = None

        # --- Handle episode completion ---
        if done:
            self._handle_episode_end(info["score"], replay_loss, terminated, truncated, info)
            self._metrics["episodes_completed"] = (self._metrics.get("episodes_completed") or 0) + 1

            # Check curriculum advance
            self._check_curriculum_advance()

        # Update timing metrics
        session_start = self._metrics.get("session_start_time")
        if session_start is not None:
            self._metrics["total_session_time"] = time.monotonic() - session_start
            self._metrics["active_execution_time"] = (
                (self._metrics.get("active_execution_time") or 0.0) + time.monotonic() - step_start
            )
            self._metrics["last_transition_time"] = time.monotonic()
        else:
            self._metrics["total_session_time"] = 0.0
            self._metrics["active_execution_time"] = 0.0

        # Update phase
        self._update_phase()

        return self._make_snapshot(
            pre_observation=pre_observation,
            pre_q_values=pre_q_values,
            pre_action_probs=pre_action_probs,
        )

    # -------------------------------------------------------------------------
    # Automatic Execution (for live dashboard)
    # -------------------------------------------------------------------------

    def advance_tick(
        self,
        execution_policy: str = "Watch",
        max_burst_steps: int = 50,
        burst_time_budget: float = 0.05,
    ) -> TrainingSnapshot:
        """
        Execute one automatic training tick according to policy.

        Args:
            execution_policy: "Watch" or "FastTraining"
            max_burst_steps: Max steps per burst (FastTraining only)
            burst_time_budget: Time budget per burst in seconds (FastTraining only)

        Returns:
            Fresh TrainingSnapshot after execution.
        """
        with self._lock:
            return self._advance_tick_impl(execution_policy, max_burst_steps, burst_time_budget)

    def _advance_tick_impl(
        self,
        execution_policy: str = "Watch",
        max_burst_steps: int = 50,
        burst_time_budget: float = 0.05,
    ) -> TrainingSnapshot:
        """Internal advance_tick implementation (must hold self._lock)."""
        if self._faulted:
            raise RuntimeError(
                f"Controller faulted: {self._fault_error}. "
                "Manual recovery required (new_run or load_checkpoint)."
            )

        self._tick_counter += 1
        tick_id = self._tick_counter

        if execution_policy not in ("Watch", "FastTraining"):
            raise ValueError(
                f"Unknown execution policy: {execution_policy!r}. Expected 'Watch' or 'FastTraining'."
            )

        if execution_policy == "FastTraining":
            if max_burst_steps <= 0:
                raise ValueError(f"max_burst_steps must be positive, got {max_burst_steps}")
            if burst_time_budget <= 0:
                raise ValueError(f"burst_time_budget must be positive, got {burst_time_budget}")

        self._execution_policy = execution_policy

        if self._paused or self._mode == ActivityMode.MANUAL:
            return self.get_snapshot()

        if self._game_state is None:
            self.restart_board()

        try:
            if execution_policy == "FastTraining":
                import time as _time

                max_steps = max_burst_steps
                time_budget = burst_time_budget
                burst_start = _time.monotonic()
                steps_executed = 0

                while steps_executed < max_steps and not self._game_over:
                    self._step_impl(skip_diagnostics=True)
                    steps_executed += 1

                    if _time.monotonic() - burst_start >= time_budget:
                        break

                if steps_executed > 0 and not self._game_over and self._agent is not None:
                    t_diag_start = time.monotonic()
                    self._q_values = self._agent.get_q_values(self._observation)
                    self._action_probs = self.agent.trainer.probabilities_from_q_values(self._q_values)
                    self._metrics["time_diagnostics"] = (
                        (self._metrics.get("time_diagnostics") or 0.0) + time.monotonic() - t_diag_start
                    )

                if self._game_over:
                    if self._mode == ActivityMode.TRAIN:
                        self.restart_board()
                    else:
                        self.set_paused(True)
            else:
                if not self._game_over:
                    self._step_impl()
                elif self._mode == ActivityMode.TRAIN:
                    self.restart_board()
                else:
                    self.set_paused(True)
        except Exception as e:
            self._mark_faulted(e, tick_id)
            raise

        return self.get_snapshot()

    def _mark_faulted(self, error: Exception, tick_id: int | None = None) -> None:
        """Mark controller as faulted, preserving error context."""
        self._faulted = True
        self._fault_error = error
        self._fault_tick_id = tick_id
        import traceback

        self._fault_traceback = traceback.format_exc()
        logger.error(
            "Controller faulted: session=%s tick=%s error=%s",
            self._session_id,
            tick_id,
            error,
            exc_info=True,
        )
        self._paused = True

    def _select_action(self) -> tuple[int, str, np.ndarray | None]:
        """Select action based on current mode and agent policy."""
        agent = self.agent
        assert agent is not None
        assert self._observation is not None

        compute_q = not (
            self._mode == ActivityMode.TRAIN
            and hasattr(self, '_execution_policy')
            and self._execution_policy == "FastTraining"
        )

        if self._mode == ActivityMode.TRAIN:
            return agent.get_action_with_q_values(self._observation, training=True, compute_q_values=compute_q)
        elif self._mode == ActivityMode.EVALUATE:
            action, reason, q_values = agent.get_action_with_q_values(
                self._observation, training=False
            )
            return action, reason, q_values
        else:  # MANUAL
            q_values = agent.get_q_values(self._observation)
            return 0, "manual", q_values

    def _handle_episode_end(
        self,
        score: int,
        replay_loss: float | None,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        """Handle episode completion: update stats, decay epsilon (TRAIN only), emit event."""
        termination_reason = info.get("termination_reason", "unknown")

        reason_labels = {
            "wall": "wall collision",
            "self": "self collision",
            "timeout": "timeout",
            "victory": "board completed",
        }
        reason_text = reason_labels.get(termination_reason, "unknown")

        self._emit_event(
            EventType.EPISODE_END,
            f"Episode {self.episode + 1} ended: {reason_text} · Score: {score}",
            score=score,
            termination_reason=termination_reason,
        )

        if self._agent and score > self._agent.high_score:
            self._emit_event(EventType.NEW_RECORD, f"New record! {score}", score=score)

        if self._mode == ActivityMode.TRAIN and self._agent is not None:
            self._agent.update_stats(score, replay_loss, termination_reason)
            if self._agent.epsilon_decay_steps == 0:
                self._agent.decay_epsilon()
            self._episode_id += 1
        else:
            self._episode_id += 1

        self._history_revision += 1

    def _reset_episode(self) -> None:
        """Reset environment for new episode, keep agent state."""
        obs, info = self.env.reset()
        self._observation = obs
        self._game_state = self.env.get_state_for_display()
        self._game_over = False
        self._action = None
        self._selection_reason = None
        self._step_id = 0

        if self._agent is not None:
            self._q_values = self._agent.get_q_values(obs)
            self._action_probs = self._agent.trainer.probabilities_from_q_values(self._q_values)

    # -------------------------------------------------------------------------
    # Snapshot Creation
    # -------------------------------------------------------------------------

    def _make_snapshot(
        self,
        pre_observation: np.ndarray | None = None,
        pre_q_values: np.ndarray | None = None,
        pre_action_probs: np.ndarray | None = None,
    ) -> TrainingSnapshot:
        """Create immutable snapshot of current state for UI."""

        obs = pre_observation if pre_observation is not None else self._observation
        q_vals = pre_q_values if pre_q_values is not None else self._q_values
        probs = pre_action_probs if pre_action_probs is not None else self._action_probs
        action = self._action
        reason = self._selection_reason

        if self._agent:
            recent_scores = self._agent.scores_history[-self._max_history :]
            recent_mean_scores = self._agent.mean_scores_history[-self._max_history :]
            recent_epsilon = self._agent.epsilon_history[-self._max_history :]
            recent_loss = list(self._agent.trainer.loss_history)[-self._loss_max_history :]
        else:
            recent_scores = []
            recent_mean_scores = []
            recent_epsilon = []
            recent_loss = []

        recent_events = self._events[-20:] if self._events else []
        bellman_details = self._agent.get_last_bellman_details() if self._agent else None

        # Build decision record
        last_decision = None
        model_revision = 0
        if self._agent and q_vals is not None and probs is not None and action is not None:
            q_gap = 0.0
            max_q = 0.0
            entropy = 0.0
            if len(q_vals) >= 2:
                sorted_q = np.sort(q_vals)
                q_gap = float(sorted_q[-1] - sorted_q[-2])
                max_q = float(sorted_q[-1])
                probs_nz = probs[probs > 0]
                if len(probs_nz) > 0:
                    entropy = float(-np.sum(probs_nz * np.log(probs_nz)))
            last_decision = DecisionRecord(
                step_id=self._step_id,
                episode_id=self._episode_id,
                model_revision=0,
                observation=obs.copy() if obs is not None else np.zeros(self.agent_config.network.input_size, dtype=np.float32),
                q_values=q_vals.copy(),
                action_probs=probs.copy(),
                action=action,
                selection_reason=reason or "unknown",
                epsilon=self.epsilon,
                timestamp=time.time(),
                q_gap=q_gap,
                max_q=max_q,
                entropy=entropy,
            )
            model_revision = 0

        return TrainingSnapshot(
            observation=obs if obs is not None else np.zeros(self.agent_config.network.input_size, dtype=np.float32),
            game_state=self._game_state or {},
            q_values=q_vals,
            action_probs=probs,
            action=action,
            epsilon=self.epsilon,
            selection_reason=reason,
            game_over=self._game_over,
            episode=self.episode,
            total_env_steps=self.total_env_steps,
            optimizer_steps=self.optimizer_steps,
            score=self.score,
            high_score=self.high_score,
            recent_scores=recent_scores,
            recent_mean_scores=recent_mean_scores,
            recent_epsilon=recent_epsilon,
            recent_loss=recent_loss,
            run_id=self._run_id,
            episode_id=self._episode_id,
            step_id=self._step_id,
            history_revision=self._history_revision,
            metrics=self.get_metrics(),
            recent_events=recent_events,
            bellman_details=bellman_details,
            last_decision=last_decision,
            model_revision=model_revision,
        )

    def get_snapshot(self) -> TrainingSnapshot:
        """Get current state snapshot without stepping."""
        with self._lock:
            return self._make_snapshot()

    # -------------------------------------------------------------------------
    # Checkpoint Operations
    # -------------------------------------------------------------------------

    def save_checkpoint(self, filepath: str) -> None:
        """Save full training checkpoint including environment and controller state."""
        with self._lock:
            if self._agent:
                from .persistence import CheckpointManager
                manager = CheckpointManager(self._agent, self._agent.trainer.device)
                manager.save_full(
                    filepath,
                    env=self._env,
                    run_id=self._run_id,
                    episode_id=self._episode_id,
                    step_id=self._step_id,
                    replay_warmed_up=self._replay_warmed_up,
                )

    def load_checkpoint(self, filepath: str) -> dict[str, Any]:
        """
        Load agent checkpoint transactionally with full state restore.

        Returns:
            Checkpoint metadata dict.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
            ValueError: If architecture/schema mismatch or invalid optimizer state.
            RuntimeError: If controller is faulted.
        """
        with self._lock:
            if self._faulted:
                raise RuntimeError(
                    f"Controller faulted: {self._fault_error}. "
                    "Manual recovery required (new_run or load_checkpoint)."
                )

            from .persistence import CheckpointManager

            # Create candidate agent and env for validation
            candidate_agent = self._create_agent()
            candidate_env = self._create_env()

            candidate_manager = CheckpointManager(candidate_agent, candidate_agent.trainer.device)
            metadata = candidate_manager.load_full(filepath, strict=True, env=candidate_env)

            # Validation passed - atomically replace
            self._agent = candidate_agent
            self._env = candidate_env
            self._run_id = metadata.get("run_id", 0)
            self._episode_id = metadata.get("episode_id", 0)
            self._step_id = metadata.get("step_id", 0)
            self._replay_warmed_up = metadata.get("replay_warmed_up", False)
            self._curriculum_stage_idx = 0  # Reset curriculum on load
            self._phase = "warmup"
            self._reset_metrics()
            self.clear_events()
            self._reset_episode()

            self._emit_event(EventType.CHECKPOINT_LOAD, f"Loaded checkpoint from {filepath}")

            return metadata

    def close(self) -> None:
        """Clean up resources."""
        if self._env:
            self._env.close()
        self._env = None
        self._agent = None


if __name__ == "__main__":
    # Quick test
    from .config import EnvConfig, AgentConfig

    controller = TrainingController(
        env_config=EnvConfig(grid_size=10),
        agent_config=AgentConfig(),
    )
    controller.set_mode(ActivityMode.TRAIN)
    controller.set_paused(False)
    controller.restart_board()

    for _ in range(10):
        snap = controller.advance_tick("FastTraining")
        print(f"Step {snap.step_id}: Score={snap.score}, ε={snap.epsilon:.3f}")

    controller.close()
    print("Controller test passed!")
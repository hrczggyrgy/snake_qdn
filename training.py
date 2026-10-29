"""
TrainingController - Framework-independent training lifecycle management.

This module contains the core training logic shared between headless training
and the Streamlit dashboard. It owns the environment, agent, observation,
mode, and episode lifecycle, with no Streamlit dependencies.
"""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from agent import Agent
from snake_env import SnakeGameAI

# Module logger
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Explicit event types emitted at transition/episode source."""
    FOOD_EATEN = "food_eaten"
    EPISODE_END = "episode_end"
    REPLAY_ACTIVE = "replay_active"
    NEW_RECORD = "new_record"
    MODE_CHANGE = "mode_change"
    CHECKPOINT_LOAD = "checkpoint_load"
    CHECKPOINT_SAVE = "checkpoint_save"
    BOARD_RESTART = "board_restart"
    HISTORY_CLEAR = "history_clear"
    NEW_RUN = "new_run"


@dataclass(frozen=True)
class TrainingEvent:
    """Explicit training event with stable ID and metadata."""
    id: int
    type: EventType
    message: str
    timestamp: float
    # Context data
    episode: int | None = None
    step: int | None = None
    score: int | None = None
    termination_reason: str | None = None  # wall, self, timeout, victory
    run_id: int | None = None
    episode_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "episode": self.episode,
            "step": self.step,
            "score": self.score,
            "termination_reason": self.termination_reason,
            "run_id": self.run_id,
            "episode_id": self.episode_id,
        }


class ActivityMode(Enum):
    """Activity mode determining who acts and whether learning occurs."""

    TRAIN = "train"  # Agent acts with exploration, learns from experience
    EVALUATE = "evaluate"  # Agent acts greedily, no learning
    MANUAL = "manual"  # External action provided, no learning


@dataclass(frozen=True)
class TrainingSnapshot:
    """
    Immutable snapshot of training state for UI consumption.

    All array fields are copied to ensure snapshot isolation.
    """

    # Current step state
    observation: np.ndarray
    game_state: dict[str, Any]
    q_values: np.ndarray | None
    action_probs: np.ndarray | None
    action: int | None
    epsilon: float
    selection_reason: str | None
    game_over: bool

    # Episode/training counters
    episode: int
    total_env_steps: int
    optimizer_steps: int
    score: int
    high_score: int

    # History (copied, bounded for UI)
    recent_scores: list[int]
    recent_mean_scores: list[float]
    recent_epsilon: list[float]
    recent_loss: list[float]

    # Metadata for cache invalidation
    run_id: int
    episode_id: int
    step_id: int

    # Performance metrics (copied for snapshot isolation)
    metrics: dict[str, Any] = field(default_factory=dict)

    # Events (recent, for UI)
    recent_events: list[TrainingEvent] = field(default_factory=list)

    def __post_init__(self):
        # Ensure arrays are copied (frozen dataclass doesn't deep-copy numpy arrays)
        object.__setattr__(
            self, "observation", self.observation.copy() if self.observation is not None else None
        )
        object.__setattr__(
            self, "q_values", self.q_values.copy() if self.q_values is not None else None
        )
        object.__setattr__(
            self,
            "action_probs",
            self.action_probs.copy() if self.action_probs is not None else None,
        )
        object.__setattr__(self, "recent_scores", list(self.recent_scores))
        object.__setattr__(self, "recent_mean_scores", list(self.recent_mean_scores))
        object.__setattr__(self, "recent_epsilon", list(self.recent_epsilon))
        object.__setattr__(self, "recent_loss", list(self.recent_loss))
        object.__setattr__(self, "recent_events", list(self.recent_events))


@dataclass
class EnvConfig:
    """Validated environment configuration."""

    grid_size: int = 20
    step_penalty: float = -0.01
    food_reward: float = 10.0
    death_penalty: float = -10.0
    render_mode: str | None = "rgb_array"
    seed: int | None = None

    def __post_init__(self):
        if self.grid_size < 3:
            raise ValueError(f"grid_size must be >= 3, got {self.grid_size}")
        if not -1.0 <= self.step_penalty <= 0.0:
            raise ValueError(f"step_penalty must be in [-1.0, 0.0], got {self.step_penalty}")
        if self.food_reward < 0.0:
            raise ValueError(f"food_reward must be >= 0, got {self.food_reward}")
        if self.death_penalty > 0.0:
            raise ValueError(f"death_penalty must be <= 0, got {self.death_penalty}")


@dataclass
class AgentConfig:
    """Validated agent configuration."""

    input_size: int = 11
    hidden_size: int = 256
    output_size: int = 3
    lr: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 100_000
    batch_size: int = 1000
    target_update_freq: int = 10
    device: str = "auto"
    dropout: float = 0.1

    def __post_init__(self):
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {self.hidden_size}")
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
            raise ValueError(
                f"batch_size ({self.batch_size}) must be <= memory_size ({self.memory_size})"
            )


class TrainingController:
    """
    Framework-independent training controller.

    Owns the complete training lifecycle: environment, agent, observations,
    mode, episode management, and metrics. Provides explicit operations for
    stepping, restarting, clearing history, and creating new runs.

    Contains NO Streamlit imports or rendering logic.
    """

    def __init__(
        self,
        env_config: EnvConfig | None = None,
        agent_config: AgentConfig | None = None,
    ):
        self.env_config = env_config or EnvConfig()
        self.agent_config = agent_config or AgentConfig()

        # Core components (lazy initialization)
        self._env: SnakeGameAI | None = None
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

        # History windows for UI (bounded)
        self._max_history = 500
        self._loss_max_history = 1000

        # Performance metrics (Phase 1)
        self._metrics = {
            "transitions": 0,
            "optimizer_updates": 0,
            "episodes_completed": 0,
            "total_training_time": 0.0,
            "replay_buffer_size": 0,
            "replay_active": False,
            "time_action_selection": 0.0,
            "time_env_step": 0.0,
            "time_optimization": 0.0,
            "time_diagnostics": 0.0,
            "time_display_prep": 0.0,
            "start_time": None,
            "last_transition_time": None,
        }

        # Event system
        self._event_counter = 0
        self._events: list[TrainingEvent] = []
        self._max_events = 200
        self._replay_warmed_up = False

        # Execution lock: serializes ALL model/optimizer mutations
        # Must be held for: step(), advance_tick(), load_checkpoint(),
        # restart_board(), new_run(), set_mode(), update_agent_config(), update_env_config()
        self._lock = threading.RLock()

        # Fault state: set on execution failure, prevents automatic retries
        self._faulted: bool = False
        self._fault_error: Exception | None = None
        self._fault_traceback: str | None = None
        self._fault_tick_id: int | None = None

        # Execution tracing
        self._tick_counter: int = 0
        self._session_id: str = uuid.uuid4().hex[:8]

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
    def env(self) -> SnakeGameAI:
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
        return self._episode_id - (self.agent.n_games if self._agent else 0)

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

    def get_metrics(self) -> dict[str, Any]:
        """Get current performance metrics (copy for isolation)."""
        metrics = self._metrics.copy()
        # Add derived rates
        elapsed = metrics.get("total_training_time", 0.0)
        if elapsed > 0:
            metrics["transitions_per_sec"] = metrics["transitions"] / elapsed
            metrics["optimizer_updates_per_sec"] = metrics["optimizer_updates"] / elapsed
        else:
            metrics["transitions_per_sec"] = 0.0
            metrics["optimizer_updates_per_sec"] = 0.0
        return metrics

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
            self._events = self._events[-self._max_events:]
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

    def _create_env(self) -> SnakeGameAI:
        return SnakeGameAI(
            grid_size=self.env_config.grid_size,
            step_penalty=self.env_config.step_penalty,
            food_reward=self.env_config.food_reward,
            death_penalty=self.env_config.death_penalty,
            render_mode=self.env_config.render_mode,
            seed=self.env_config.seed,
        )

    def _create_agent(self) -> Agent:
        return Agent(
            input_size=self.agent_config.input_size,
            hidden_size=self.agent_config.hidden_size,
            output_size=self.agent_config.output_size,
            lr=self.agent_config.lr,
            gamma=self.agent_config.gamma,
            epsilon_start=self.agent_config.epsilon_start,
            epsilon_end=self.agent_config.epsilon_end,
            epsilon_decay=self.agent_config.epsilon_decay,
            memory_size=self.agent_config.memory_size,
            batch_size=self.agent_config.batch_size,
            target_update_freq=self.agent_config.target_update_freq,
            device=self.agent_config.device,
        )

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
            # Ensure agent exists so _reset_episode can compute initial Q-values
            _ = self.agent
            self._reset_episode()
            self._emit_event(
                EventType.BOARD_RESTART,
                "Board restarted",
            )

    def clear_history(self) -> None:
        """Clear displayed charts and statistics, keep model weights and replay buffer."""
        with self._lock:
            if self._agent:
                self._agent.reset_stats()
            self._episode_id = 0
            self._step_id = 0
            self.clear_events()
            self._emit_event(
                EventType.HISTORY_CLEAR,
                "Training history cleared",
            )

    def new_run(self) -> None:
        """Create fresh agent and replay buffer, discard current model."""
        with self._lock:
            self._agent = self._create_agent()
            self._env = self._create_env()
            self._run_id += 1
            self._episode_id = 0
            self._step_id = 0
            self._replay_warmed_up = False
            self.clear_events()
            self._reset_episode()
            self._emit_event(
                EventType.NEW_RUN,
                "New training run started",
            )

    def update_env_config(self, config: EnvConfig) -> None:
        """Update environment configuration and recreate env/agent."""
        with self._lock:
            self.env_config = config
            self._env = self._create_env()
            self._agent = self._create_agent()
            self._run_id += 1
            self._episode_id = 0
            self._step_id = 0
            self._replay_warmed_up = False
            self.clear_events()
            self._reset_episode()
            self._emit_event(
                EventType.MODE_CHANGE,
                "Environment config updated — new run",
            )

    def update_agent_config(self, config: AgentConfig) -> None:
        """Update agent configuration and recreate agent."""
        with self._lock:
            self.agent_config = config
            self._agent = self._create_agent()
            self._reset_episode()
            self.clear_events()
            self._emit_event(
                EventType.MODE_CHANGE,
                "Agent config updated — new agent created",
            )

    # -------------------------------------------------------------------------
    # Public API - Step Operations
    # -------------------------------------------------------------------------

    def step(self, action: int | None = None) -> TrainingSnapshot:
        """
        Execute one environment step.

        Args:
            action: Optional action for MANUAL mode. Ignored in TRAIN/EVALUATE.

        Returns:
            TrainingSnapshot with pre-action decision data and post-step state.
        """
        with self._lock:
            return self._step_impl(action)

    def _step_impl(self, action: int | None = None) -> TrainingSnapshot:
        """Internal step implementation (must hold self._lock)."""
        if self._faulted:
            raise RuntimeError(
                f"Controller faulted: {self._fault_error}. "
                "Manual recovery required (new_run or load_checkpoint)."
            )
        if self._game_over or self._env is None:
            return self._make_snapshot()

        # Initialize metrics start time if not set
        if self._metrics["start_time"] is None:
            self._metrics["start_time"] = time.monotonic()

        # --- Get action (pre-step) ---
        t_action_start = time.monotonic()
        # Ensure agent exists for automatic action selection
        if action is None:
            # This will create agent if needed
            _ = self.agent
            action, self._selection_reason, pre_q_values = self._select_action()
        elif action is not None:
            # Manual action provided
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
        self._metrics["time_action_selection"] += time.monotonic() - t_action_start

        # --- Capture PRE-ACTION decision snapshot ---
        # This is what the agent saw and decided BEFORE stepping
        pre_observation = self._observation.copy() if self._observation is not None else None
        pre_action_probs = None

        if pre_q_values is not None:
            pre_action_probs = self.agent.trainer.probabilities_from_q_values(pre_q_values)

        # --- Step environment ---
        t_env_start = time.monotonic()
        next_obs, reward, terminated, truncated, info = self._env.step(action)
        self._metrics["time_env_step"] += time.monotonic() - t_env_start
        done = terminated or truncated

        # --- Emit FOOD_EATEN event ---
        # Food reward is positive and distinct from step penalty
        if reward > 0 and self._mode == ActivityMode.TRAIN:
            # Check if this is actually food (not death penalty which is negative)
            if reward == self.env_config.food_reward or reward > self.env_config.food_reward * 0.5:
                self._emit_event(
                    EventType.FOOD_EATEN,
                    f"Food eaten! Score: {info['score']}",
                    score=info["score"],
                )

        # --- Learning (only in TRAIN mode) ---
        replay_loss = None
        if self._mode == ActivityMode.TRAIN and self._agent is not None:
            t_opt_start = time.monotonic()
            # Store transition (terminated controls bootstrap, truncated stored separately)
            self._agent.remember(self._observation, action, reward, next_obs, terminated, truncated)
            # Train on short memory (immediate)
            self._agent.train_short_memory(self._observation, action, reward, next_obs, terminated)
            # Train on replay memory
            replay_loss = self._agent.train_long_memory()
            self._metrics["time_optimization"] += time.monotonic() - t_opt_start

            # Update replay metrics
            self._metrics["replay_buffer_size"] = len(self._agent.memory)
            replay_ready = self._agent.memory.is_ready(self._agent.batch_size)
            self._metrics["replay_active"] = replay_ready

            # Emit REPLAY_ACTIVE event once when replay becomes ready
            if replay_ready and not self._replay_warmed_up:
                self._replay_warmed_up = True
                self._emit_event(
                    EventType.REPLAY_ACTIVE,
                    "Replay buffer ready — replay training active",
                )

            if replay_loss is not None:
                self._metrics["optimizer_updates"] += 1

            # Sync agent's step counter
            self._agent.total_steps += 1

        # --- Update state ---
        self._observation = next_obs
        self._game_state = self._env.get_state_for_display()
        self._game_over = done
        self._action = action
        self._step_id += 1
        self._metrics["transitions"] += 1

        # --- Update Q-values for NEXT state visualization ---
        if self._agent is not None:
            t_diag_start = time.monotonic()
            self._q_values = self._agent.get_q_values(next_obs)
            self._action_probs = self.agent.trainer.probabilities_from_q_values(self._q_values)
            self._metrics["time_diagnostics"] += time.monotonic() - t_diag_start

        # --- Handle episode completion ---
        if done:
            self._handle_episode_end(info["score"], replay_loss, terminated, truncated, info)
            self._metrics["episodes_completed"] += 1

        # Update total training time
        self._metrics["total_training_time"] = time.monotonic() - self._metrics["start_time"]

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

        This is the single entry point for automatic execution (used by live dashboard).
        It handles:
        - Watch mode: one step, with auto-restart on terminal board in TRAIN
        - FastTraining: bounded burst of steps, with auto-restart on terminal board in TRAIN
        - Evaluation/Manual: pauses on terminal board

        Args:
            execution_policy: "Watch" or "FastTraining"
            max_burst_steps: Max steps per burst (FastTraining only)
            burst_time_budget: Time budget per burst in seconds (FastTraining only)

        Returns:
            Fresh TrainingSnapshot after execution.

        Raises:
            ValueError: If execution_policy is invalid or burst settings are invalid.
            RuntimeError: If controller is faulted.
            Any exception from step()/restart_board() - caller should handle.
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

        # Validate execution policy
        if execution_policy not in ("Watch", "FastTraining"):
            raise ValueError(f"Unknown execution policy: {execution_policy!r}. Expected 'Watch' or 'FastTraining'.")

        # Validate FastTraining parameters
        if execution_policy == "FastTraining":
            if max_burst_steps <= 0:
                raise ValueError(f"max_burst_steps must be positive, got {max_burst_steps}")
            if burst_time_budget <= 0:
                raise ValueError(f"burst_time_budget must be positive, got {burst_time_budget}")

        # Don't advance if paused or in manual mode
        if self._paused or self._mode == ActivityMode.MANUAL:
            return self.get_snapshot()

        # Ensure board is initialized (only if never started)
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
                    self._step_impl()
                    steps_executed += 1

                    if _time.monotonic() - burst_start >= time_budget:
                        break

                # After burst, handle terminal board based on mode
                if self._game_over:
                    if self._mode == ActivityMode.TRAIN:
                        # Training: auto-restart on terminal board
                        self.restart_board()
                    else:
                        # Evaluation/Manual: pause on terminal board
                        self.set_paused(True)
            else:
                # Watch mode
                if not self._game_over:
                    self._step_impl()
                elif self._mode == ActivityMode.TRAIN:
                    # Training: auto-restart on terminal board
                    self.restart_board()
                else:
                    # Evaluation: pause on terminal board
                    self.set_paused(True)
        except Exception as e:
            # Mark controller faulted on ANY exception during execution
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
        # Pause to prevent automatic retries
        self._paused = True

    def _select_action(self) -> tuple[int, str, np.ndarray | None]:
        """Select action based on current mode and agent policy."""
        agent = self.agent  # Ensure agent is created
        assert agent is not None
        assert self._observation is not None

        if self._mode == ActivityMode.TRAIN:
            return agent.get_action_with_q_values(self._observation, training=True)
        elif self._mode == ActivityMode.EVALUATE:
            action, reason, q_values = agent.get_action_with_q_values(
                self._observation, training=False
            )
            return action, reason, q_values
        else:  # MANUAL
            # Should not reach here - manual actions passed explicitly
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
        # Classify termination reason
        termination_reason = self._classify_termination(terminated, truncated, info)

        reason_labels = {
            "wall": "wall collision",
            "self": "self collision",
            "timeout": "timeout",
            "victory": "board completed",
        }
        reason_text = reason_labels.get(termination_reason, "unknown")

        # Emit EPISODE_END event
        self._emit_event(
            EventType.EPISODE_END,
            f"Episode {self.episode + 1} ended: {reason_text} · Score: {score}",
            score=score,
            termination_reason=termination_reason,
        )

        # Check for new high score
        if self._agent and score > self._agent.high_score:
            self._emit_event(
                EventType.NEW_RECORD,
                f"New record! {score}",
                score=score,
            )

        if self._mode == ActivityMode.TRAIN and self._agent is not None:
            # Update agent's authoritative stats
            self._agent.update_stats(score, replay_loss, termination_reason)
            # Decay epsilon ONCE per completed training episode
            self._agent.decay_epsilon()

            self._episode_id += 1
        else:
            # For EVALUATE/MANUAL, still track episode_id for UI
            self._episode_id += 1

    def _classify_termination(
        self, terminated: bool, truncated: bool, info: dict[str, Any]
    ) -> str:
        """Classify episode termination reason from environment info."""
        if truncated:
            return "timeout"

        if not terminated:
            return "unknown"

        # terminated=True - check collision type using game state
        game_state = self._game_state or {}
        # Food is None means board full (victory)
        if game_state.get("food") is None:
            return "victory"

        # Check if head collided with wall or self
        snake = game_state.get("snake", [])
        if not snake:
            return "wall"  # fallback

        head = snake[0]
        grid_size = game_state.get("grid_size", 20)

        # Wall collision
        if head[0] < 0 or head[0] >= grid_size or head[1] < 0 or head[1] >= grid_size:
            return "wall"

        # Self collision (head in body)
        if head in snake[1:]:
            return "self"

        # Default to wall if we can't determine
        return "wall"

    def _reset_episode(self) -> None:
        """Reset environment for new episode, keep agent state."""
        obs, info = self.env.reset()
        self._observation = obs
        self._game_state = self.env.get_state_for_display()
        self._game_over = False
        self._action = None
        self._selection_reason = None
        self._step_id = 0

        # Get initial Q-values if agent exists
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

        # Use pre-action data if provided (for decision visualization)
        obs = pre_observation if pre_observation is not None else self._observation
        q_vals = pre_q_values if pre_q_values is not None else self._q_values
        probs = pre_action_probs if pre_action_probs is not None else self._action_probs
        action = self._action
        reason = self._selection_reason

        # Get history (bounded copies)
        if self._agent:
            recent_scores = self._agent.scores_history[-self._max_history :]
            recent_mean_scores = self._agent.mean_scores_history[-self._max_history :]
            recent_epsilon = self._agent.epsilon_history[-self._max_history :]
            recent_loss = self._agent.trainer.loss_history[-self._loss_max_history :]
        else:
            recent_scores = []
            recent_mean_scores = []
            recent_epsilon = []
            recent_loss = []

        # Get recent events (last 20)
        recent_events = self._events[-20:] if self._events else []

        return TrainingSnapshot(
            observation=obs if obs is not None else np.zeros(11, dtype=np.float32),
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
            metrics=self.get_metrics(),
            recent_events=recent_events,
        )

    def get_snapshot(self) -> TrainingSnapshot:
        """Get current state snapshot without stepping."""
        with self._lock:
            return self._make_snapshot()

    # -------------------------------------------------------------------------
    # Checkpoint Operations
    # -------------------------------------------------------------------------

    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint."""
        with self._lock:
            if self._agent:
                self._agent.save_checkpoint(filepath)

    def load_checkpoint(self, filepath: str) -> dict[str, Any]:
        """
        Load agent checkpoint transactionally.

        Loads into a candidate agent first, validates architecture and optimizer state,
        then atomically replaces the live agent. If validation fails, the live agent
        is unchanged.

        Returns:
            Checkpoint metadata dict.

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
            ValueError: If architecture mismatch or invalid optimizer state.
            RuntimeError: If controller is faulted.
        """
        with self._lock:
            if self._faulted:
                raise RuntimeError(
                    f"Controller faulted: {self._fault_error}. "
                    "Manual recovery required (new_run or load_checkpoint)."
                )

            # Create candidate agent
            candidate_agent = self._create_agent()

            # Load checkpoint into candidate
            try:
                info = candidate_agent.load_checkpoint(filepath)
            except Exception as e:
                raise ValueError(f"Failed to load checkpoint: {e}") from e

            # Validate architecture matches current config
            if not self._validate_agent_architecture(candidate_agent):
                raise ValueError(
                    f"Checkpoint architecture mismatch: "
                    f"expected {self.agent_config}, "
                    f"got hidden={candidate_agent.trainer.model.hidden_size}"
                )

            # Validate optimizer state (Adam must have exp_avg and exp_avg_sq for all params)
            self._validate_optimizer_state(candidate_agent)

            # All validations passed - atomically swap
            self._agent = candidate_agent
            # Sync internal counters
            self._episode_id = self._agent.n_games
            self.clear_events()
            self._replay_warmed_up = self._agent.memory.is_ready(self._agent.batch_size)
            self._emit_event(
                EventType.CHECKPOINT_LOAD,
                f"Loaded checkpoint: ep {info.get('episode', 0)}, ε={info.get('epsilon', 1.0):.3f}",
            )
            return info

    def _validate_agent_architecture(self, agent: Agent) -> bool:
        """Validate that agent architecture matches current config."""
        model = agent.trainer.model
        return (
            model.input_size == self.agent_config.input_size
            and model.hidden_size == self.agent_config.hidden_size
            and model.output_size == self.agent_config.output_size
        )

    def _validate_optimizer_state(self, agent: Agent) -> None:
        """
        Validate that Adam optimizer state is complete and consistent.

        Adam requires 'exp_avg' and 'exp_avg_sq' for each parameter.
        This catches incomplete/mismatched optimizer state from checkpoints.
        """
        optimizer = agent.trainer.optimizer
        state = optimizer.state_dict()["state"]

        # If optimizer has never been stepped, state may be empty (legitimate)
        if not state:
            return

        param_groups = optimizer.state_dict()["param_groups"]
        expected_params = sum(len(pg["params"]) for pg in param_groups)

        for param_id, param_state in state.items():
            # Check required Adam fields
            if "exp_avg" not in param_state:
                raise ValueError(
                    f"Optimizer state missing 'exp_avg' for parameter {param_id}. "
                    "Checkpoint may be corrupted or from incompatible optimizer."
                )
            if "exp_avg_sq" not in param_state:
                raise ValueError(
                    f"Optimizer state missing 'exp_avg_sq' for parameter {param_id}. "
                    "Checkpoint may be corrupted or from incompatible optimizer."
                )

            # Validate shapes match parameter shapes
            # Note: exact shape validation requires mapping param_id to actual param
            # This is a basic sanity check

        if len(state) != expected_params:
            raise ValueError(
                f"Optimizer state has {len(state)} parameters, expected {expected_params}. "
                "Checkpoint may be from different model architecture."
            )

    def recover_from_fault(self, strategy: str = "new_run") -> None:
        """
        Recover from a faulted state.

        Args:
            strategy: "new_run" to create fresh agent, "load_checkpoint" to load from file
                     (requires filepath to be set via config or default)
        """
        with self._lock:
            if not self._faulted:
                return  # Not faulted, nothing to do

            if strategy == "new_run":
                self._agent = self._create_agent()
                self._env = self._create_env()
                self._run_id += 1
                self._episode_id = 0
                self._step_id = 0
                self._replay_warmed_up = False
                self.clear_events()
                self._reset_episode()
                self._faulted = False
                self._fault_error = None
                self._fault_traceback = None
                self._fault_tick_id = None
                self._emit_event(
                    EventType.NEW_RUN,
                    "Recovered from fault — new training run started",
                )
            elif strategy == "load_checkpoint":
                # Would need a filepath; for now just new_run
                self._agent = self._create_agent()
                self._env = self._create_env()
                self._run_id += 1
                self._episode_id = 0
                self._step_id = 0
                self._replay_warmed_up = False
                self.clear_events()
                self._reset_episode()
                self._faulted = False
                self._fault_error = None
                self._fault_traceback = None
                self._fault_tick_id = None
                self._emit_event(
                    EventType.NEW_RUN,
                    "Recovered from fault — new training run started",
                )
            else:
                raise ValueError(f"Unknown recovery strategy: {strategy}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Clean up resources."""
        if self._env:
            self._env.close()
            self._env = None


# -------------------------------------------------------------------------
# Headless Training Function (uses TrainingController)
# -------------------------------------------------------------------------


def train_headless(
    controller: TrainingController,
    n_episodes: int = 1000,
    max_steps_per_episode: int = 2000,
    save_every: int = 100,
    save_path: str = "checkpoints/snake_dqn.pth",
    callback=None,
) -> dict[str, list]:
    """
    Headless training loop using TrainingController.

    This replaces the old train_agent() function with shared logic.
    """
    controller.set_mode(ActivityMode.TRAIN)
    controller.set_paused(False)

    for episode in range(n_episodes):
        # Check for faulted state
        if controller.faulted:
            raise RuntimeError(
                f"Controller faulted at tick {controller._fault_tick_id}: {controller.fault_error}. "
                "Call controller.recover_from_fault() before continuing."
            )

        # Start new episode
        controller.restart_board()

        episode_done = False
        for _step in range(max_steps_per_episode):
            snapshot = controller.step()

            if snapshot.game_over:
                episode_done = True
                break

        # If episode didn't end naturally (truncated by max_steps), force episode end
        if not episode_done:
            # Manually update stats for truncated episode
            if controller.agent is not None:
                controller.agent.update_stats(controller.score, None, "timeout")
                controller.agent.decay_epsilon()
                controller._episode_id += 1

        # Callback for progress
        if callback:
            callback(
                episode,
                {
                    "score": controller.score,
                    "avg_score_50": controller.agent.mean_scores_history[-1]
                    if controller.agent.mean_scores_history
                    else 0,
                    "epsilon": controller.epsilon,
                    "loss": controller.agent.trainer.loss_history[-1]
                    if controller.agent.trainer.loss_history
                    else None,
                },
            )

        # Save checkpoint
        if save_every > 0 and (episode + 1) % save_every == 0:
            controller.save_checkpoint(save_path)

    # Final save
    controller.save_checkpoint(save_path)

    return {
        "scores": controller.agent.scores_history,
        "mean_scores": controller.agent.mean_scores_history,
        "epsilon": controller.agent.epsilon_history,
        "loss": controller.agent.loss_history,
    }


def evaluate_headless(
    controller: TrainingController,
    n_episodes: int = 10,
    max_steps_per_episode: int = 2000,
) -> dict[str, Any]:
    """Evaluate agent performance using TrainingController."""
    if controller.faulted:
        raise RuntimeError(
            f"Controller faulted at tick {controller._fault_tick_id}: {controller.fault_error}. "
            "Call controller.recover_from_fault() before continuing."
        )

    controller.set_mode(ActivityMode.EVALUATE)
    controller.set_paused(False)

    scores = []
    steps_list = []

    for _ in range(n_episodes):
        controller.restart_board()
        steps = 0

        while True:
            snapshot = controller.step()
            steps += 1

            if snapshot.game_over or steps >= max_steps_per_episode:
                scores.append(snapshot.score)
                steps_list.append(steps)
                break

    return {
        "mean_score": np.mean(scores) if scores else 0.0,
        "std_score": np.std(scores) if scores else 0.0,
        "max_score": np.max(scores) if scores else 0,
        "min_score": np.min(scores) if scores else 0,
        "mean_steps": np.mean(steps_list) if steps_list else 0.0,
        "scores": scores,
    }


if __name__ == "__main__":
    # Quick test
    print("Testing TrainingController...")

    controller = TrainingController()

    # Test TRAIN mode
    controller.set_mode(ActivityMode.TRAIN)
    controller.set_paused(False)

    for _ in range(3):
        controller.restart_board()
        done = False
        while not done:
            snap = controller.step()
            done = snap.game_over
        print(
            f"  Episode {controller.episode}: score={controller.score}, epsilon={controller.epsilon:.3f}"
        )

    # Test EVALUATE mode
    controller.set_mode(ActivityMode.EVALUATE)
    controller.restart_board()
    done = False
    while not done:
        snap = controller.step()
        done = snap.game_over
    print(f"  Evaluation: score={controller.score}, epsilon unchanged={controller.epsilon:.3f}")

    # Test MANUAL mode
    controller.set_mode(ActivityMode.MANUAL)
    controller.restart_board()
    snap = controller.step(action=0)  # Straight
    print(f"  Manual step: action={snap.action}, score={snap.score}")

    # Test snapshot isolation
    snap1 = controller.get_snapshot()
    snap2 = controller.get_snapshot()
    assert snap1.run_id == snap2.run_id
    assert snap1.observation is not snap2.observation  # Copied
    print("  Snapshot isolation verified")

    controller.close()
    print("TrainingController test passed!")

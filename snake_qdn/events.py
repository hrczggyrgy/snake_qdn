"""
Event types and records for training lifecycle.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
    CURRICULUM_ADVANCE = "curriculum_advance"
    PHASE_CHANGE = "phase_change"  # warmup, exploring, exploiting, level-up


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
class DecisionRecord:
    """Record of a single decision for educational/debugging purposes."""

    step_id: int
    episode_id: int
    model_revision: int
    observation: Any  # np.ndarray
    q_values: Any  # np.ndarray
    action_probs: Any  # np.ndarray
    action: int
    selection_reason: str
    epsilon: float
    timestamp: float
    # Computed fields
    q_gap: float = 0.0
    max_q: float = 0.0
    entropy: float = 0.0


@dataclass(frozen=True)
class BellmanDetails:
    """Bellman update explanation for a sampled replay transition."""

    state: Any  # np.ndarray
    action: int
    reward: float
    next_state: Any  # np.ndarray
    terminated: bool
    current_q: Any  # np.ndarray
    current_q_taken: float
    target_q: float
    td_error: float
    target_q_max: float
    greedy_action: int
    gamma: float
    n_step: int = 1

    def to_story(self) -> str:
        """Generate human-readable Bellman explanation."""
        action_names = ["Straight", "Right", "Left"]
        action_name = action_names[self.action] if self.action < 3 else str(self.action)
        return (
            f"I chose {action_name} (Q={self.current_q_taken:.3f}). "
            f"Reward was {self.reward:.3f}. "
            f"Target says {self.target_q:.3f} "
            f"(γ={self.gamma:.2f} × max Q'={self.target_q_max:.3f}) "
            f"→ TD error {self.td_error:+.3f}"
        )


@dataclass
class PhaseInfo:
    """Current training phase for UI banner."""

    name: str  # "warmup", "exploring", "exploiting", "level-up"
    progress: float  # 0.0 to 1.0 within phase
    description: str

    @staticmethod
    def from_controller(controller) -> "PhaseInfo":
        """Determine phase from controller state."""
        metrics = controller.get_metrics()
        training_transitions = metrics.get("training_transitions", 0)
        agent = controller.agent
        learning_starts = agent.learning_starts if agent else 1000
        epsilon_decay_steps = agent.epsilon_decay_steps if agent else 50_000

        if training_transitions < learning_starts:
            progress = training_transitions / max(1, learning_starts)
            return PhaseInfo(
                name="warmup",
                progress=progress,
                description=f"Filling replay buffer ({training_transitions}/{learning_starts})",
            )
        elif training_transitions < learning_starts + epsilon_decay_steps:
            progress = (training_transitions - learning_starts) / max(1, epsilon_decay_steps)
            eps = agent.epsilon if agent else 1.0
            return PhaseInfo(
                name="exploring",
                progress=progress,
                description=f"Exploring (ε={eps:.3f})",
            )
        else:
            eps = agent.epsilon if agent else 0.01
            return PhaseInfo(
                name="exploiting",
                progress=1.0,
                description=f"Exploiting (ε={eps:.3f})",
            )


@dataclass(frozen=True)
class TrainingSnapshot:
    """
    Immutable snapshot of training state for UI consumption.

    All array fields are copied to ensure snapshot isolation.
    """

    # Current step state
    observation: Any
    game_state: dict[str, Any]
    q_values: Any | None
    action_probs: Any | None
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
    history_revision: int = 0

    # Performance metrics (copied for snapshot isolation)
    metrics: dict[str, Any] = field(default_factory=dict)

    # Decision context (for Inspect Learning)
    last_decision: DecisionRecord | None = None
    model_revision: int = 0  # Increments when model weights change significantly

    # Events (recent, for UI)
    recent_events: list[TrainingEvent] = field(default_factory=list)

    # Bellman update explanation (sampled replay transition)
    bellman_details: dict | None = None

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
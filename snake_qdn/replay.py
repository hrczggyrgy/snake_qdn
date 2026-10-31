"""
Experience Replay Buffer with n-step returns.

Uses preallocated arrays for efficient storage and sampling.
Supports n-step returns for faster credit assignment.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class NStepTransition:
    """Aggregated n-step transition for replay buffer."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool


class ReplayBuffer:
    """
    Compact Experience Replay Buffer using preallocated arrays.

    Stores terminated flag separately from truncated for correct
    bootstrapping in Q-learning (terminated suppresses bootstrap).

    Supports n-step returns via an internal n-step buffer.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        state_dim: int = 14,
        n_step: int = 3,
        gamma: float = 0.99,
    ):
        self.capacity = capacity
        self.state_dim = state_dim
        self.n_step = n_step
        self.gamma = gamma
        self.size = 0
        self.position = 0

        # Preallocate arrays
        self.states: np.ndarray = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self.rewards: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self.next_states: np.ndarray = np.zeros((capacity, state_dim), dtype=np.float32)
        self.terminated: np.ndarray = np.zeros(capacity, dtype=bool)
        self.truncated: np.ndarray = np.zeros(capacity, dtype=bool)

        # n-step buffer
        self._n_step_buffer: deque[tuple] = deque(maxlen=n_step)

    def _compute_n_step_return(self) -> NStepTransition | None:
        """Compute n-step return from buffer."""
        if len(self._n_step_buffer) < self.n_step:
            return None

        # Get the first transition (oldest)
        state, action, _, _, _, _ = self._n_step_buffer[0]

        # Accumulate rewards with discount
        reward = 0.0
        for i, (_, _, r, _, term, trunc) in enumerate(self._n_step_buffer):
            reward += (self.gamma**i) * r
            if term or trunc:
                # Episode ended - bootstrap only from this point
                next_state, _, _, _, term, trunc = self._n_step_buffer[i]
                return NStepTransition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminated=term,
                    truncated=trunc,
                )

        # Full n-step without termination
        _, _, _, next_state, term, trunc = self._n_step_buffer[-1]
        return NStepTransition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=term,
            truncated=trunc,
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        Add experience to n-step buffer, possibly pushing aggregated transition to main buffer.

        Args:
            terminated: True if episode ended due to game-over (wall/self collision)
            truncated: True if episode ended due to external limit (timeout)
        """
        # Add to n-step buffer
        self._n_step_buffer.append(
            (state, action, reward, next_state, terminated, truncated)
        )

        # Try to compute n-step return
        n_step_transition = self._compute_n_step_return()
        if n_step_transition is not None:
            self._push_transition(n_step_transition)
            # Remove the oldest transition since we've used it for n-step return
            if len(self._n_step_buffer) >= self.n_step:
                self._n_step_buffer.popleft()

        # If episode ended, flush remaining n-step buffer
        if terminated or truncated:
            while self._n_step_buffer:
                n_step_transition = self._compute_n_step_return()
                if n_step_transition is not None:
                    self._push_transition(n_step_transition)
                # Remove the oldest transition
                if self._n_step_buffer:
                    self._n_step_buffer.popleft()
            self._n_step_buffer.clear()

    def _push_transition(self, transition: NStepTransition) -> None:
        """Push aggregated transition to main buffer."""
        self.states[self.position] = transition.state
        self.actions[self.position] = transition.action
        self.rewards[self.position] = transition.reward
        self.next_states[self.position] = transition.next_state
        self.terminated[self.position] = transition.terminated
        self.truncated[self.position] = transition.truncated

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

    def clear(self) -> None:
        """Clear the buffer."""
        self.size = 0
        self.position = 0
        self._n_step_buffer.clear()

    def get_state_dict(self) -> dict[str, Any]:
        """Get buffer state for checkpointing."""
        return {
            "states": self.states[:self.size].copy(),
            "actions": self.actions[:self.size].copy(),
            "rewards": self.rewards[:self.size].copy(),
            "next_states": self.next_states[:self.size].copy(),
            "terminated": self.terminated[:self.size].copy(),
            "truncated": self.truncated[:self.size].copy(),
            "size": self.size,
            "position": self.position,
            "capacity": self.capacity,
            "n_step": self.n_step,
            "gamma": self.gamma,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load buffer state from checkpoint."""
        size = min(state_dict["size"], self.capacity)
        self.size = size
        self.position = state_dict["position"] % self.capacity

        # Validate capacity matches
        if state_dict["capacity"] != self.capacity:
            raise ValueError(
                f"Replay buffer capacity mismatch: checkpoint has {state_dict['capacity']}, "
                f"current agent has {self.capacity}. Cannot resume with different capacity."
            )

        # Validate n_step and gamma match
        if state_dict.get("n_step", self.n_step) != self.n_step:
            raise ValueError(f"n_step mismatch: checkpoint={state_dict.get('n_step')}, current={self.n_step}")
        if abs(state_dict.get("gamma", self.gamma) - self.gamma) > 1e-6:
            raise ValueError(f"gamma mismatch: checkpoint={state_dict.get('gamma')}, current={self.gamma}")

        # Restore arrays
        for field in ["states", "actions", "rewards", "next_states", "terminated", "truncated"]:
            if field in state_dict:
                arr = state_dict[field]
                self.__dict__[field][:self.size] = arr[:self.size]


if __name__ == "__main__":
    # Quick test
    buf = ReplayBuffer(capacity=1000, state_dim=14, n_step=3, gamma=0.99)

    # Add some transitions
    for i in range(10):
        state = np.random.randn(14).astype(np.float32)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(14).astype(np.float32)
        terminated = i == 9
        truncated = False
        buf.push(state, action, reward, next_state, terminated, truncated)

    print(f"Buffer size: {len(buf)}")
    print(f"Buffer ready for batch 32: {buf.is_ready(32)}")

    # Sample
    batch = buf.sample(5)
    print(f"Sampled states shape: {batch[0].shape}")
    print(f"Sampled actions shape: {batch[1].shape}")
    print(f"Sampled rewards shape: {batch[2].shape}")

    # Test checkpoint
    state_dict = buf.get_state_dict()
    print(f"State dict keys: {state_dict.keys()}")
    print(f"Saved size: {state_dict['size']}")

    print("Replay buffer test passed!")
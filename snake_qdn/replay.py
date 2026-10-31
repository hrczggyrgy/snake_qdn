"""
Experience Replay Buffer with n-step returns.

Uses preallocated arrays for efficient storage and sampling.
Supports n-step returns for faster credit assignment.

Fixes:
- C2: Proper flush of remaining transitions at episode end
- Stores per-sample discount (gamma^k) for correct bootstrapping
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
    discount: float  # gamma^k for this transition


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
        state_dim: int = 17,  # Updated to 17 for new observation
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
        self.discounts: np.ndarray = np.zeros(capacity, dtype=np.float32)  # gamma^k per sample

        # n-step buffer
        self._n_step_buffer: deque[tuple] = deque(maxlen=n_step)

    def _emit(self, k: int) -> None:
        """
        Emit a k-step aggregated transition from the front of the n-step buffer.
        Only removes the FIRST item from buffer (allows overlapping n-step returns).
        
        Args:
            k: Number of steps to aggregate (1 <= k <= n_step)
        """
        if k <= 0 or k > len(self._n_step_buffer):
            return

        items = list(self._n_step_buffer)[:k]
        
        # First transition's state and action
        state, action, _, _, _, _ = items[0]
        # Last transition's next_state, terminated, truncated
        _, _, _, next_state, term, trunc = items[-1]

        # Accumulate rewards with discount
        reward = 0.0
        for i, (_, _, r, _, _, _) in enumerate(items):
            reward += (self.gamma**i) * r
            if i > 0:
                # Check if any intermediate transition terminated/truncated
                _, _, _, _, t, tr = items[i]
                if t or tr:
                    # Early termination - use that transition's next_state and flags
                    _, _, _, next_state, term, trunc = items[i]
                    break
            else:
                term = items[-1][4]
                trunc = items[-1][5]

        # Discount for this k-step return
        discount = self.gamma ** k

        self._push_transition(state, action, reward, next_state, term, trunc, discount)

        # Remove ONLY the first item (allows overlapping n-step returns)
        self._n_step_buffer.popleft()

    def _push_transition(self, state, action, reward, next_state, term, trunc, discount) -> None:
        """Push aggregated transition to main buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.terminated[self.position] = term
        self.truncated[self.position] = trunc
        self.discounts[self.position] = discount

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

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

        # If buffer is full, emit the oldest n_step items
        if len(self._n_step_buffer) >= self.n_step:
            self._emit(self.n_step)

        # If episode ended, flush remaining buffer: emit 1-step at a time
        if terminated or truncated:
            while self._n_step_buffer:
                self._emit(1)

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Sample random batch from memory.

        Returns:
            states, actions, rewards, next_states, terminated, truncated, discounts
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminated[indices],
            self.truncated[indices],
            self.discounts[indices],
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
            "discounts": self.discounts[:self.size].copy(),
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
        for field in ["states", "actions", "rewards", "next_states", "terminated", "truncated", "discounts"]:
            if field in state_dict:
                arr = state_dict[field]
                self.__dict__[field][:self.size] = arr[:self.size]


if __name__ == "__main__":
    # Quick test
    import numpy as np
    buf = ReplayBuffer(capacity=1000, state_dim=16, n_step=3, gamma=0.99)

    # Add some transitions
    for i in range(10):
        state = np.random.randn(16).astype(np.float32)
        action = np.random.randint(0, 3)
        reward = np.random.randn()
        next_state = np.random.randn(16).astype(np.float32)
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
    print(f"Sampled discounts shape: {batch[6].shape}")

    # Test checkpoint
    state_dict = buf.get_state_dict()
    print(f"State dict keys: {state_dict.keys()}")
    print(f"Saved size: {state_dict['size']}")

    print("Replay buffer test passed!")
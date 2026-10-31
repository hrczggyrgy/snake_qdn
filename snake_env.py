"""
Snake Environment following Gymnasium API conventions.

This module implements a discrete grid-based Snake game environment
with a 22-dimensional observation space and 3 discrete actions.

Observation Space (22-dim boolean/normalized vector):
- [0-2]: Danger straight, right, left (collision within 1 step)
- [3-6]: Current direction (left, right, up, down) - one-hot
- [7-10]: Food relative location (left, right, up, down)
- [11-12]: Food displacement (dx, dy) normalized to [-1, 1]
- [13-16]: Wall distances (left, right, up, down) normalized to [0, 1]
- [17]: Snake length / board area [0, 1]
- [18]: Timeout progress (steps since food / max steps) [0, 1]
- [19-21]: Reachable space after straight/right/left actions [0, 1]

Action Space (3 discrete):
- 0: Straight
- 1: Turn Right
- 2: Turn Left

Rewards:
- Eating food: +10
- Game Over: -10
- Step penalty: -0.01 (configurable)

Collision Rules:
- allow_tail_entry: If True, moving into the tail cell is allowed when the snake
  doesn't grow that step (tail moves away). This is the standard Snake rule.
  If False, the tail cell is always treated as collision.

Uses instance-local random generators for RNG isolation.
"""

import random
from collections import deque
from typing import Any

import numpy as np

from observation_schema import OBSERVATION_SIZE


class SnakeGameAI:
    """
    Snake Game Environment compatible with Gymnasium API.

    Observation Space (11-dim boolean/normalized vector):
    - [0-2]: Danger straight, right, left (collision within 1 step)
    - [3-6]: Current direction (left, right, up, down) - one-hot
    - [7-10]: Food relative location (left, right, up, down)

    Action Space (3 discrete):
    - 0: Straight
    - 1: Turn Right
    - 2: Turn Left

    Rewards:
    - Eating food: +10
    - Game Over: -10
    - Step penalty: -0.01 (configurable)

    Collision Rules:
    - allow_tail_entry: If True, moving into the tail cell is allowed when the snake
      doesn't grow that step (tail moves away). This is the standard Snake rule.
      If False, the tail cell is always treated as collision.
    """

    def __init__(
        self,
        grid_size: int = 20,
        step_penalty: float = -0.01,
        food_reward: float = 10.0,
        death_penalty: float = -10.0,
        render_mode: str | None = None,
        seed: int | None = None,
        allow_tail_entry: bool = True,
    ):
        self.grid_size = grid_size
        self.step_penalty = step_penalty
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.render_mode = render_mode
        self.allow_tail_entry = allow_tail_entry

        # Instance-local random generators for RNG isolation
        self._rng = random.Random()
        self._np_rng = np.random.default_rng()

        if seed is not None:
            self.seed(seed)

        # Action mapping: 0=straight, 1=right, 2=left
        self.action_space_n = 3
        self.observation_space_shape = (OBSERVATION_SIZE,)

        # Direction vectors: [right, down, left, up] (clockwise)
        self.directions = np.array(
            [
                [1, 0],  # right
                [0, 1],  # down
                [-1, 0],  # left
                [0, -1],  # up
            ]
        )

        # For rendering
        self.cell_size = 20
        self.window_size = self.grid_size * self.cell_size

        # food can be None when board is full (victory)
        self.food: tuple[int, int] | None = None

        # Precompute empty cells for efficient food placement
        self._empty_cells: list[tuple[int, int]] = []

        self.reset()

    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility (instance-local)."""
        self._seed = seed
        self._rng.seed(seed)
        self._np_rng = np.random.default_rng(seed)

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed(seed)

        # Initialize snake in center, moving right
        center = self.grid_size // 2

        # Ensure initial snake fits within grid (minimum grid_size=3)
        # Snake length is 3, so we need at least 3 cells horizontally
        if self.grid_size < 3:
            raise ValueError(f"grid_size must be >= 3, got {self.grid_size}")

        # Place snake horizontally in center row, ensuring all segments are in bounds
        # For grid_size=3: center=1, snake at (1,1), (0,1) - but we need 3 segments
        # So we place at (1, center), (0, center), and wrap or adjust
        start_x = max(2, center)  # Ensure we have room for 3 segments going left
        self.snake = deque([(start_x, center), (start_x - 1, center), (start_x - 2, center)])
        self.direction = 0  # 0=right, 1=down, 2=left, 3=up
        self.score = 0
        self.steps_since_food = 0
        self.max_steps_without_food = self.grid_size * self.grid_size  # Prevent infinite loops

        # Rebuild empty cells and place food
        self._rebuild_empty_cells()
        self._place_food()
        self.game_over = False

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _rebuild_empty_cells(self) -> None:
        """Rebuild the list of empty cells for efficient food placement."""
        snake_set = set(self.snake)
        self._empty_cells = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in snake_set
        ]

    def _place_food(self) -> None:
        """Place food at random empty position using instance-local RNG."""
        # Rebuild empty cells if snake was manually modified (fallback for tests)
        if not hasattr(self, '_empty_cells') or len(self._empty_cells) + len(self.snake) != self.grid_size * self.grid_size:
            self._rebuild_empty_cells()

        if self._empty_cells:
            self.food = self._rng.choice(self._empty_cells)
            # Remove food position from empty cells
            self._empty_cells.remove(self.food)
        else:
            # Board is full - snake has won (filled entire grid)
            # This is a terminal success state
            self.food = None

    def is_board_full(self) -> bool:
        """Check if snake fills the entire board."""
        return len(self.snake) >= self.grid_size * self.grid_size

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: 0=straight, 1=right, 2=left

        Returns:
            observation, reward, terminated, truncated, info
            info includes "termination_reason" when episode ends (one of: "wall", "self", "timeout", "victory")
        """
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()

        # Validate action
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action: {action}. Must be 0 (straight), 1 (right), or 2 (left).")

        # Update direction based on action
        self._update_direction(action)

        # Calculate new head position
        head = self.snake[0]
        new_head = (
            head[0] + self.directions[self.direction][0],
            head[1] + self.directions[self.direction][1],
        )

        # Check collisions
        reward = self.step_penalty
        terminated = False
        truncated = False
        termination_reason = None

        # Wall collision
        if (
            new_head[0] < 0
            or new_head[0] >= self.grid_size
            or new_head[1] < 0
            or new_head[1] >= self.grid_size
        ):
            reward = self.death_penalty
            terminated = True
            self.game_over = True
            termination_reason = "wall"

        # Self collision (with tail entry rule)
        elif new_head in self.snake:
            # Allow entering the tail cell if it will move away (snake not growing)
            tail = self.snake[-1]
            will_grow = (self.food is not None and new_head == self.food)

            if self.allow_tail_entry and new_head == tail and not will_grow:
                # Tail entry allowed - treat as valid move
                pass
            else:
                reward = self.death_penalty
                terminated = True
                self.game_over = True
                termination_reason = "self"

        if not terminated:
            # Valid move
            self.snake.appendleft(new_head)
            self.steps_since_food += 1

            # Update empty cells: head occupies new cell
            if new_head in self._empty_cells:
                self._empty_cells.remove(new_head)

            # Check food
            if new_head == self.food:
                self.score += 1
                reward = self.food_reward
                self.steps_since_food = 0
                self._place_food()

                # Check for victory (board full)
                if self.is_board_full():
                    terminated = True
                    self.game_over = True
                    reward += self.food_reward  # Bonus for winning
                    termination_reason = "victory"
            else:
                # Remove tail and add it back to empty cells
                tail = self.snake.pop()
                # Don't add tail to empty_cells if it's the same as new_head (tail-entry move)
                if tail != new_head:
                    self._empty_cells.append(tail)

            # Truncation: too many steps without food (time limit)
            if self.steps_since_food >= self.max_steps_without_food:
                truncated = True
                reward = self.death_penalty
                self.game_over = True
                termination_reason = "timeout"
                reward = self.death_penalty
                self.game_over = True
                termination_reason = "timeout"

        observation = self._get_observation()
        info = self._get_info()
        if termination_reason is not None:
            info["termination_reason"] = termination_reason

        return observation, reward, terminated, truncated, info

    def _update_direction(self, action: int) -> None:
        """Update snake direction based on action."""
        # Action: 0=straight, 1=right, 2=left
        if action == 1:  # Turn right
            self.direction = (self.direction + 1) % 4
        elif action == 2:  # Turn left
            self.direction = (self.direction - 1) % 4
        # action == 0: keep current direction

    def _get_observation(self) -> np.ndarray:
        """Return 22-dimensional observation vector as float32 array."""
        hx, hy = self.snake[0]
        n = self.grid_size
        scale = n - 1

        # Relative action order: straight, right, left.
        headings = (
            self.direction,
            (self.direction + 1) % 4,
            (self.direction - 1) % 4,
        )

        next_positions = [
            (
                hx + int(self.directions[d][0]),
                hy + int(self.directions[d][1]),
            )
            for d in headings
        ]

        dangers = [
            float(self._is_collision(pos))
            for pos in next_positions
        ]

        # Preserve the original heading order: left, right, up, down.
        direction_flags = [
            float(self.direction == d)
            for d in (2, 0, 3, 1)
        ]

        if self.food is None:
            food_flags = [0.0] * 4
            food_dx = food_dy = 0.0
        else:
            fx, fy = self.food
            food_flags = [
                float(fx < hx),
                float(fx > hx),
                float(fy < hy),
                float(fy > hy),
            ]
            food_dx = (fx - hx) / scale
            food_dy = (fy - hy) / scale

        # Terminal wall collisions can put the head outside the board.
        wall_distances = np.clip(
            [
                hx / scale,
                (n - 1 - hx) / scale,
                hy / scale,
                (n - 1 - hy) / scale,
            ],
            0.0,
            1.0,
        ).tolist()

        length_fraction = len(self.snake) / (n * n)
        timeout_progress = min(
            1.0,
            self.steps_since_food / self.max_steps_without_food,
        )

        # Terminal states do not need hypothetical future moves.
        if self.game_over:
            reachable_space = [0.0] * 3
        else:
            reachable_space = [
                self._reachable_space_after_action(action)
                for action in range(3)
            ]

        obs_list = (
            dangers
            + direction_flags
            + food_flags
            + [food_dx, food_dy]
            + wall_distances
            + [length_fraction, timeout_progress]
            + reachable_space
        )
        obs = np.array(obs_list, dtype=np.float32)

        assert obs.shape == self.observation_space_shape
        return obs  # type: ignore[no-any-return]

    def _reachable_space_after_action(self, action: int) -> float:
        """Compute reachable space fraction after taking an action.

        Simulates only the next body configuration, then flood-fills free space.
        Does not call step(), modify the live snake, or consume random numbers.

        Args:
            action: 0=straight, 1=right, 2=left

        Returns:
            Fraction of board reachable from new head (0.0 to 1.0).
        """
        if action not in (0, 1, 2):
            raise ValueError(f"Invalid action: {action}")

        direction_delta = (0, 1, -1)[action]
        heading = (self.direction + direction_delta) % 4

        hx, hy = self.snake[0]
        dx, dy = self.directions[heading]
        new_head = (hx + int(dx), hy + int(dy))

        # Uses the same collision and tail-entry rules as the game.
        if self._is_collision(new_head):
            return 0.0

        will_grow = self.food is not None and new_head == self.food

        # Build a hypothetical post-move body without changing self.snake.
        next_snake = list(self.snake)
        if not will_grow:
            next_snake.pop()
        next_snake.insert(0, new_head)

        # The new head is the flood-fill starting cell, not an obstacle.
        blocked = set(next_snake[1:])
        visited = {new_head}
        queue = deque([new_head])
        n = self.grid_size

        while queue:
            x, y = queue.popleft()

            for nx, ny in (
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ):
                pos = (nx, ny)

                if not (0 <= nx < n and 0 <= ny < n):
                    continue
                if pos in blocked or pos in visited:
                    continue

                visited.add(pos)
                queue.append(pos)

        return len(visited) / (n * n)

    def _is_collision(self, pos: tuple[int, int]) -> bool:
        """Check if position results in collision."""
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        if pos in self.snake:
            # Allow entering the tail cell if it will move away (snake not growing)
            tail = self.snake[-1]
            will_grow = (self.food is not None and pos == self.food)
            if self.allow_tail_entry and pos == tail and not will_grow:
                return False
            return True
        return False

    def _get_info(self) -> dict[str, Any]:
        """Get auxiliary info dict."""
        return {
            "score": self.score,
            "snake_length": len(self.snake),
            "steps_since_food": self.steps_since_food,
            "head_position": self.snake[0],
            "food_position": self.food,
            "direction": self.direction,
        }

    def render(self) -> np.ndarray | None:
        """
        Render the game state as RGB array.
        Returns numpy array of shape (H, W, 3) for 'rgb_array' mode.
        """
        if self.render_mode != "rgb_array":
            return None

        # Create RGB image
        img: np.ndarray = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
        cell = self.cell_size

        # Background
        img[:, :] = (30, 30, 40)  # Dark blue-gray

        # Draw grid lines (subtle)
        for i in range(self.grid_size + 1):
            y = i * cell
            x = i * cell
            if y < self.window_size:
                img[y, :, :] = (50, 50, 60)
            if x < self.window_size:
                img[:, x, :] = (50, 50, 60)

        # Draw snake body
        for i, segment in enumerate(self.snake):
            x, y = segment
            x1, y1 = x * cell, y * cell
            x2, y2 = x1 + cell, y1 + cell

            # Head is brighter green, body darker
            if i == 0:
                color = (50, 220, 50)  # Bright green head
            else:
                intensity = max(100, 200 - i * 5)
                color = (30, intensity, 30)

            img[y1:y2, x1:x2] = color

            # Draw eyes on head
            if i == 0:
                eye_size = cell // 5
                eye_offset = cell // 4
                if self.direction == 0:  # Right
                    img[
                        y1 + eye_offset : y1 + eye_offset + eye_size,
                        x2 - eye_offset - eye_size : x2 - eye_offset,
                    ] = (255, 255, 255)
                    img[
                        y2 - eye_offset - eye_size : y2 - eye_offset,
                        x2 - eye_offset - eye_size : x2 - eye_offset,
                    ] = (255, 255, 255)
                elif self.direction == 2:  # Left
                    img[
                        y1 + eye_offset : y1 + eye_offset + eye_size,
                        x1 + eye_offset : x1 + eye_offset + eye_size,
                    ] = (255, 255, 255)
                    img[
                        y2 - eye_offset - eye_size : y2 - eye_offset,
                        x1 + eye_offset : x1 + eye_offset + eye_size,
                    ] = (255, 255, 255)
                elif self.direction == 3:  # Up
                    img[
                        y1 + eye_offset : y1 + eye_offset + eye_size,
                        x1 + eye_offset : x1 + eye_offset + eye_size,
                    ] = (255, 255, 255)
                    img[
                        y1 + eye_offset : y1 + eye_offset + eye_size,
                        x2 - eye_offset - eye_size : x2 - eye_offset,
                    ] = (255, 255, 255)
                else:  # Down
                    img[
                        y2 - eye_offset - eye_size : y2 - eye_offset,
                        x1 + eye_offset : x1 + eye_offset + eye_size,
                    ] = (255, 255, 255)
                    img[
                        y2 - eye_offset - eye_size : y2 - eye_offset,
                        x2 - eye_offset - eye_size : x2 - eye_offset,
                    ] = (255, 255, 255)

        # Draw food (red apple)
        if self.food is not None:
            fx, fy = self.food
            fx1, fy1 = fx * cell, fy * cell
            fx2, fy2 = fx1 + cell, fy1 + cell
            img[fy1:fy2, fx1:fx2] = (220, 50, 50)

            # Add shine to food
            shine_size = cell // 4
            img[fy1 + 2 : fy1 + 2 + shine_size, fx1 + 2 : fx1 + 2 + shine_size] = (255, 150, 150)

        return img

    def close(self) -> None:
        """Clean up resources."""
        pass

    def get_state_for_display(self) -> dict[str, Any]:
        """Get full state for visualization."""
        return {
            "snake": list(self.snake),
            "food": self.food,
            "direction": self.direction,
            "score": self.score,
            "game_over": self.game_over,
            "grid_size": self.grid_size,
            "observation": self._get_observation(),
        }


# Convenience function for gymnasium registration
def make_snake_env(**kwargs) -> SnakeGameAI:
    """Factory function to create Snake environment."""
    return SnakeGameAI(**kwargs)


if __name__ == "__main__":
    # Quick test
    env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    print(f"Info: {info}")

    for _ in range(10):
        action = np.random.randint(0, 3)
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Score: {info['score']}"
        )
        if terminated:
            break

    img = env.render()
    print(f"Render shape: {img.shape if img is not None else None}")
    env.close()
    print("Test passed!")

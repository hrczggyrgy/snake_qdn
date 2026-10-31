"""
Snake Environment following Gymnasium API conventions.

Implements a discrete grid-based Snake game environment
with a 14-dimensional observation space and 3 discrete actions.

Bug fixes from original:
- B1: Removed duplicate timeout block
- B2: Directions as tuple (no np.int64 leakage)
- B9: Added allow_tail_entry to EnvConfig
"""

import random
from collections import deque
from typing import Any

import numpy as np

from .config import EnvConfig


# Direction vectors as plain tuples (no numpy int64 leakage)
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))  # right, down, left, up (clockwise)


class SnakeEnv:
    """
    Snake Game Environment compatible with Gymnasium API.

    Observation Space (14-dim vector):
    - [0-2]: Danger straight, right, left (collision within 1 step)
    - [3-5]: Danger straight, right, left (collision within 2 steps) - NEW
    - [6-9]: Current direction (left, right, up, down) - one-hot
    - [10-13]: Food relative location (left, right, up, down)
    - [14]: Normalized distance to food - NEW
    - [15]: Relative tail direction (0=same, 1=right, 2=back, 3=left) - NEW

    Action Space (3 discrete):
    - 0: Straight
    - 1: Turn Right
    - 2: Turn Left

    Rewards:
    - Eating food: +10 (configurable)
    - Game Over: -10 (configurable)
    - Step penalty: -0.01 (configurable)
    - Potential-based shaping: +0.1 * (prev_dist - new_dist) - NEW
    """

    def __init__(
        self,
        config: EnvConfig | None = None,
        # Backward compatibility: accept flat kwargs
        grid_size: int | None = None,
        step_penalty: float | None = None,
        food_reward: float | None = None,
        death_penalty: float | None = None,
        render_mode: str | None = None,
        seed: int | None = None,
        allow_tail_entry: bool | None = None,
    ):
        if config is not None:
            self.config = config
        else:
            # Build config from flat kwargs for backward compatibility
            kwargs = {}
            if grid_size is not None: kwargs["grid_size"] = grid_size
            if step_penalty is not None: kwargs["step_penalty"] = step_penalty
            if food_reward is not None: kwargs["food_reward"] = food_reward
            if death_penalty is not None: kwargs["death_penalty"] = death_penalty
            if render_mode is not None: kwargs["render_mode"] = render_mode
            if seed is not None: kwargs["seed"] = seed
            if allow_tail_entry is not None: kwargs["allow_tail_entry"] = allow_tail_entry
            self.config = EnvConfig(**kwargs) if kwargs else EnvConfig()

        # Instance-local random generators for RNG isolation
        self._rng = random.Random()
        self._np_rng = np.random.default_rng()

        if self.config.seed is not None:
            self.seed(self.config.seed)

        # Action/observation space info
        self.action_space_n = 3
        self.observation_space_shape = (14,)  # Extended to 14 features

        # For rendering
        self.cell_size = 20
        self.window_size = self.config.grid_size * self.cell_size

        # Precompute empty cells for efficient food placement
        self._empty_cells: list[tuple[int, int]] = []
        # O(1) collision check: set of occupied cells
        self._occupied: set[tuple[int, int]] = set()

        # For reward shaping
        self._prev_food_dist: float = 0.0
        self._shaping_coef = 0.1

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
        center = self.config.grid_size // 2

        # Ensure initial snake fits within grid (minimum grid_size=3)
        if self.config.grid_size < 3:
            raise ValueError(f"grid_size must be >= 3, got {self.config.grid_size}")

        # Place snake horizontally in center row, ensuring all segments are in bounds
        start_x = max(2, center)  # Ensure we have room for 3 segments going left
        self.snake = deque(
            [(start_x, center), (start_x - 1, center), (start_x - 2, center)]
        )
        self.direction = 0  # 0=right, 1=down, 2=left, 3=up
        self.score = 0
        self.steps_since_food = 0
        # Scale timeout with snake length for fairness
        self.max_steps_without_food = (
            self.config.grid_size * self.config.grid_size + 2 * len(self.snake)
        )

        # Rebuild empty cells and occupied set
        self._rebuild_empty_cells()
        self._place_food()
        self.game_over = False

        # Initialize previous distance for shaping
        head = self.snake[0]
        if self.food is not None:
            self._prev_food_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        else:
            self._prev_food_dist = 0.0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _rebuild_empty_cells(self) -> None:
        """Rebuild the list of empty cells and occupied set for efficient operations."""
        snake_set = set(self.snake)
        self._occupied = snake_set
        self._empty_cells = [
            (x, y)
            for x in range(self.config.grid_size)
            for y in range(self.config.grid_size)
            if (x, y) not in snake_set
        ]

    def _place_food(self) -> None:
        """Place food at random empty position using instance-local RNG."""
        if not hasattr(self, "_empty_cells") or len(self._empty_cells) + len(self.snake) != self.config.grid_size * self.config.grid_size:
            self._rebuild_empty_cells()

        if self._empty_cells:
            self.food = self._rng.choice(self._empty_cells)
            # Remove food position from empty cells and add to occupied
            self._empty_cells.remove(self.food)
            self._occupied.add(self.food)
        else:
            # Board is full - snake has won (filled entire grid)
            self.food = None

    def is_board_full(self) -> bool:
        """Check if snake fills the entire board."""
        return len(self.snake) >= self.config.grid_size * self.config.grid_size

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

        # Calculate new head position (using tuple DIRS - no np.int64)
        head = self.snake[0]
        dx, dy = DIRS[self.direction]
        new_head = (head[0] + dx, head[1] + dy)

        # Check collisions
        reward = self.config.step_penalty
        terminated = False
        truncated = False
        termination_reason = None

        # Wall collision
        if (
            new_head[0] < 0
            or new_head[0] >= self.config.grid_size
            or new_head[1] < 0
            or new_head[1] >= self.config.grid_size
        ):
            reward = self.config.death_penalty
            terminated = True
            self.game_over = True
            termination_reason = "wall"

        # Self collision (with tail entry rule)
        elif new_head in self._occupied:
            # Allow entering the tail cell if it will move away (snake not growing)
            tail = self.snake[-1]
            will_grow = (self.food is not None and new_head == self.food)

            if self.config.allow_tail_entry and new_head == tail and not will_grow:
                # Tail entry allowed - treat as valid move
                pass
            else:
                reward = self.config.death_penalty
                terminated = True
                self.game_over = True
                termination_reason = "self"

        if not terminated:
            # Valid move
            self.snake.appendleft(new_head)
            self._occupied.add(new_head)
            self.steps_since_food += 1

            # Update empty cells: head occupies new cell
            if new_head in self._empty_cells:
                self._empty_cells.remove(new_head)

            # Check food
            if new_head == self.food:
                self.score += 1
                reward = self.config.food_reward
                self.steps_since_food = 0
                self._place_food()

                # Check for victory (board full)
                if self.is_board_full():
                    terminated = True
                    self.game_over = True
                    reward += self.config.food_reward  # Bonus for winning
                    termination_reason = "victory"
            else:
                # Remove tail and add it back to empty cells
                tail = self.snake.pop()
                self._occupied.remove(tail)
                # Don't add tail to empty_cells if it's the same as new_head (tail-entry move)
                if tail != new_head:
                    self._empty_cells.append(tail)

            # Truncation: too many steps without food (time limit)
            # This is a deliberate loop penalty - timeout gives death penalty
            if self.steps_since_food >= self.max_steps_without_food:
                truncated = True
                reward = self.config.death_penalty
                self.game_over = True
                termination_reason = "timeout"

            # Reward shaping: potential-based (Manhattan distance to food)
            if self.food is not None and not terminated and not truncated:
                new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
                shaping_reward = self._shaping_coef * (self._prev_food_dist - new_dist)
                reward += shaping_reward
                self._prev_food_dist = new_dist

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
        """
        Get 14-dimensional observation vector:
        [danger_straight_1, danger_right_1, danger_left_1,
         danger_straight_2, danger_right_2, danger_left_2,
         dir_left, dir_right, dir_up, dir_down,
         food_left, food_right, food_up, food_down,
         food_dist_norm, tail_dir_rel]
        """
        head = self.snake[0]

        # Danger detection (one step ahead in each relative direction)
        # Straight danger (1 step)
        straight_dir = self.direction
        straight_pos = (head[0] + DIRS[straight_dir][0], head[1] + DIRS[straight_dir][1])
        danger_straight_1 = self._is_collision(straight_pos)

        # Right danger (1 step)
        right_dir = (self.direction + 1) % 4
        right_pos = (head[0] + DIRS[right_dir][0], head[1] + DIRS[right_dir][1])
        danger_right_1 = self._is_collision(right_pos)

        # Left danger (1 step)
        left_dir = (self.direction - 1) % 4
        left_pos = (head[0] + DIRS[left_dir][0], head[1] + DIRS[left_dir][1])
        danger_left_1 = self._is_collision(left_pos)

        # Danger detection (two steps ahead) - NEW
        # Straight danger (2 steps)
        straight_pos_2 = (straight_pos[0] + DIRS[straight_dir][0], straight_pos[1] + DIRS[straight_dir][1])
        danger_straight_2 = self._is_collision(straight_pos_2)

        # Right danger (2 steps)
        right_pos_2 = (right_pos[0] + DIRS[right_dir][0], right_pos[1] + DIRS[right_dir][1])
        danger_right_2 = self._is_collision(right_pos_2)

        # Left danger (2 steps)
        left_pos_2 = (left_pos[0] + DIRS[left_dir][0], left_pos[1] + DIRS[left_dir][1])
        danger_left_2 = self._is_collision(left_pos_2)

        # Direction one-hot
        dir_left = 1.0 if self.direction == 2 else 0.0
        dir_right = 1.0 if self.direction == 0 else 0.0
        dir_up = 1.0 if self.direction == 3 else 0.0
        dir_down = 1.0 if self.direction == 1 else 0.0

        # Food relative position
        if self.food is None:
            # Board full - all false
            food_left = food_right = food_up = food_down = 0.0
            food_dist_norm = 0.0
        else:
            food_left = 1.0 if self.food[0] < head[0] else 0.0
            food_right = 1.0 if self.food[0] > head[0] else 0.0
            food_up = 1.0 if self.food[1] < head[1] else 0.0
            food_down = 1.0 if self.food[1] > head[1] else 0.0
            # Normalized Manhattan distance (0 to 1)
            max_dist = 2 * (self.config.grid_size - 1)
            food_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            food_dist_norm = food_dist / max_dist if max_dist > 0 else 0.0

        # Relative tail direction - NEW
        if len(self.snake) >= 2:
            tail = self.snake[-1]
            # Vector from head to tail
            dx = tail[0] - head[0]
            dy = tail[1] - head[1]
            # Determine relative direction
            if dx == 1 and dy == 0:
                tail_dir_rel = 0.0  # Same direction (right)
            elif dx == 0 and dy == 1:
                tail_dir_rel = 1.0  # Right turn (down)
            elif dx == -1 and dy == 0:
                tail_dir_rel = 2.0  # Back (left)
            elif dx == 0 and dy == -1:
                tail_dir_rel = 3.0  # Left turn (up)
            else:
                # Not adjacent - approximate
                if abs(dx) > abs(dy):
                    tail_dir_rel = 0.0 if dx > 0 else 2.0
                else:
                    tail_dir_rel = 1.0 if dy > 0 else 3.0
        else:
            tail_dir_rel = 0.0

        obs: np.ndarray = np.array(
            [
                float(danger_straight_1),
                float(danger_right_1),
                float(danger_left_1),
                float(danger_straight_2),
                float(danger_right_2),
                float(danger_left_2),
                dir_left,
                dir_right,
                dir_up,
                dir_down,
                food_left,
                food_right,
                food_up,
                food_down,
                food_dist_norm,
                tail_dir_rel,
            ],
            dtype=np.float32,
        )

        return obs

    def _is_collision(self, pos: tuple[int, int]) -> bool:
        """Check if position results in collision."""
        x, y = pos
        if x < 0 or x >= self.config.grid_size or y < 0 or y >= self.config.grid_size:
            return True
        if pos in self._occupied:
            # Allow entering the tail cell if it will move away (snake not growing)
            tail = self.snake[-1]
            will_grow = (self.food is not None and pos == self.food)
            if self.config.allow_tail_entry and pos == tail and not will_grow:
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
        if self.config.render_mode != "rgb_array":
            return None

        # Create RGB image
        img: np.ndarray = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
        cell = self.cell_size

        # Background
        img[:, :] = (30, 30, 40)  # Dark blue-gray

        # Draw grid lines (subtle)
        for i in range(self.config.grid_size + 1):
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
            "grid_size": self.config.grid_size,
            "observation": self._get_observation(),
        }

    # For curriculum - change grid size mid-training
    def set_grid_size(self, grid_size: int) -> None:
        """Change grid size (used by curriculum). Requires reset."""
        if grid_size < 3:
            raise ValueError(f"grid_size must be >= 3, got {grid_size}")
        self.config.grid_size = grid_size
        self.window_size = grid_size * self.cell_size
        # Update timeout for new grid
        self.max_steps_without_food = grid_size * grid_size + 2 * len(self.snake)
        # Rebuild empty cells cache
        self._rebuild_empty_cells()


# Convenience function for gymnasium registration
def make_snake_env(**kwargs) -> SnakeEnv:
    """Factory function to create Snake environment."""
    return SnakeEnv(**kwargs)


if __name__ == "__main__":
    # Quick test
    env = SnakeEnv(config=EnvConfig(grid_size=10, render_mode="rgb_array"))
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
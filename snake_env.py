"""
Snake Environment following Gymnasium API conventions.

This module implements a discrete grid-based Snake game environment
with an 11-dimensional observation space and 3 discrete actions.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import random
from collections import deque


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
    """
    
    def __init__(
        self,
        grid_size: int = 20,
        step_penalty: float = -0.01,
        food_reward: float = 10.0,
        death_penalty: float = -10.0,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        self.grid_size = grid_size
        self.step_penalty = step_penalty
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.render_mode = render_mode
        
        if seed is not None:
            self.seed(seed)
        
        # Action mapping: 0=straight, 1=right, 2=left
        self.action_space_n = 3
        self.observation_space_shape = (11,)
        
        # Direction vectors: [right, down, left, up] (clockwise)
        self.directions = np.array([
            [1, 0],   # right
            [0, 1],   # down
            [-1, 0],  # left
            [0, -1]   # up
        ])
        
        # For rendering
        self.cell_size = 20
        self.window_size = self.grid_size * self.cell_size
        
        self.reset()
    
    def seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed(seed)
        
        # Initialize snake in center, moving right
        center = self.grid_size // 2
        self.snake = deque([
            (center, center),
            (center - 1, center),
            (center - 2, center)
        ])
        self.direction = 0  # 0=right, 1=down, 2=left, 3=up
        self.score = 0
        self.steps_since_food = 0
        self.max_steps_without_food = self.grid_size * self.grid_size  # Prevent infinite loops
        
        self._place_food()
        self.game_over = False
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _place_food(self) -> None:
        """Place food at random empty position."""
        empty_cells = [
            (x, y) for x in range(self.grid_size) 
            for y in range(self.grid_size) 
            if (x, y) not in self.snake
        ]
        if empty_cells:
            self.food = random.choice(empty_cells)
        else:
            self.food = (0, 0)  # Should not happen in normal play
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: 0=straight, 1=right, 2=left
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Update direction based on action
        self._update_direction(action)
        
        # Calculate new head position
        head = self.snake[0]
        new_head = (
            head[0] + self.directions[self.direction][0],
            head[1] + self.directions[self.direction][1]
        )
        
        # Check collisions
        reward = self.step_penalty
        terminated = False
        
        # Wall collision
        if (new_head[0] < 0 or new_head[0] >= self.grid_size or
            new_head[1] < 0 or new_head[1] >= self.grid_size):
            reward = self.death_penalty
            terminated = True
            self.game_over = True
        
        # Self collision
        elif new_head in self.snake:
            reward = self.death_penalty
            terminated = True
            self.game_over = True
        
        else:
            # Valid move
            self.snake.appendleft(new_head)
            self.steps_since_food += 1
            
            # Check food
            if new_head == self.food:
                self.score += 1
                reward = self.food_reward
                self.steps_since_food = 0
                self._place_food()
            else:
                self.snake.pop()  # Remove tail
            
            # Truncation: too many steps without food
            if self.steps_since_food >= self.max_steps_without_food:
                terminated = True
                reward = self.death_penalty
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
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
        Get 11-dimensional observation vector:
        [danger_straight, danger_right, danger_left,
         dir_left, dir_right, dir_up, dir_down,
         food_left, food_right, food_up, food_down]
        """
        head = self.snake[0]
        
        # Danger detection (one step ahead in each relative direction)
        # Straight danger
        straight_dir = self.direction
        straight_pos = (
            head[0] + self.directions[straight_dir][0],
            head[1] + self.directions[straight_dir][1]
        )
        danger_straight = self._is_collision(straight_pos)
        
        # Right danger
        right_dir = (self.direction + 1) % 4
        right_pos = (
            head[0] + self.directions[right_dir][0],
            head[1] + self.directions[right_dir][1]
        )
        danger_right = self._is_collision(right_pos)
        
        # Left danger
        left_dir = (self.direction - 1) % 4
        left_pos = (
            head[0] + self.directions[left_dir][0],
            head[1] + self.directions[left_dir][1]
        )
        danger_left = self._is_collision(left_pos)
        
        # Direction one-hot
        dir_left = 1.0 if self.direction == 2 else 0.0
        dir_right = 1.0 if self.direction == 0 else 0.0
        dir_up = 1.0 if self.direction == 3 else 0.0
        dir_down = 1.0 if self.direction == 1 else 0.0
        
        # Food relative position
        food_left = 1.0 if self.food[0] < head[0] else 0.0
        food_right = 1.0 if self.food[0] > head[0] else 0.0
        food_up = 1.0 if self.food[1] < head[1] else 0.0
        food_down = 1.0 if self.food[1] > head[1] else 0.0
        
        obs = np.array([
            float(danger_straight),
            float(danger_right),
            float(danger_left),
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            food_left,
            food_right,
            food_up,
            food_down
        ], dtype=np.float32)
        
        return obs
    
    def _is_collision(self, pos: Tuple[int, int]) -> bool:
        """Check if position results in collision."""
        x, y = pos
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        if pos in self.snake:
            return True
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get auxiliary info dict."""
        return {
            'score': self.score,
            'snake_length': len(self.snake),
            'steps_since_food': self.steps_since_food,
            'head_position': self.snake[0],
            'food_position': self.food,
            'direction': self.direction
        }
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the game state as RGB array.
        Returns numpy array of shape (H, W, 3) for 'rgb_array' mode.
        """
        if self.render_mode != 'rgb_array':
            return None
        
        # Create RGB image
        img = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
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
                    img[y1+eye_offset:y1+eye_offset+eye_size, x2-eye_offset-eye_size:x2-eye_offset] = (255, 255, 255)
                    img[y2-eye_offset-eye_size:y2-eye_offset, x2-eye_offset-eye_size:x2-eye_offset] = (255, 255, 255)
                elif self.direction == 2:  # Left
                    img[y1+eye_offset:y1+eye_offset+eye_size, x1+eye_offset:x1+eye_offset+eye_size] = (255, 255, 255)
                    img[y2-eye_offset-eye_size:y2-eye_offset, x1+eye_offset:x1+eye_offset+eye_size] = (255, 255, 255)
                elif self.direction == 3:  # Up
                    img[y1+eye_offset:y1+eye_offset+eye_size, x1+eye_offset:x1+eye_offset+eye_size] = (255, 255, 255)
                    img[y1+eye_offset:y1+eye_offset+eye_size, x2-eye_offset-eye_size:x2-eye_offset] = (255, 255, 255)
                else:  # Down
                    img[y2-eye_offset-eye_size:y2-eye_offset, x1+eye_offset:x1+eye_offset+eye_size] = (255, 255, 255)
                    img[y2-eye_offset-eye_size:y2-eye_offset, x2-eye_offset-eye_size:x2-eye_offset] = (255, 255, 255)
        
        # Draw food (red apple)
        fx, fy = self.food
        fx1, fy1 = fx * cell, fy * cell
        fx2, fy2 = fx1 + cell, fy1 + cell
        img[fy1:fy2, fx1:fx2] = (220, 50, 50)
        
        # Add shine to food
        shine_size = cell // 4
        img[fy1+2:fy1+2+shine_size, fx1+2:fx1+2+shine_size] = (255, 150, 150)
        
        return img
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def get_state_for_display(self) -> Dict[str, Any]:
        """Get full state for visualization."""
        return {
            'snake': list(self.snake),
            'food': self.food,
            'direction': self.direction,
            'score': self.score,
            'game_over': self.game_over,
            'grid_size': self.grid_size,
            'observation': self._get_observation()
        }


# Convenience function for gymnasium registration
def make_snake_env(**kwargs) -> SnakeGameAI:
    """Factory function to create Snake environment."""
    return SnakeGameAI(**kwargs)


if __name__ == "__main__":
    # Quick test
    env = SnakeGameAI(grid_size=10, render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    print(f"Info: {info}")
    
    for _ in range(10):
        action = np.random.randint(0, 3)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Score: {info['score']}")
        if terminated:
            break
    
    img = env.render()
    print(f"Render shape: {img.shape if img is not None else None}")
    env.close()
    print("Test passed!")
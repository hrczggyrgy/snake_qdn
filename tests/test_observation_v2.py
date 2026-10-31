"""
Tests for the new 22-dimensional observation schema (v2).
"""

from collections import deque

import numpy as np
import pytest

from observation_schema import OBSERVATION_SIZE
from snake_env import SnakeGameAI


def test_observation_shape_and_ranges():
    env = SnakeGameAI(grid_size=10)
    obs, _ = env.reset(seed=42)

    assert obs.shape == (OBSERVATION_SIZE,)
    assert obs.dtype == np.float32
    assert np.isfinite(obs).all()

    # Original 11 features: 0 to 1
    assert np.all((obs[:11] >= 0) & (obs[:11] <= 1))
    # Food displacement: -1 to 1
    assert np.all((obs[11:13] >= -1) & (obs[11:13] <= 1))
    # Remaining features (wall distances, length, timeout, reachable space): 0 to 1
    assert np.all((obs[13:] >= 0) & (obs[13:] <= 1))


def test_food_displacement_and_wall_distances():
    env = SnakeGameAI(grid_size=5)
    env.snake = deque([(2, 2), (1, 2), (0, 2)])
    env.direction = 0  # Right
    env.food = (4, 0)

    obs = env._get_observation()

    # Food at (4, 0), head at (2, 2)
    # dx = (4-2)/(5-1) = 2/4 = 0.5
    # dy = (0-2)/(5-1) = -2/4 = -0.5
    np.testing.assert_allclose(obs[11:13], [0.5, -0.5])

    # Wall distances from head at (2, 2) in 5x5 grid:
    # left = 2/4 = 0.5, right = (4-2)/4 = 0.5
    # up = 2/4 = 0.5, down = (4-2)/4 = 0.5
    np.testing.assert_allclose(obs[13:17], [0.5] * 4)

    # Snake length = 3, board area = 25
    np.testing.assert_allclose(obs[17], 3 / 25)


def test_space_calculation_does_not_mutate_environment():
    env = SnakeGameAI(grid_size=5)
    before_snake = list(env.snake)
    before_food = env.food
    before_direction = env.direction
    before_rng = env._rng.getstate()

    for action in range(3):
        value = env._reachable_space_after_action(action)
        assert 0.0 <= value <= 1.0

    assert list(env.snake) == before_snake
    assert env.food == before_food
    assert env.direction == before_direction
    assert env._rng.getstate() == before_rng


def test_tail_entry_rule():
    env = SnakeGameAI(grid_size=5)
    # Snake: head at (2,2), body at (2,3), (1,3), (1,2) - tail at (1,2)
    # Direction = 3 (Up), so left turn (action=2) goes to (1,2) = tail
    env.snake = deque([(2, 2), (2, 3), (1, 3), (1, 2)])
    env.direction = 3  # Up
    env.food = (4, 4)

    # With tail entry allowed, left turn should have reachable space
    env.allow_tail_entry = True
    assert env._reachable_space_after_action(2) > 0

    # With tail entry forbidden, left turn should be collision (0 reachable space)
    env.allow_tail_entry = False
    assert env._reachable_space_after_action(2) == 0


def test_wall_collision_has_zero_reachable_space():
    env = SnakeGameAI(grid_size=5)
    # Head at (4, 2), direction = 0 (Right) - wall straight ahead
    env.snake = deque([(4, 2), (3, 2), (2, 2)])
    env.direction = 0

    assert env._reachable_space_after_action(0) == 0.0


def test_food_directions_still_work():
    """Ensure original food direction features (indices 7-10) still work correctly."""
    env = SnakeGameAI(grid_size=10)
    env.snake = deque([(5, 5), (4, 5), (3, 5)])
    env.direction = 0  # Right
    env.food = (7, 3)  # Right and Up from head

    obs = env._get_observation()

    # Food is at (7, 3), head at (5, 5)
    # food_left (fx < hx): 7 < 5 = False -> 0.0
    # food_right (fx > hx): 7 > 5 = True -> 1.0
    # food_up (fy < hy): 3 < 5 = True -> 1.0
    # food_down (fy > hy): 3 > 5 = False -> 0.0
    assert obs[7] == 0.0  # food_left
    assert obs[8] == 1.0  # food_right
    assert obs[9] == 1.0  # food_up
    assert obs[10] == 0.0  # food_down


def test_direction_one_hot_still_works():
    """Ensure original direction one-hot (indices 3-6) still works correctly."""
    env = SnakeGameAI(grid_size=10)
    env.snake = deque([(5, 5), (4, 5), (3, 5)])
    env.food = (7, 7)

    # Direction order in observation: left(2), right(0), up(3), down(1)
    # Indices: 3=left, 4=right, 5=up, 6=down
    for direction, expected_idx in [(0, 4), (1, 6), (2, 3), (3, 5)]:
        env.direction = direction
        obs = env._get_observation()
        for i in range(3, 7):
            assert obs[i] == (1.0 if i == expected_idx else 0.0), f"Failed for direction {direction}"


def test_danger_detection_still_works():
    """Ensure original danger detection (indices 0-2) still works correctly."""
    env = SnakeGameAI(grid_size=5)
    # Snake at right edge, facing right
    env.snake = deque([(4, 2), (3, 2), (2, 2)])
    env.direction = 0  # Right
    env.food = (0, 0)

    obs = env._get_observation()

    # Straight danger (wall): should be 1.0
    assert obs[0] == 1.0
    # Right danger (down): should be 0.0 (no wall/body at (4,3))
    assert obs[1] == 0.0
    # Left danger (up): should be 0.0 (no wall/body at (4,1))
    assert obs[2] == 0.0


def test_game_over_returns_zero_reachable_space():
    """When game_over is True, reachable space should be all zeros."""
    env = SnakeGameAI(grid_size=10)
    env.snake = deque([(5, 5), (4, 5), (3, 5)])
    env.food = (7, 7)
    env.game_over = True

    obs = env._get_observation()

    # Reachable space indices 19, 20, 21 should be 0
    assert obs[19] == 0.0
    assert obs[20] == 0.0
    assert obs[21] == 0.0


def test_timeout_progress():
    """Test timeout progress feature (index 18)."""
    env = SnakeGameAI(grid_size=10)
    env.snake = deque([(5, 5), (4, 5), (3, 5)])
    env.food = (7, 7)
    env.steps_since_food = 50
    env.max_steps_without_food = 100

    obs = env._get_observation()

    # timeout_progress = min(1.0, 50/100) = 0.5
    assert obs[18] == 0.5

    # Test saturation at 1.0
    env.steps_since_food = 150
    obs = env._get_observation()
    assert obs[18] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

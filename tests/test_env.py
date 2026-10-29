"""
Tests for the Snake environment.
"""

from collections import deque

import numpy as np
import pytest

from snake_env import SnakeGameAI


class TestSnakeEnvironment:
    """Tests for SnakeGameAI environment."""

    def test_reset_standard_grid(self):
        """Test reset works on standard grid."""
        env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
        obs, info = env.reset(seed=42)

        assert obs.shape == (11,), f"Expected obs shape (11,), got {obs.shape}"
        assert "score" in info
        assert info["score"] == 0
        env.close()

    def test_step_basic(self):
        """Test basic step functionality."""
        env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
        obs, info = env.reset(seed=42)

        for _ in range(5):
            action = np.random.randint(0, 3)
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (11,)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            if terminated or truncated:
                break

        env.close()

    def test_render(self):
        """Test render returns valid RGB array."""
        env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
        env.reset()

        img = env.render()
        assert img is not None
        assert img.shape == (200, 200, 3), f"Expected (200, 200, 3), got {img.shape}"
        assert img.dtype == np.uint8

        env.close()

    def test_minimum_grid_size_3x3(self):
        """Test minimum grid size (3x3) works."""
        env = SnakeGameAI(grid_size=3, render_mode="rgb_array")
        obs, info = env.reset(seed=42)

        assert obs.shape == (11,)
        for seg in env.snake:
            assert 0 <= seg[0] < 3 and 0 <= seg[1] < 3, f"Snake segment {seg} out of bounds"

        env.close()

    def test_grid_size_4x4(self):
        """Test 4x4 grid works."""
        env = SnakeGameAI(grid_size=4, render_mode="rgb_array")
        obs, info = env.reset(seed=42)

        for seg in env.snake:
            assert 0 <= seg[0] < 4 and 0 <= seg[1] < 4

        env.close()

    def test_grid_size_below_3_raises(self):
        """Test grid_size < 3 raises ValueError."""
        with pytest.raises(ValueError, match="grid_size must be >= 3"):
            SnakeGameAI(grid_size=2)

    def test_seed_reproducibility(self):
        """Test seed produces reproducible results."""
        # Create environments with seed, but don't call reset() again
        env1 = SnakeGameAI(grid_size=10, seed=42)
        env2 = SnakeGameAI(grid_size=10, seed=42)

        # The constructor already calls reset(), so we can just check the state
        obs1 = env1._get_observation()
        obs2 = env2._get_observation()

        assert np.array_equal(obs1, obs2)
        assert list(env1.snake) == list(env2.snake)
        assert env1.food == env2.food

        env1.close()
        env2.close()

    def test_wall_collision_terminated(self):
        """Test wall collision returns terminated=True."""
        env = SnakeGameAI(grid_size=5)
        env.snake = deque([(0, 0), (1, 0), (2, 0)])
        env.direction = 2  # Left
        env.food = (4, 4)

        obs, reward, terminated, truncated, info = env.step(0)  # Straight (left into wall)

        assert terminated is True
        assert truncated is False
        assert reward == env.death_penalty

        env.close()

    def test_self_collision_terminated(self):
        """Test self collision returns terminated=True."""
        env = SnakeGameAI(grid_size=5)
        # Create a loop: head at (2,2), body at (2,1), (2,0), (1,0), (1,1), (1,2)
        env.snake = deque([(2, 2), (2, 1), (2, 0), (1, 0), (1, 1), (1, 2)])
        env.direction = 3  # Up
        env.food = (4, 4)

        obs, reward, terminated, truncated, info = env.step(0)  # Straight (up into body at 2,1)

        assert terminated is True
        assert truncated is False

        env.close()

    def test_food_eating_reward(self):
        """Test eating food gives food_reward."""
        env = SnakeGameAI(grid_size=5, food_reward=10.0)
        env.snake = deque([(2, 2), (1, 2), (0, 2)])
        env.direction = 0  # Right
        env.food = (3, 2)  # Right next to head

        obs, reward, terminated, truncated, info = env.step(0)  # Straight (right into food)

        assert reward == 10.0
        assert info["score"] == 1
        assert len(env.snake) == 4  # Snake grew
        assert not terminated

        env.close()

    def test_timeout_returns_truncated(self):
        """Test timeout returns truncated=True, not terminated."""
        env = SnakeGameAI(grid_size=10, step_penalty=0.0, food_reward=0.0, death_penalty=-1.0)
        env.reset()
        env.max_steps_without_food = 3
        env.food = (9, 9)  # Far away

        terminated = truncated = False
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(0)
            if terminated or truncated:
                break

        assert truncated is True, (
            f"Expected truncated=True, got terminated={terminated}, truncated={truncated}"
        )
        assert not terminated
        assert env.steps_since_food == 3
        assert env.game_over is True

        env.close()

    def test_full_board_victory(self):
        """Test filling board results in terminated=True (victory)."""
        env = SnakeGameAI(grid_size=3)

        # Fill board except one cell
        env.snake = deque([(1, 0), (2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2)])
        env.direction = 2  # Left
        env.food = (0, 0)
        env.score = 7

        obs, reward, terminated, truncated, info = env.step(0)  # Straight (left into food)

        assert terminated is True
        assert truncated is False
        assert info["score"] == 8  # 3x3 - 1 = 8 max
        assert env.food is None
        assert len(env.snake) == 9

        env.close()

    def test_food_none_observation(self):
        """Test Food=None produces all-zero food direction observations."""
        env = SnakeGameAI(grid_size=3)
        env.food = None
        obs = env._get_observation()

        # Food directions are indices 7-10
        assert obs[7] == 0.0  # food_left
        assert obs[8] == 0.0  # food_right
        assert obs[9] == 0.0  # food_up
        assert obs[10] == 0.0  # food_down

        env.close()

    def test_game_over_state(self):
        """Test game over state returns zero reward, terminated=True."""
        env = SnakeGameAI(grid_size=10)
        env.game_over = True

        obs, reward, terminated, truncated, info = env.step(0)

        assert reward == 0.0
        assert terminated is True
        assert truncated is False

        env.close()

    def test_observation_structure(self):
        """Test observation has correct structure."""
        env = SnakeGameAI(grid_size=10)
        env.reset()
        obs = env._get_observation()

        assert obs.shape == (11,)
        assert obs.dtype == np.float32

        # Danger values should be 0 or 1
        assert all(v in [0.0, 1.0] for v in obs[:3])

        # Direction one-hot: exactly one should be 1
        dir_sum = sum(obs[3:7])
        assert dir_sum == 1.0

        # Food directions: 0 or 1
        assert all(v in [0.0, 1.0] for v in obs[7:])

        env.close()

    def test_get_state_for_display(self):
        """Test get_state_for_display returns complete state."""
        env = SnakeGameAI(grid_size=10)
        env.reset()

        state = env.get_state_for_display()

        assert "snake" in state
        assert "food" in state
        assert "direction" in state
        assert "score" in state
        assert "game_over" in state
        assert "grid_size" in state
        assert "observation" in state
        assert state["observation"].shape == (11,)

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

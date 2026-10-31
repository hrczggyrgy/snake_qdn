"""
Tests for the Snake environment.
"""

from collections import deque

import numpy as np
import pytest

from observation_schema import OBSERVATION_SIZE
from snake_env import SnakeGameAI


class TestSnakeEnvironment:
    """Tests for SnakeGameAI environment."""

    def test_reset_standard_grid(self):
        """Test reset works on standard grid."""
        env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
        obs, info = env.reset(seed=42)

        assert obs.shape == (OBSERVATION_SIZE,), f"Expected obs shape ({OBSERVATION_SIZE},), got {obs.shape}"
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
            assert obs.shape == (OBSERVATION_SIZE,)
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

        assert obs.shape == (OBSERVATION_SIZE,)
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

        assert obs.shape == (OBSERVATION_SIZE,)
        assert obs.dtype == np.float32

        # Danger values should be 0 or 1
        assert all(v in [0.0, 1.0] for v in obs[:3])

        # Direction one-hot: exactly one should be 1
        dir_sum = sum(obs[3:7])
        assert dir_sum == 1.0

        # Food directions: 0 or 1
        assert all(v in [0.0, 1.0] for v in obs[7:11])

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
        assert state["observation"].shape == (OBSERVATION_SIZE,)

        env.close()

    def test_rng_isolation_independent_environments(self):
        """Test that two environments with same seed produce identical sequences."""
        # Create environments and step them identically
        env1 = SnakeGameAI(grid_size=10, seed=42)
        env2 = SnakeGameAI(grid_size=10, seed=42)

        # The constructor already calls reset(), so initial states should match
        assert list(env1.snake) == list(env2.snake)
        assert env1.food == env2.food

        # Use a fixed action sequence for both environments
        actions = [0, 1, 2, 0, 0, 1, 2, 2, 1, 0, 0, 0, 1, 1, 2, 2, 2, 0, 1, 2]
        for action in actions:
            obs1, reward1, term1, trunc1, info1 = env1.step(action)
            obs2, reward2, term2, trunc2, info2 = env2.step(action)

            assert np.array_equal(obs1, obs2), f"Observations diverged: {obs1} vs {obs2}"
            assert reward1 == reward2, f"Rewards diverged: {reward1} vs {reward2}"
            assert term1 == term2
            assert trunc1 == trunc2
            assert info1["score"] == info2["score"]
            assert list(env1.snake) == list(env2.snake)
            assert env1.food == env2.food

            if term1 or trunc1:
                break

        env1.close()
        env2.close()

    def test_rng_isolation_no_cross_contamination(self):
        """Test that using one environment doesn't affect another's RNG state."""
        env1 = SnakeGameAI(grid_size=10, seed=100)
        env2 = SnakeGameAI(grid_size=10, seed=200)

        # Step env1 a few times
        for _ in range(10):
            action = env1._rng.randint(0, 2)
            env1.step(action)

        # Step env2 a few times
        for _ in range(10):
            action = env2._rng.randint(0, 2)
            env2.step(action)

        # Create new environments with same seeds - they should match original states
        env1_ref = SnakeGameAI(grid_size=10, seed=100)
        env2_ref = SnakeGameAI(grid_size=10, seed=200)

        for _ in range(10):
            action = env1_ref._rng.randint(0, 2)
            env1_ref.step(action)

        for _ in range(10):
            action = env2_ref._rng.randint(0, 2)
            env2_ref.step(action)

        # States should match
        assert list(env1.snake) == list(env1_ref.snake)
        assert env1.food == env1_ref.food
        assert list(env2.snake) == list(env2_ref.snake)
        assert env2.food == env2_ref.food

        env1.close()
        env2.close()
        env1_ref.close()
        env2_ref.close()

    def test_reset_with_seed_reproducibility(self):
        """Test that reset(seed=X) produces reproducible initial state."""
        # Create environment without seed, then reset with seed
        env1 = SnakeGameAI(grid_size=10)
        env2 = SnakeGameAI(grid_size=10)

        obs1, info1 = env1.reset(seed=999)
        obs2, info2 = env2.reset(seed=999)

        # Initial states should match after reset with same seed
        assert np.array_equal(obs1, obs2)
        assert list(env1.snake) == list(env2.snake)
        assert env1.food == env2.food

        # Step both with same actions
        for _ in range(10):
            action = env1._rng.randint(0, 2)
            o1, r1, t1, tr1, i1 = env1.step(action)
            o2, r2, t2, tr2, i2 = env2.step(action)
            assert np.array_equal(o1, o2)
            assert r1 == r2
            assert list(env1.snake) == list(env2.snake)

        env1.close()
        env2.close()

    def test_different_seeds_produce_different_sequences(self):
        """Test that different seeds typically produce different food placement."""
        # Use a larger grid to reduce collision probability
        seeds = list(range(1, 101))
        foods = []

        for seed in seeds:
            env = SnakeGameAI(grid_size=20, seed=seed)  # 20x20 = 400 cells, snake=3, 397 available
            foods.append(env.food)
            env.close()

        # With 100 seeds on 20x20 grid, expected unique ≈ 397 * (1 - (396/397)^99) ≈ 88
        # So we expect ~88 unique, allow some variance
        unique_foods = len(set(foods))
        assert unique_foods >= 80, f"Too many food collisions: {unique_foods}/100 unique"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

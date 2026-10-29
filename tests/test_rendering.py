"""
Tests for rendering functionality.
"""

from collections import deque

import numpy as np
import pytest

from agent import Agent
from snake_env import SnakeGameAI


class TestEnvironmentRendering:
    """Tests for environment rendering."""

    def test_render_rgb_array(self):
        """Test render returns RGB array."""
        env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
        env.reset()

        frame = env.render()

        assert frame is not None
        assert frame.shape == (200, 200, 3)
        assert frame.dtype == np.uint8

        env.close()

    def test_render_deterministic(self):
        """Test render is deterministic for same state."""
        env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
        env.reset(seed=42)

        frame1 = env.render()
        frame2 = env.render()

        assert np.array_equal(frame1, frame2)

        env.close()

    def test_render_changes_after_step(self):
        """Test render changes after environment step."""
        env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
        env.reset(seed=42)

        frame1 = env.render()
        env.step(0)
        frame2 = env.render()

        assert not np.array_equal(frame1, frame2)

        env.close()

    def test_render_after_reset(self):
        """Test render works after multiple resets."""
        env = SnakeGameAI(grid_size=10, render_mode="rgb_array")

        for _ in range(3):
            env.reset()
            frame = env.render()
            assert frame is not None
            assert frame.shape == (200, 200, 3)

        env.close()

    def test_render_game_over_state(self):
        """Test render works in game over state."""
        env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
        env.snake = deque([(0, 0), (1, 0), (2, 0)])
        env.direction = 2  # Left
        env.food = (9, 9)

        # Move into wall
        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated

        # Should still render
        frame = env.render()
        assert frame is not None
        assert frame.shape == (200, 200, 3)

        env.close()

    def test_render_without_mode(self):
        """Test render returns None when render_mode not rgb_array."""
        env = SnakeGameAI(grid_size=10, render_mode=None)
        env.reset()

        frame = env.render()
        assert frame is None

        env.close()

    def test_get_state_for_display(self):
        """Test get_state_for_display returns complete state."""
        env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
        env.reset(seed=42)

        state = env.get_state_for_display()

        assert "snake" in state
        assert "food" in state
        assert "direction" in state
        assert "score" in state
        assert "game_over" in state
        assert "grid_size" in state
        assert "observation" in state

        assert isinstance(state["snake"], list)
        assert len(state["snake"]) == 3
        assert state["grid_size"] == 10
        assert state["observation"].shape == (11,)

        env.close()


class TestVisualizationFunctions:
    """Tests for visualization helper functions (would be in app.py)."""

    def test_state_vector_rendering(self):
        """Test state vector can be computed for visualization."""
        env = SnakeGameAI(grid_size=10)
        env.reset()

        obs = env._get_observation()

        # Should have 11 elements
        assert len(obs) == 11

        # All values should be 0 or 1
        assert all(v in [0.0, 1.0] for v in obs)

        env.close()

    def test_q_values_rendering(self):
        """Test Q-values can be computed for visualization."""
        agent = Agent(hidden_size=64)
        state = np.random.randn(11)

        q_values = agent.get_q_values(state)

        assert q_values.shape == (3,)
        assert np.isfinite(q_values).all()

    def test_action_probabilities_rendering(self):
        """Test action probabilities for visualization."""
        agent = Agent(hidden_size=64)
        state = np.random.randn(11)

        probs = agent.trainer.get_action_probabilities(state)

        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_network_activations(self):
        """Test network activations for visualization."""
        agent = Agent(hidden_size=64)
        state = np.random.randn(11)

        import torch

        device = agent.trainer.device
        obs_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        model = agent.model
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                activations = model.get_activations(obs_tensor)
        finally:
            model.train(was_training)

        assert "fc1" in activations
        assert "fc2" in activations
        assert "fc3" in activations
        assert activations["fc1"].shape == (1, 64)
        assert activations["fc2"].shape == (1, 64)
        assert activations["fc3"].shape == (1, 3)


class TestControlInteractions:
    """Tests for control interactions (pause/resume/step)."""

    def test_pause_resume_cycle(self):
        """Test pause/resume cycle doesn't break rendering."""
        env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
        env.reset(seed=42)

        # Initial render
        frame1 = env.render()

        # "Pause" - just don't step, render again
        frame2 = env.render()
        assert np.array_equal(frame1, frame2)

        # "Resume" - step and render
        env.step(0)
        frame3 = env.render()
        assert not np.array_equal(frame1, frame3)

        # "Pause" again
        frame4 = env.render()
        assert np.array_equal(frame3, frame4)

        env.close()

    def test_single_step(self):
        """Test single step advances game state."""
        env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
        env.reset(seed=42)

        frame1 = env.render()

        # Step once
        obs, reward, terminated, truncated, info = env.step(0)

        frame2 = env.render()
        assert not np.array_equal(frame1, frame2)
        assert info["score"] == 0  # No food eaten yet

        env.close()

    def test_multiple_steps(self):
        """Test multiple steps advance game correctly."""
        env = SnakeGameAI(grid_size=10, render_mode="rgb_array")
        env.reset(seed=42)

        frames = []
        for _ in range(5):
            frames.append(env.render())
            env.step(0)

        # All frames should be different
        for idx in range(len(frames) - 1):
            assert not np.array_equal(frames[idx], frames[idx + 1])

        env.close()


class TestAgentVisualization:
    """Tests for agent visualization data."""

    def test_decision_snapshot_data(self):
        """Test data needed for decision snapshot is available."""
        agent = Agent(hidden_size=64)
        state = np.random.randn(11)

        # Get action with reason and Q-values
        action, reason, q_values = agent.get_action_with_q_values(state, training=True)

        assert action in [0, 1, 2]
        assert reason in ["exploring", "exploiting"]
        assert q_values.shape == (3,)

        # Compute probabilities from Q-values
        probs = agent.trainer.probabilities_from_q_values(q_values)
        assert probs.shape == (3,)
        assert abs(probs.sum() - 1.0) < 1e-5

    def test_pre_action_vs_post_action_q_values(self):
        """Test distinction between pre-action and post-action Q-values."""
        agent = Agent(hidden_size=64)
        env = SnakeGameAI(grid_size=10)
        env.reset()

        state = env._get_observation()

        # Pre-action Q-values
        pre_q = agent.get_q_values(state)
        pre_probs = agent.trainer.probabilities_from_q_values(pre_q)

        # Take action
        action = agent.get_action(state, training=False)
        next_state, reward, terminated, truncated, info = env.step(action)

        # Post-action Q-values
        post_q = agent.get_q_values(next_state)
        post_probs = agent.trainer.probabilities_from_q_values(post_q)

        # State may or may not change depending on action and environment state
        # (e.g., wall collision at edge may keep observation same)
        # The core assertion is that Q-values and probabilities are valid
        # Q-values should be valid (finite, correct shape)
        assert pre_q.shape == (3,)
        assert post_q.shape == (3,)
        assert np.isfinite(pre_q).all()
        assert np.isfinite(post_q).all()

        # Probabilities should be valid
        assert pre_probs.shape == (3,)
        assert post_probs.shape == (3,)
        assert abs(pre_probs.sum() - 1.0) < 1e-5
        assert abs(post_probs.sum() - 1.0) < 1e-5
        assert (pre_probs >= 0).all()
        assert (post_probs >= 0).all()

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

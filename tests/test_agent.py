"""
Tests for the DQN Agent.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from agent import Agent, ReplayMemory
from observation_schema import OBSERVATION_SIZE
from snake_env import SnakeGameAI


class TestReplayMemory:
    """Tests for ReplayMemory."""

    def test_push_and_len(self):
        """Test pushing experiences and length."""
        memory = ReplayMemory(capacity=1000)

        for _ in range(10):
            memory.push(
                np.random.randn(OBSERVATION_SIZE).astype(np.float32),
                np.random.randint(0, 3),
                np.random.randn(),
                np.random.randn(OBSERVATION_SIZE).astype(np.float32),
                False,
                False,
            )

        assert len(memory) == 10

    def test_sample_shapes_and_dtypes(self):
        """Test sample returns correct shapes and dtypes."""
        memory = ReplayMemory(capacity=100)

        for _ in range(50):
            memory.push(
                np.random.randn(OBSERVATION_SIZE).astype(np.float32),
                np.random.randint(0, 3),
                np.random.randn(),
                np.random.randn(OBSERVATION_SIZE).astype(np.float32),
                bool(np.random.randint(0, 2)),
                bool(np.random.randint(0, 2)),
            )

        batch_size = 32
        states, actions, rewards, next_states, terminated, truncated = memory.sample(batch_size)

        assert states.shape == (batch_size, OBSERVATION_SIZE)
        assert states.dtype == np.float32
        assert actions.shape == (batch_size,)
        assert actions.dtype == np.int64
        assert rewards.shape == (batch_size,)
        assert rewards.dtype == np.float32
        assert next_states.shape == (batch_size, OBSERVATION_SIZE)
        assert next_states.dtype == np.float32
        assert terminated.shape == (batch_size,)
        assert terminated.dtype == bool
        assert truncated.shape == (batch_size,)
        assert truncated.dtype == bool

    def test_is_ready(self):
        """Test is_ready returns correct value."""
        memory = ReplayMemory(capacity=100)
        assert not memory.is_ready(32)

        for _ in range(32):
            memory.push(np.random.randn(OBSERVATION_SIZE), 0, 0.0, np.random.randn(OBSERVATION_SIZE), False, False)

        assert memory.is_ready(32)

    def test_sample_returns_terminated_truncated(self):
        """Test sample returns both terminated and truncated arrays."""
        memory = ReplayMemory(capacity=100)

        for _ in range(10):
            memory.push(np.random.randn(OBSERVATION_SIZE), 0, 0.0, np.random.randn(OBSERVATION_SIZE), True, False)

        states, actions, rewards, next_states, terminated, truncated = memory.sample(5)

        assert terminated.shape == (5,)
        assert truncated.shape == (5,)
        assert all(terminated)  # All True
        assert not any(truncated)  # All False


class TestAgent:
    """Tests for DQN Agent."""

    def test_agent_creation(self):
        """Test agent creation with default and custom params."""
        agent = Agent(hidden_size=64, memory_size=1000, batch_size=32)
        assert agent is not None
        assert agent.model is not None
        assert agent.trainer is not None
        assert agent.memory is not None

    def test_action_selection_training(self):
        """Test action selection in training mode."""
        agent = Agent(hidden_size=64, memory_size=1000, batch_size=32, epsilon_start=1.0)
        state = np.random.randn(OBSERVATION_SIZE)

        action = agent.get_action(state, training=True)
        assert 0 <= action <= 2

    def test_action_selection_evaluation(self):
        """Test action selection in evaluation mode (greedy)."""
        agent = Agent(hidden_size=64, memory_size=1000, batch_size=32, epsilon_start=0.0)
        state = np.random.randn(OBSERVATION_SIZE)

        action = agent.get_action(state, training=False)
        assert 0 <= action <= 2

    def test_get_action_with_reason(self):
        """Test get_action_with_reason returns action and reason."""
        agent = Agent(hidden_size=64, memory_size=1000, batch_size=32, epsilon_start=0.0)
        state = np.random.randn(OBSERVATION_SIZE)

        action, reason = agent.get_action_with_reason(state, training=False)
        assert 0 <= action <= 2
        assert reason in ["exploring", "exploiting"]

    def test_get_action_with_q_values(self):
        """Test get_action_with_q_values returns action, reason, and Q-values."""
        agent = Agent(hidden_size=64, memory_size=1000, batch_size=32, epsilon_start=0.0)
        state = np.random.randn(OBSERVATION_SIZE)

        action, reason, q_values = agent.get_action_with_q_values(state, training=False)
        assert 0 <= action <= 2
        assert reason in ["exploring", "exploiting"]
        assert q_values.shape == (3,)
        assert action == int(np.argmax(q_values))

    def test_memory_store(self):
        """Test storing experiences in memory."""
        agent = Agent(hidden_size=64, memory_size=1000, batch_size=32)
        state = np.random.randn(OBSERVATION_SIZE)

        agent.remember(state, 0, 1.0, np.random.randn(OBSERVATION_SIZE), False, False)
        assert len(agent.memory) == 1

    def test_short_memory_training(self):
        """Test short memory training returns loss."""
        agent = Agent(hidden_size=64, memory_size=1000, batch_size=32)
        state = np.random.randn(OBSERVATION_SIZE)

        loss = agent.train_short_memory(state, 0, 1.0, np.random.randn(OBSERVATION_SIZE), False)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_long_memory_training_not_ready(self):
        """Test long memory training returns None when not ready."""
        agent = Agent(hidden_size=64, memory_size=1000, batch_size=32)

        loss = agent.train_long_memory()
        assert loss is None

    def test_long_memory_training_ready(self):
        """Test long memory training returns loss when ready."""
        agent = Agent(hidden_size=64, memory_size=100, batch_size=16)

        for _ in range(16):
            agent.remember(
                np.random.randn(OBSERVATION_SIZE),
                np.random.randint(0, 3),
                np.random.randn(),
                np.random.randn(OBSERVATION_SIZE),
                False,
                False,
            )

        loss = agent.train_long_memory()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0

    def test_decay_epsilon(self):
        """Test epsilon decay."""
        agent = Agent(epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.5)

        initial = agent.epsilon
        agent.decay_epsilon()
        expected = max(0.01, initial * 0.5)
        assert abs(agent.epsilon - expected) < 0.001

    def test_update_stats(self):
        """Test updating training statistics."""
        agent = Agent(hidden_size=64)

        agent.update_stats(5)
        assert agent.n_games == 1
        assert agent.scores_history == [5]
        assert agent.high_score == 5
        assert len(agent.mean_scores_history) == 1
        assert len(agent.epsilon_history) == 1

    def test_update_stats_with_loss(self):
        """Test update_stats with loss."""
        agent = Agent(hidden_size=64)

        agent.update_stats(5, loss=0.5)
        assert agent.loss_history == [0.5]
        # Note: trainer.loss_history is updated by train_step, not update_stats
        # This test just verifies agent's loss_history is updated

    def test_reset_stats(self):
        """Test reset_stats clears display statistics but keeps lifetime counters and model."""
        agent = Agent(hidden_size=64, memory_size=1000, batch_size=32)

        # Build some history
        for i in range(3):
            agent.update_stats(i)
            agent.decay_epsilon()

        assert agent.n_games == 3
        assert len(agent.scores_history) == 3
        assert len(agent.memory) == 0  # No experiences added yet
        high_score_before = agent.high_score
        epsilon_before = agent.epsilon

        agent.reset_stats()

        # Lifetime counters preserved
        assert agent.n_games == 3
        assert agent.high_score == high_score_before
        assert agent.epsilon == epsilon_before
        # Display histories cleared
        assert len(agent.scores_history) == 0
        assert len(agent.mean_scores_history) == 0
        assert len(agent.epsilon_history) == 0
        assert len(agent.loss_history) == 0
        assert len(agent.recent_terminations) == 0
        # Trainer state preserved
        assert agent.trainer.training_step == 0  # Not stepped yet in this test
        # Model weights preserved (can still act)
        action = agent.get_action(np.random.randn(OBSERVATION_SIZE), training=False)
        assert 0 <= action <= 2


class TestCheckpoint:
    """Tests for checkpoint save/load."""

    def test_save_load_roundtrip(self):
        """Test checkpoint save/load roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")

            agent = Agent(hidden_size=128, memory_size=1000, batch_size=32)
            agent.n_games = 42
            agent.high_score = 15
            agent.epsilon = 0.5

            agent.save_checkpoint(path)

            # Create new agent and load
            new_agent = Agent(hidden_size=128, memory_size=1000, batch_size=32)
            info = new_agent.load_checkpoint(path)

            assert info["episode"] == 42
            assert info["high_score"] == 15
            assert abs(info["epsilon"] - 0.5) < 0.01

            # Verify agent state restored
            assert new_agent.n_games == 42
            assert new_agent.high_score == 15
            assert abs(new_agent.epsilon - 0.5) < 0.01

            # Verify Q-values match
            test_state = np.random.randn(OBSERVATION_SIZE)
            orig_q = agent.get_q_values(test_state)
            loaded_q = new_agent.get_q_values(test_state)
            assert np.allclose(orig_q, loaded_q)

    def test_incompatible_architecture_rejected(self):
        """Test incompatible architecture raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")

            agent = Agent(hidden_size=128)
            agent.save_checkpoint(path)

            # Try to load into different architecture
            bad_agent = Agent(hidden_size=256)
            with pytest.raises(ValueError):
                bad_agent.load_checkpoint(path)

    def test_legacy_model_only_checkpoint(self):
        """Test legacy model-only checkpoint can be loaded manually."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.pth")

            agent = Agent(hidden_size=128)
            torch.save(
                {
                    "model_state_dict": agent.model.state_dict(),
                    "input_size": OBSERVATION_SIZE,
                    "hidden_size": 128,
                    "output_size": 3,
                },
                path,
            )

            legacy_agent = Agent(hidden_size=128)
            legacy_agent.model.load_state_dict(torch.load(path)["model_state_dict"])

            # Should be able to use loaded model
            test_state = np.random.randn(OBSERVATION_SIZE)
            q = legacy_agent.get_q_values(test_state)
            assert q.shape == (3,)

    def test_save_weights_only(self):
        """Test save_weights_only for inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.pth")

            agent = Agent(hidden_size=128)
            agent.save_weights_only(path)

            # Load into new agent
            new_agent = Agent(hidden_size=128)
            config = new_agent.load_weights_only(path)

            assert config.hidden_size == 128
            assert config.input_size == OBSERVATION_SIZE

            # Q-values should match
            test_state = np.random.randn(OBSERVATION_SIZE)
            orig_q = agent.get_q_values(test_state)
            loaded_q = new_agent.get_q_values(test_state)
            assert np.allclose(orig_q, loaded_q)


class TestTrainingIntegration:
    """Integration tests for training loop."""

    def test_train_agent_function(self):
        """Test train_agent function runs without error."""

        env = SnakeGameAI(grid_size=10)
        agent = Agent(hidden_size=64, memory_size=1000, batch_size=32, epsilon_start=0.1)

        # Train for a few episodes
        for _ in range(3):
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.get_action(state, training=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                agent.remember(state, action, reward, next_state, terminated, truncated)
                agent.train_short_memory(state, action, reward, next_state, terminated)

                state = next_state

            agent.decay_epsilon()
            agent.update_stats(info["score"])

        assert agent.n_games == 3
        env.close()

    def test_evaluate_agent_function(self):
        """Test evaluate_agent function."""
        from agent import evaluate_agent

        env = SnakeGameAI(grid_size=10)
        agent = Agent(hidden_size=64, memory_size=1000, batch_size=32, epsilon_start=0.0)

        results = evaluate_agent(agent, env, n_episodes=2)

        assert "mean_score" in results
        assert "std_score" in results
        assert "max_score" in results
        assert "min_score" in results
        assert "mean_steps" in results
        assert "scores" in results
        assert len(results["scores"]) == 2

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

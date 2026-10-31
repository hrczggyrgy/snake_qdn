"""
Tests for the TrainingController.
"""

import numpy as np
import pytest

from snake_qdn import ActivityMode, AgentConfig, EnvConfig, TrainingController


class TestTrainingController:
    """Tests for TrainingController."""

    def test_controller_creation(self):
        """Test controller can be created."""
        controller = TrainingController()
        assert controller is not None
        assert controller.env is not None
        assert controller.agent is not None

    def test_controller_with_custom_config(self):
        """Test controller with custom config."""
        env_config = EnvConfig(grid_size=10, step_penalty=-0.05)
        agent_config = AgentConfig(hidden_size=128, lr=0.002)

        controller = TrainingController(env_config=env_config, agent_config=agent_config)

        assert controller.env_config.grid_size == 10
        assert controller.env_config.step_penalty == -0.05
        assert controller.agent_config.hidden_size == 128
        assert controller.agent_config.lr == 0.002

    def test_env_config_validation(self):
        """Test EnvConfig validation."""
        with pytest.raises(ValueError, match="grid_size must be >= 3"):
            EnvConfig(grid_size=2)

        with pytest.raises(ValueError, match="step_penalty must be in"):
            EnvConfig(step_penalty=0.5)

        with pytest.raises(ValueError, match="food_reward must be >= 0"):
            EnvConfig(food_reward=-1.0)

        with pytest.raises(ValueError, match="death_penalty must be <= 0"):
            EnvConfig(death_penalty=1.0)

    def test_agent_config_validation(self):
        """Test AgentConfig validation."""
        with pytest.raises(ValueError, match="hidden_size must be > 0"):
            AgentConfig(hidden_size=0)

        with pytest.raises(ValueError, match="lr must be in"):
            AgentConfig(lr=1.5)

        with pytest.raises(ValueError, match="gamma must be in"):
            AgentConfig(gamma=1.5)

        with pytest.raises(ValueError, match="epsilon_end must be in"):
            AgentConfig(epsilon_start=0.5, epsilon_end=0.6)

        with pytest.raises(ValueError, match="batch_size.*must be <= memory_size"):
            AgentConfig(memory_size=100, batch_size=200)

    def test_mode_switching(self):
        """Test switching activity modes."""
        controller = TrainingController()

        controller.set_mode(ActivityMode.TRAIN)
        assert controller.mode == ActivityMode.TRAIN

        controller.set_mode(ActivityMode.EVALUATE)
        assert controller.mode == ActivityMode.EVALUATE

        controller.set_mode(ActivityMode.MANUAL)
        assert controller.mode == ActivityMode.MANUAL

    def test_paused_state(self):
        """Test paused state management."""
        controller = TrainingController()

        assert controller.paused is True  # Default

        controller.set_paused(False)
        assert controller.paused is False

        controller.set_paused(True)
        assert controller.paused is True

    def test_restart_board(self):
        """Test board restart keeps agent stats."""
        controller = TrainingController()
        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)

        # Play one episode
        controller.restart_board()
        done = False
        while not done:
            snap = controller.step()
            done = snap.game_over

        episode_after_1 = controller.episode
        assert episode_after_1 == 1

        # Restart board
        controller.restart_board()
        assert controller.episode == episode_after_1  # Episode count unchanged
        assert not controller.game_over
        assert controller.step_id == 0

    def test_clear_history(self):
        """Test clear_history resets display stats but keeps model and lifetime counters."""
        controller = TrainingController()
        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)

        # Play a few episodes
        for _ in range(3):
            controller.restart_board()
            done = False
            while not done:
                snap = controller.step()
                done = snap.game_over

        assert controller.episode == 3
        assert len(controller.agent.scores_history) == 3
        high_score_before = controller.agent.high_score
        total_steps_before = controller.agent.total_steps
        training_step_before = controller.agent.trainer.training_step

        # Clear history (display only)
        controller.clear_history()

        assert controller.episode == 0
        assert len(controller.agent.scores_history) == 0
        # Lifetime counters preserved
        assert controller.agent.high_score == high_score_before
        assert controller.agent.total_steps == total_steps_before
        assert controller.agent.trainer.training_step == training_step_before
        # Epsilon preserved
        assert controller.agent.epsilon > 0
        # Model weights preserved (can still act)
        action = controller.agent.get_action(controller.observation, training=False)
        assert 0 <= action <= 2

    def test_new_run(self):
        """Test new_run creates fresh agent."""
        controller = TrainingController()
        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)

        # Play one episode
        controller.restart_board()
        done = False
        while not done:
            snap = controller.step()
            done = snap.game_over

        assert controller.episode == 1
        assert len(controller.agent.memory) > 0

        # New run
        controller.new_run()

        assert controller.episode == 0
        assert len(controller.agent.memory) == 0
        assert controller.run_id == 1

    def test_step_train_mode(self):
        """Test step in TRAIN mode."""
        controller = TrainingController()
        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)

        # Reset to start
        controller.restart_board()

        # Take a step
        snap = controller.step()

        assert snap.action is not None
        assert snap.selection_reason in ["exploring", "exploiting"]
        assert snap.q_values is not None
        assert snap.action_probs is not None
        assert snap.epsilon >= 0
        assert snap.run_id == 0
        assert snap.episode_id == 0
        assert snap.step_id == 1

    def test_step_evaluate_mode(self):
        """Test step in EVALUATE mode (no learning)."""
        controller = TrainingController()
        controller.set_mode(ActivityMode.EVALUATE)
        controller.set_paused(False)

        initial_epsilon = controller.epsilon

        controller.restart_board()
        snap = controller.step()

        assert snap.selection_reason == "exploiting"
        # Epsilon should not change during evaluation steps
        assert controller.epsilon == initial_epsilon

    def test_step_manual_mode(self):
        """Test step in MANUAL mode with explicit action."""
        controller = TrainingController()
        controller.set_mode(ActivityMode.MANUAL)
        controller.set_paused(False)

        controller.restart_board()
        snap = controller.step(action=1)  # Turn right

        assert snap.action == 1
        assert snap.selection_reason == "manual"

    def test_step_returns_snapshot(self):
        """Test step returns proper snapshot with all fields."""
        controller = TrainingController()
        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)
        controller.restart_board()

        snap = controller.step()

        # Check all snapshot fields
        assert snap.observation is not None
        assert snap.observation.shape == (11,)
        assert snap.game_state is not None
        assert snap.q_values is not None
        assert snap.q_values.shape == (3,)
        assert snap.action_probs is not None
        assert snap.action_probs.shape == (3,)
        assert snap.action in [0, 1, 2]
        assert snap.epsilon >= 0
        assert snap.selection_reason in ["exploring", "exploiting"]
        assert isinstance(snap.game_over, bool)
        assert isinstance(snap.episode, int)
        assert isinstance(snap.total_env_steps, int)
        assert isinstance(snap.optimizer_steps, int)
        assert isinstance(snap.score, int)
        assert isinstance(snap.high_score, int)
        assert isinstance(snap.recent_scores, list)
        assert isinstance(snap.recent_mean_scores, list)
        assert isinstance(snap.recent_epsilon, list)
        assert isinstance(snap.recent_loss, list)
        assert isinstance(snap.run_id, int)
        assert isinstance(snap.episode_id, int)
        assert isinstance(snap.step_id, int)

    def test_snapshot_isolation(self):
        """Test snapshots are independent copies."""
        controller = TrainingController()
        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)
        controller.restart_board()

        snap1 = controller.get_snapshot()
        snap2 = controller.get_snapshot()

        # Arrays should be copies, not same object
        assert snap1.observation is not snap2.observation
        assert snap1.q_values is not snap2.q_values
        # But values should be equal
        assert np.array_equal(snap1.observation, snap2.observation)
        assert np.array_equal(snap1.q_values, snap2.q_values)

    def test_pre_action_snapshot(self):
        """Test snapshot contains pre-action observation/Q-values."""
        controller = TrainingController()
        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)
        controller.restart_board()

        # Get initial state
        initial_obs = controller.observation.copy()
        initial_q = controller.q_values.copy()

        # Step
        snap = controller.step()

        # Snapshot should have pre-action data
        assert np.array_equal(snap.observation, initial_obs)
        assert np.array_equal(snap.q_values, initial_q)
        # Note: observation may or may not change depending on game state
        # This test just verifies the snapshot captures pre-action state correctly

    def test_episode_completion_updates_stats(self):
        """Test episode completion updates agent stats and decays epsilon."""
        controller = TrainingController()
        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)

        initial_epsilon = controller.epsilon

        # Play until game over
        controller.restart_board()
        done = False
        while not done:
            snap = controller.step()
            done = snap.game_over

        # Episode should be counted
        assert controller.episode == 1
        assert controller.agent.n_games == 1
        # Epsilon should have decayed once
        assert controller.epsilon < initial_epsilon
        # Should be exactly one decay
        expected = max(
            controller.agent_config.epsilon_end,
            initial_epsilon * controller.agent_config.epsilon_decay,
        )
        assert abs(controller.epsilon - expected) < 0.001

    def test_evaluation_no_epsilon_decay(self):
        """Test evaluation mode doesn't decay epsilon."""
        controller = TrainingController()
        controller.set_mode(ActivityMode.EVALUATE)
        controller.set_paused(False)

        initial_epsilon = controller.epsilon

        controller.restart_board()
        done = False
        while not done:
            snap = controller.step()
            done = snap.game_over

        # Epsilon unchanged
        assert controller.epsilon == initial_epsilon
        # Episode count unchanged (agent.n_games not incremented)
        assert controller.agent.n_games == 0

    def test_update_env_config(self):
        """Test updating environment config recreates env and agent."""
        controller = TrainingController()
        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)
        controller.restart_board()

        # Play a bit
        controller.step()
        initial_run_id = controller.run_id

        # Update config
        new_config = EnvConfig(grid_size=15, step_penalty=-0.02)
        controller.update_env_config(new_config)

        assert controller.env_config.grid_size == 15
        assert controller.env_config.step_penalty == -0.02
        assert controller.run_id == initial_run_id + 1
        assert controller.episode == 0
        assert controller.step_id == 0

    def test_update_agent_config(self):
        """Test updating agent config recreates agent."""
        controller = TrainingController()

        initial_run_id = controller.run_id

        new_config = AgentConfig(hidden_size=512, lr=0.005)
        controller.update_agent_config(new_config)

        assert controller.agent_config.hidden_size == 512
        assert controller.agent_config.lr == 0.005
        assert controller.run_id == initial_run_id  # Env unchanged

    def test_checkpoint_save_load(self):
        """Test checkpoint save/load through controller."""
        import os
        import tempfile

        controller = TrainingController()
        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)

        # Play one episode
        controller.restart_board()
        done = False
        while not done:
            snap = controller.step()
            done = snap.game_over

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")

            # Save
            controller.save_checkpoint(path)

            # Create new controller and load
            new_controller = TrainingController()
            info = new_controller.load_checkpoint(path)

            assert info["episode"] == 1
            assert new_controller.agent.n_games == 1
            assert new_controller.episode_id == 1

    def test_get_snapshot_without_step(self):
        """Test get_snapshot works without stepping."""
        controller = TrainingController()
        controller.set_mode(ActivityMode.TRAIN)
        controller.set_paused(False)
        controller.restart_board()

        snap = controller.get_snapshot()

        assert snap.observation is not None
        assert snap.q_values is not None
        assert snap.step_id == 0  # No steps taken yet


class TestHeadlessTraining:
    """Tests for headless training functions."""

    def test_train_headless(self):
        """Test train_headless function."""
        from snake_qdn import train_headless

        controller = TrainingController()
        # Use small config for fast test
        controller.update_agent_config(AgentConfig(hidden_size=32, memory_size=100, batch_size=8))

        history = train_headless(controller, n_episodes=2, max_steps_per_episode=50)

        assert "scores" in history
        assert "mean_scores" in history
        assert "epsilon" in history
        assert "loss" in history
        assert len(history["scores"]) == 2
        assert controller.episode == 2

    def test_evaluate_headless(self):
        """Test evaluate_headless function."""
        from snake_qdn import evaluate_headless

        controller = TrainingController()
        controller.update_agent_config(
            AgentConfig(hidden_size=32, epsilon_start=0.0, epsilon_end=0.0)
        )

        results = evaluate_headless(controller, n_episodes=2, max_steps_per_episode=50)

        assert "mean_score" in results
        assert "scores" in results
        assert len(results["scores"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

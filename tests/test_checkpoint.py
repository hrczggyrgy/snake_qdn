"""
Tests for checkpoint persistence.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from agent import Agent
from model import Linear_QNet
from observation_schema import OBSERVATION_SIZE
from persistence import (
    CHECKPOINT_VERSION,
    CheckpointManager,
    ModelConfig,
    TrainingState,
    capture_rng_state,
    load_checkpoint,
    load_model_only,
    restore_rng_state,
    save_checkpoint,
    save_model_only,
)


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_matches_same_config(self):
        """Test matches returns True for identical configs."""
        config1 = ModelConfig(11, 256, 3, 0.1)
        config2 = ModelConfig(11, 256, 3, 0.1)
        assert config1.matches(config2)

    def test_matches_different_hidden_size(self):
        """Test matches returns False for different hidden size."""
        config1 = ModelConfig(11, 256, 3, 0.1)
        config2 = ModelConfig(11, 128, 3, 0.1)
        assert not config1.matches(config2)

    def test_matches_different_dropout(self):
        """Test matches returns False for different dropout."""
        config1 = ModelConfig(11, 256, 3, 0.1)
        config2 = ModelConfig(11, 256, 3, 0.2)
        assert not config1.matches(config2)


class TestRNGState:
    """Tests for RNG state capture/restore."""

    def test_capture_restore_python_random(self):
        """Test Python random state capture/restore."""
        import random

        # Set known state
        random.seed(42)
        _ = random.random()

        # Capture
        py_state, np_state, torch_state, cuda_state = capture_rng_state()

        # Generate more
        val2 = random.random()
        val3 = random.random()

        # Restore
        restore_rng_state(py_state, np_state, torch_state, cuda_state)

        # Should get same values
        assert random.random() == val2
        assert random.random() == val3

    def test_capture_restore_numpy_random(self):
        """Test numpy random state capture/restore."""
        np.random.seed(42)
        _ = np.random.randn()

        py_state, np_state, torch_state, cuda_state = capture_rng_state()

        val2 = np.random.randn()
        val3 = np.random.randn()

        restore_rng_state(py_state, np_state, torch_state, cuda_state)

        assert np.random.randn() == val2
        assert np.random.randn() == val3

    def test_capture_restore_torch_random(self):
        """Test torch random state capture/restore."""
        torch.manual_seed(42)
        _ = torch.randn(1).item()

        py_state, np_state, torch_state, cuda_state = capture_rng_state()

        val2 = torch.randn(1).item()
        val3 = torch.randn(1).item()

        restore_rng_state(py_state, np_state, torch_state, cuda_state)

        assert torch.randn(1).item() == val2
        assert torch.randn(1).item() == val3


class TestCheckpointSaveLoad:
    """Tests for checkpoint save/load functions."""

    def test_save_load_full_checkpoint(self):
        """Test full checkpoint save/load roundtrip."""
        model = Linear_QNet(11, 64, 3)
        target_model = Linear_QNet(11, 64, 3)
        target_model.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        training_state = TrainingState(
            episode=42,
            epsilon=0.5,
            high_score=15,
            total_steps=1000,
            training_step=500,
            scores_history=[1, 2, 3],
            mean_scores_history=[1.0, 1.5, 2.0],
            epsilon_history=[1.0, 0.9, 0.8],
            loss_history=[0.5, 0.4, 0.3],
            memory=[],  # Empty for simplicity
            memory_position=0,
            rng_state_python=None,
            rng_state_numpy=None,
            rng_state_torch=None,
            rng_state_cuda=None,
        )

        model_config = ModelConfig(11, 64, 3, 0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")

            save_checkpoint(path, model, target_model, optimizer, training_state, model_config)

            # Create new instances and load
            new_model = Linear_QNet(11, 64, 3)
            new_target = Linear_QNet(11, 64, 3)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            loaded_state, loaded_config = load_checkpoint(
                path, new_model, new_target, new_optimizer, torch.device("cpu")
            )

            # Verify model weights match
            for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=True):
                assert torch.allclose(p1, p2)

            for p1, p2 in zip(target_model.parameters(), new_target.parameters(), strict=True):
                assert torch.allclose(p1, p2)

            # Verify training state
            assert loaded_state.episode == 42
            assert loaded_state.epsilon == 0.5
            assert loaded_state.high_score == 15
            assert loaded_state.total_steps == 1000
            assert loaded_state.training_step == 500
            assert loaded_state.scores_history == [1, 2, 3]
            assert loaded_config.matches(model_config)

    def test_version_check(self):
        """Test version checking rejects future versions."""
        model = Linear_QNet(11, 64, 3)
        target_model = Linear_QNet(11, 64, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "future.pth")

            # Manually create checkpoint with future version
            checkpoint = {
                "version": CHECKPOINT_VERSION + 1,
                "model_config": {
                    "input_size": 11,
                    "hidden_size": 64,
                    "output_size": 3,
                    "dropout": 0.1,
                },
                "model_state_dict": model.state_dict(),
                "target_model_state_dict": target_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_state": {
                    "episode": 0,
                    "epsilon": 1.0,
                    "high_score": 0,
                    "total_steps": 0,
                    "training_step": 0,
                    "scores_history": [],
                    "mean_scores_history": [],
                    "epsilon_history": [],
                    "loss_history": [],
                    "memory": [],
                    "memory_position": 0,
                    "rng_state_python": None,
                    "rng_state_numpy": None,
                    "rng_state_torch": None,
                    "rng_state_cuda": None,
                },
            }
            torch.save(checkpoint, path)

            # Should raise on load
            new_model = Linear_QNet(11, 64, 3)
            new_target = Linear_QNet(11, 64, 3)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            with pytest.raises(ValueError, match="newer than supported"):
                load_checkpoint(path, new_model, new_target, new_optimizer, torch.device("cpu"))

    def test_architecture_mismatch_strict(self):
        """Test architecture mismatch raises in strict mode."""
        model = Linear_QNet(11, 64, 3)
        target_model = Linear_QNet(11, 64, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        training_state = TrainingState(
            episode=0,
            epsilon=1.0,
            high_score=0,
            total_steps=0,
            training_step=0,
            scores_history=[],
            mean_scores_history=[],
            epsilon_history=[],
            loss_history=[],
            memory=[],
            memory_position=0,
            rng_state_python=None,
            rng_state_numpy=None,
            rng_state_torch=None,
            rng_state_cuda=None,
        )
        model_config = ModelConfig(11, 64, 3, 0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")
            save_checkpoint(path, model, target_model, optimizer, training_state, model_config)

            # Try to load into different architecture
            new_model = Linear_QNet(11, 128, 3)  # Different hidden size
            new_target = Linear_QNet(11, 128, 3)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            with pytest.raises(ValueError, match="architecture mismatch"):
                load_checkpoint(
                    path, new_model, new_target, new_optimizer, torch.device("cpu"), strict=True
                )

    def test_architecture_mismatch_non_strict(self):
        """Test architecture mismatch raises even with strict=False for shape mismatches."""
        model = Linear_QNet(11, 64, 3)
        target_model = Linear_QNet(11, 64, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        training_state = TrainingState(
            episode=0,
            epsilon=1.0,
            high_score=0,
            total_steps=0,
            training_step=0,
            scores_history=[],
            mean_scores_history=[],
            epsilon_history=[],
            loss_history=[],
            memory=[],
            memory_position=0,
            rng_state_python=None,
            rng_state_numpy=None,
            rng_state_torch=None,
            rng_state_cuda=None,
        )
        model_config = ModelConfig(11, 64, 3, 0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")
            save_checkpoint(path, model, target_model, optimizer, training_state, model_config)

            # Try to load into different architecture (non-strict)
            # Shape mismatches still raise even with strict=False
            new_model = Linear_QNet(11, 128, 3)
            new_target = Linear_QNet(11, 128, 3)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            with pytest.raises(RuntimeError, match="size mismatch"):
                load_checkpoint(
                    path, new_model, new_target, new_optimizer, torch.device("cpu"), strict=False
                )

    def test_legacy_checkpoint_no_training_state(self):
        """Test loading legacy checkpoint without training_state."""
        model = Linear_QNet(11, 64, 3)
        target_model = Linear_QNet(11, 64, 3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "legacy.pth")

            # Save legacy format (v1)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_model_state_dict": target_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "training_step": 100,
                    "episode": 10,
                    "epsilon": 0.5,
                    "high_score": 5,
                    "loss_history": [0.5, 0.4],
                },
                path,
            )

            new_model = Linear_QNet(11, 64, 3)
            new_target = Linear_QNet(11, 64, 3)
            new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

            loaded_state, loaded_config = load_checkpoint(
                path, new_model, new_target, new_optimizer, torch.device("cpu")
            )

            assert loaded_state.episode == 10
            assert loaded_state.epsilon == 0.5
            assert loaded_state.high_score == 5
            assert loaded_state.training_step == 100
            assert loaded_state.loss_history == [0.5, 0.4]


class TestModelOnlyCheckpoint:
    """Tests for model-only (weights) checkpoints."""

    def test_save_load_weights_only(self):
        """Test save/load weights only."""
        model = Linear_QNet(11, 64, 3)
        model_config = ModelConfig(11, 64, 3, 0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.pth")

            save_model_only(path, model, model_config)

            new_model = Linear_QNet(11, 64, 3)
            loaded_config = load_model_only(path, new_model, torch.device("cpu"))

            # Verify weights match
            for p1, p2 in zip(model.parameters(), new_model.parameters(), strict=True):
                assert torch.allclose(p1, p2)

            assert loaded_config.matches(model_config)
            assert not new_model.training  # Should be in eval mode


class TestCheckpointManager:
    """Tests for CheckpointManager high-level API."""

    def test_manager_save_load_full(self):
        """Test CheckpointManager save/load full state."""
        agent = Agent(hidden_size=64, memory_size=1000, batch_size=32)
        agent.n_games = 42
        agent.high_score = 15
        agent.epsilon = 0.3

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")

            manager = CheckpointManager(agent, agent.trainer.device)
            manager.save_full(path)

            # Create new agent and load
            new_agent = Agent(hidden_size=64, memory_size=1000, batch_size=32)
            new_manager = CheckpointManager(new_agent, new_agent.trainer.device)
            info = new_manager.load_full(path)

            assert info["episode"] == 42
            assert info["high_score"] == 15
            assert abs(info["epsilon"] - 0.3) < 0.01

            # Verify agent state
            assert new_agent.n_games == 42
            assert new_agent.high_score == 15
            assert abs(new_agent.epsilon - 0.3) < 0.01

            # Verify Q-values match
            test_state = np.random.randn(OBSERVATION_SIZE)
            orig_q = agent.get_q_values(test_state)
            loaded_q = new_agent.get_q_values(test_state)
            assert np.allclose(orig_q, loaded_q)

    def test_manager_save_load_weights_only(self):
        """Test CheckpointManager save/load weights only."""
        agent = Agent(hidden_size=64)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.pth")

            manager = CheckpointManager(agent, agent.trainer.device)
            manager.save_weights_only(path)

            new_agent = Agent(hidden_size=64)
            new_manager = CheckpointManager(new_agent, new_agent.trainer.device)
            config = new_manager.load_weights_only(path)

            assert config.hidden_size == 64

            # Q-values match
            test_state = np.random.randn(OBSERVATION_SIZE)
            orig_q = agent.get_q_values(test_state)
            loaded_q = new_agent.get_q_values(test_state)
            assert np.allclose(orig_q, loaded_q)


class TestBackwardCompatibility:
    """Tests for backward compatibility with Agent.save/load_checkpoint."""

    def test_agent_save_load_backward_compat(self):
        """Test Agent.save_checkpoint/load_checkpoint still work."""
        from persistence import load_agent_checkpoint, save_agent_checkpoint

        agent = Agent(hidden_size=64, memory_size=1000, batch_size=32)
        agent.n_games = 42
        agent.high_score = 15
        agent.epsilon = 0.3

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "checkpoint.pth")

            save_agent_checkpoint(agent, path)

            new_agent = Agent(hidden_size=64, memory_size=1000, batch_size=32)
            info = load_agent_checkpoint(new_agent, path)

            assert info["episode"] == 42
            assert info["high_score"] == 15
            assert abs(info["epsilon"] - 0.3) < 0.01

            assert new_agent.n_games == 42
            assert new_agent.high_score == 15
            assert abs(new_agent.epsilon - 0.3) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

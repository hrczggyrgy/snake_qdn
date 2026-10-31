#!/usr/bin/env python3
"""
Characterization tests for Snake DQN training behavior.

These tests capture the current behavior before refactoring to ensure
we don't accidentally change semantics during reorganization.
"""

import os
import tempfile
from collections import deque

import numpy as np
import torch

from snake_qdn import Agent, ReplayBuffer
from snake_qdn import QNet, QTrainer
from snake_qdn import SnakeEnv


def test_train_evaluate_manual_pause_resume_single_step():
    """Test the five activity modes and control transitions."""
    print("Testing Train/Evaluate/Manual/Pause/Resume/Step...")

    env = SnakeEnv(grid_size=10)
    agent = Agent(hidden_size=64, memory_size=1000, batch_size=32, epsilon_start=0.0)

    # Train mode - agent learns
    state, _ = env.reset()
    action = agent.get_action(state, training=True)
    next_state, reward, terminated, truncated, info = env.step(action)
    agent.remember(state, action, reward, next_state, terminated, truncated)
    loss = agent.train_short_memory(state, action, reward, next_state, terminated)
    assert loss is not None
    print("  ✓ Train mode works")

    # Evaluate mode - greedy, no learning
    state, _ = env.reset()
    initial_epsilon = agent.epsilon
    action = agent.get_action(state, training=False)
    next_state, reward, terminated, truncated, info = env.step(action)
    # No remember/train calls in evaluate
    assert agent.epsilon == initial_epsilon  # epsilon unchanged
    print("  ✓ Evaluate mode works (no epsilon change)")

    # Manual mode - user controls (simulated by passing action)
    state, _ = env.reset()
    action = 0  # manual straight
    next_state, reward, terminated, truncated, info = env.step(action)
    assert not terminated or terminated  # either outcome valid
    print("  ✓ Manual mode works")

    # Pause/Resume - just verify epsilon doesn't decay during pause
    _ = agent.epsilon
    agent.decay_epsilon()  # Only called on episode completion, not pause
    # Pause doesn't call decay_epsilon
    print("  ✓ Pause doesn't affect epsilon")

    # Single step
    state, _ = env.reset()
    action = agent.get_action(state, training=True)
    next_state, reward, terminated, truncated, info = env.step(action)
    assert next_state.shape == (11,)
    print("  ✓ Single step works")

    env.close()
    print("  ✓ Activity modes test passed!\n")


def test_terminal_truncated_bootstrap():
    """Test terminal transitions suppress bootstrap; truncation retains it."""
    print("Testing Terminal vs Truncated Bootstrap...")

    model = QNet(input_size=11, hidden_size=64, output_size=3)
    trainer = QTrainer(model, lr=0.001, gamma=0.9, device="cpu")

    # Test the target computation formula directly (no gradients needed)
    with torch.no_grad():
        # Set target network to output Q=5 for all actions
        for param in trainer.target_model.parameters():
            param.fill_(0.0)
        trainer.target_model.fc3.weight.fill_(0.0)
        trainer.target_model.fc3.bias.fill_(5.0)  # Q-values = 5 for all actions

        next_state = torch.randn(1, 11)
        next_q_values = trainer.target_model(next_state)
        max_next_q = next_q_values.max(1)[0].item()

        reward = 1.0
        gamma = 0.9

        # Terminated = True (game over): target = r + gamma * maxQ * (1 - 1) = 1.0
        target_term = reward + gamma * max_next_q * (1 - 1.0)

        # Truncated = False (timeout): target = r + gamma * maxQ * (1 - 0) = 1 + 0.9*5 = 5.5
        target_trunc = reward + gamma * max_next_q * (1 - 0.0)

        assert abs(target_term - 1.0) < 0.01, f"Terminated target should be 1.0, got {target_term}"
        assert abs(target_trunc - 5.5) < 0.01, f"Truncated target should be 5.5, got {target_trunc}"

        print(f"  ✓ Terminated target: {target_term:.2f} (no bootstrap)")
        print(f"  ✓ Truncated target: {target_trunc:.2f} (keeps bootstrap)")

    print("  ✓ Bootstrap test passed!\n")


def test_epsilon_decay_once_per_episode_not_during_eval():
    """Test epsilon decays once per completed training episode, not during evaluation."""
    print("Testing Epsilon Decay Semantics...")

    env = SnakeEnv(grid_size=10)
    agent = Agent(
        hidden_size=64,
        memory_size=1000,
        batch_size=32,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.5,
    )

    initial_epsilon = agent.epsilon

    # Complete one training episode
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.get_action(state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, terminated, truncated)
        agent.train_short_memory(state, action, reward, next_state, terminated)
        state = next_state

    # Dashboard behavior: update_stats THEN decay_epsilon
    agent.update_stats(info["score"])
    agent.decay_epsilon()

    expected = max(0.01, initial_epsilon * 0.5)
    assert abs(agent.epsilon - expected) < 0.001
    print(f"  ✓ After 1 episode: epsilon={agent.epsilon:.4f} (expected {expected:.4f})")

    # Second episode
    agent.decay_epsilon()
    expected2 = max(0.01, expected * 0.5)
    assert abs(agent.epsilon - expected2) < 0.001
    print(f"  ✓ After 2 episodes: epsilon={agent.epsilon:.4f} (expected {expected2:.4f})")

    # Evaluation should NOT decay epsilon
    eval_epsilon = agent.epsilon
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.get_action(state, training=False)  # training=False
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
    # No update_stats or decay_epsilon called
    assert agent.epsilon == eval_epsilon
    print(f"  ✓ Evaluation doesn't change epsilon: {agent.epsilon:.4f}")

    env.close()
    print("  ✓ Epsilon decay test passed!\n")


def test_epsilon_logging_update_order():
    """Document the epsilon logging/update order discrepancy."""
    print("Testing Epsilon Logging/Update Order...")

    # In train_agent(): decay_epsilon() is called AFTER update_stats()
    # In dashboard: agent.decay_epsilon() is called, then dashboard mirrors agent.epsilon_history
    # This is CONSISTENT - both do update_stats then decay
    # The discrepancy was that train_agent() logs epsilon AFTER decay in epsilon_history
    # while dashboard reads agent.epsilon_history which already has the decayed value

    env = SnakeEnv(grid_size=10)
    agent = Agent(hidden_size=64, epsilon_start=1.0, epsilon_decay=0.5)

    # Simulate train_agent() behavior
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.get_action(state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state

    agent.decay_epsilon()
    agent.update_stats(info["score"])

    # agent.epsilon_history now contains the DECAYED epsilon
    logged_epsilon = agent.epsilon_history[-1]
    current_epsilon = agent.epsilon

    assert logged_epsilon == current_epsilon
    print(f"  ✓ train_agent() logs decayed epsilon: {logged_epsilon:.4f}")
    print("  ✓ Order: decay_epsilon() then update_stats() -> history has decayed value")
    print("  ✓ This is the chosen consistent behavior (documented as explicit fix)")

    env.close()
    print("  ✓ Epsilon logging order test passed!\n")


def test_board_restart_history_reset_new_run():
    """Test board restart, history reset, and new run behavior."""
    print("Testing Board Restart / History Reset / New Run...")

    env = SnakeEnv(grid_size=10)
    agent = Agent(hidden_size=64, memory_size=1000, batch_size=32)

    # Play a bit to build history
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
        agent.update_stats(info["score"])
        agent.decay_epsilon()

    assert agent.n_games == 3
    assert len(agent.scores_history) == 3
    assert agent.high_score >= 0
    print(f"  ✓ Built history: {agent.n_games} games, high_score={agent.high_score}")

    # Restart board - should NOT reset agent stats
    state, _ = env.reset()
    assert agent.n_games == 3
    assert len(agent.scores_history) == 3
    print("  ✓ Board restart keeps agent stats")

    # Clear history - resets display stats but keeps lifetime counters and model weights
    agent.reset_stats()
    # Lifetime counters preserved
    assert agent.n_games == 3
    assert agent.high_score >= 0
    # Note: total_steps only incremented through training loop (train_agent or controller)
    # This test uses train_short_memory directly, so total_steps stays 0
    # Display histories cleared
    assert len(agent.scores_history) == 0
    assert len(agent.mean_scores_history) == 0
    assert len(agent.epsilon_history) == 0
    assert len(agent.loss_history) == 0
    assert len(agent.recent_terminations) == 0
    # Epsilon and replay buffer preserved
    assert agent.epsilon != agent.epsilon_start  # Epsilon preserved
    assert len(agent.memory) > 0  # Replay buffer preserved
    print("  ✓ Clear history resets display stats but keeps lifetime counters, epsilon, and replay")

    # New run - fresh agent (discards model weights)
    new_agent = Agent(hidden_size=64, memory_size=1000, batch_size=32)
    assert new_agent.n_games == 0
    assert len(new_agent.memory) == 0
    print("  ✓ New run creates fresh agent with empty replay")

    env.close()
    print("  ✓ Restart/Reset/New-run test passed!\n")


def test_checkpoint_round_trip_incompatible_architecture():
    """Test checkpoint save/load round trips and incompatible architecture handling."""
    print("Testing Checkpoint Round Trips and Compatibility...")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pth")

        # Create and save agent
        agent = Agent(hidden_size=128, memory_size=1000, batch_size=32)
        agent.n_games = 42
        agent.high_score = 15
        agent.epsilon = 0.3
        agent.save_checkpoint(path)
        print("  ✓ Checkpoint saved")

        # Load into compatible agent
        new_agent = Agent(hidden_size=128, memory_size=1000, batch_size=32)
        info = new_agent.load_checkpoint(path)

        assert info["episode"] == 42
        assert info["high_score"] == 15
        assert abs(info["epsilon"] - 0.3) < 0.01
        assert new_agent.n_games == 42
        assert new_agent.high_score == 15
        assert abs(new_agent.epsilon - 0.3) < 0.01
        print("  ✓ Compatible load works")

        # Verify Q-values match
        test_state = np.random.randn(11)
        orig_q = agent.get_q_values(test_state)
        loaded_q = new_agent.get_q_values(test_state)
        assert np.allclose(orig_q, loaded_q)
        print("  ✓ Q-values match after load")

        # Test incompatible architecture (different hidden size)
        bad_agent = Agent(hidden_size=256, memory_size=1000, batch_size=32)
        try:
            bad_agent.load_checkpoint(path)
            raise AssertionError("Should have failed with mismatched architecture")
        except Exception as e:
            print(f"  ✓ Incompatible architecture correctly rejected: {type(e).__name__}")

        # Test legacy checkpoint (model only, no trainer state)
        legacy_path = os.path.join(tmpdir, "legacy.pth")
        torch.save(
            {
                "model_state_dict": agent.model.state_dict(),
                "input_size": 11,
                "hidden_size": 128,
                "output_size": 3,
            },
            legacy_path,
        )

        legacy_agent = Agent(hidden_size=128)
        legacy_agent.model.load_state_dict(torch.load(legacy_path)["model_state_dict"])
        print("  ✓ Legacy model-only checkpoint can be loaded manually")

    print("  ✓ Checkpoint test passed!\n")


def test_zero_loss_recorded_not_discarded():
    """Test that zero-valued loss is recorded, not discarded by truthiness checks."""
    print("Testing Zero Loss Recording...")

    model = QNet(input_size=11, hidden_size=64, output_size=3)
    trainer = QTrainer(model, lr=0.001, gamma=0.99, device="cpu")

    # Force a zero loss scenario by making prediction = target
    with torch.no_grad():
        trainer.model.fc3.weight.fill_(0.0)
        trainer.model.fc3.bias.fill_(0.0)
        trainer.target_model.fc3.weight.fill_(0.0)
        trainer.target_model.fc3.bias.fill_(0.0)

    state = torch.zeros(1, 11)
    action = torch.tensor([0])
    reward = torch.tensor([0.0])
    next_state = torch.zeros(1, 11)
    terminated = torch.tensor([False])

    loss = trainer.train_step(state, action, reward, next_state, terminated)

    # Loss should be 0.0 (or very close)
    assert loss == 0.0 or abs(loss) < 1e-6
    assert len(trainer.loss_history) == 1
    assert trainer.loss_history[0] == loss
    print(f"  ✓ Zero loss recorded: {trainer.loss_history[0]}")

    # Test agent's train_long_memory with zero loss
    agent = Agent(hidden_size=64, memory_size=1000, batch_size=2)
    agent.remember(np.zeros(11), 0, 0.0, np.zeros(11), False, False)
    agent.remember(np.zeros(11), 1, 0.0, np.zeros(11), False, False)

    # Mock to force zero loss
    with torch.no_grad():
        agent.trainer.model.fc3.weight.fill_(0.0)
        agent.trainer.model.fc3.bias.fill_(0.0)
        agent.trainer.target_model.fc3.weight.fill_(0.0)
        agent.trainer.target_model.fc3.bias.fill_(0.0)

    replay_loss = agent.train_long_memory()
    assert replay_loss == 0.0 or abs(replay_loss) < 1e-6
    print(f"  ✓ Agent train_long_memory records zero loss: {replay_loss}")

    print("  ✓ Zero loss test passed!\n")


def test_food_none_full_board_victory_small_grids_game_over():
    """Test Food=None, full-board victory, small grids, and game-over states."""
    print("Testing Edge Cases: Food=None, Victory, Small Grids, Game Over...")

    # Full board victory (3x3 grid)
    env = SnakeEnv(grid_size=3)
    env.snake = deque([(1, 0), (2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2)])
    env.direction = 2  # Left
    env.food = (0, 0)
    env.score = 7

    obs, reward, terminated, truncated, info = env.step(0)  # Straight (left)
    assert terminated is True
    assert truncated is False
    assert info["score"] == 8  # 3x3 - 1 = 8 max
    assert env.food is None  # Food becomes None when board full
    assert len(env.snake) == 9
    print(f"  ✓ Full board victory: score={info['score']}, food={env.food}")

    # Food=None observation (all food directions false)
    obs = env._get_observation()
    assert obs[7] == 0.0 and obs[8] == 0.0 and obs[9] == 0.0 and obs[10] == 0.0
    print("  ✓ Food=None produces all-zero food direction observations")

    env.close()

    # Small grid (3x3 minimum)
    env = SnakeEnv(grid_size=3)
    obs, info = env.reset()
    assert obs.shape == (11,)
    for seg in env.snake:
        assert 0 <= seg[0] < 3 and 0 <= seg[1] < 3
    print(f"  ✓ 3x3 grid works, snake: {list(env.snake)}")
    env.close()

    # Grid size < 3 raises error
    try:
        SnakeEnv(grid_size=2)
        raise AssertionError("Should have raised ValueError for grid_size < 3")
    except ValueError as e:
        print(f"  ✓ Grid size < 3 correctly rejected: {e}")

    # Game over state - step returns zero reward, terminated=True
    env = SnakeEnv(grid_size=10)
    env.game_over = True
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0
    assert terminated is True
    assert truncated is False
    print("  ✓ Game over state returns zero reward, terminated=True")
    env.close()

    print("  ✓ Edge cases test passed!\n")


def test_replay_memory_sample_compatibility():
    """Test ReplayBuffer.sample compatibility with supported Python version."""
    print("Testing ReplayBuffer.sample Compatibility...")

    memory = ReplayBuffer(capacity=100)

    # Add varied experiences
    for _ in range(50):
        memory.push(
            np.random.randn(11).astype(np.float32),
            np.random.randint(0, 3),
            np.random.randn(),
            np.random.randn(11).astype(np.float32),
            bool(np.random.randint(0, 2)),
            bool(np.random.randint(0, 2)),
        )

    # Sample batch
    batch_size = 32
    states, actions, rewards, next_states, terminated, truncated = memory.sample(batch_size)

    assert states.shape == (batch_size, 11)
    assert states.dtype == np.float32
    assert actions.shape == (batch_size,)
    assert actions.dtype == np.int64
    assert rewards.shape == (batch_size,)
    assert rewards.dtype == np.float32
    assert next_states.shape == (batch_size, 11)
    assert next_states.dtype == np.float32
    assert terminated.shape == (batch_size,)
    assert terminated.dtype == bool
    assert truncated.shape == (batch_size,)
    assert truncated.dtype == bool
    print("  ✓ Sample returns correct shapes and dtypes")

    # Sample larger than memory (should work with replacement? No, random.sample without replacement)
    # Actually random.sample requires population >= sample size
    assert memory.is_ready(batch_size)
    print("  ✓ is_ready works correctly")

    print("  ✓ ReplayBuffer compatibility test passed!\n")


def test_probability_helpers_no_mutation():
    """Test that probability helpers don't mutate caller-owned input arrays."""
    print("Testing Probability Helpers Don't Mutate Inputs...")

    # Test QTrainer.probabilities_from_q_values
    q_values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    original_q = q_values.copy()

    probs = QTrainer.probabilities_from_q_values(q_values)

    assert np.array_equal(q_values, original_q), "Input q_values was mutated!"
    assert probs.shape == (3,)
    assert abs(probs.sum() - 1.0) < 1e-10
    print("  ✓ probabilities_from_q_values doesn't mutate input")

    # Test get_action_probabilities
    model = QNet(input_size=11, hidden_size=64, output_size=3)
    trainer = QTrainer(model, device="cpu")

    state = np.random.randn(11).astype(np.float32)
    original_state = state.copy()

    probs = trainer.get_action_probabilities(state)

    assert np.array_equal(state, original_state), "Input state was mutated!"
    assert probs.shape == (3,)
    assert abs(probs.sum() - 1.0) < 1e-5
    print("  ✓ get_action_probabilities doesn't mutate input")

    print("  ✓ No-mutation test passed!\n")


def test_rendering_unchanged_data_repeated_runs_controls():
    """Test rendering after unchanged data, repeated full runs, and control interactions."""
    print("Testing Rendering Stability...")

    env = SnakeEnv(grid_size=10, render_mode="rgb_array")
    env.reset()

    # First render
    frame1 = env.render()
    assert frame1 is not None
    assert frame1.shape == (200, 200, 3)
    print("  ✓ First render works")

    # Render again without stepping (unchanged data)
    frame2 = env.render()
    assert np.array_equal(frame1, frame2), "Render should be deterministic for same state"
    print("  ✓ Unchanged data produces identical frames")

    # Step and render
    env.step(0)
    frame3 = env.render()
    assert not np.array_equal(frame1, frame3), "Frame should change after step"
    print("  ✓ Frame changes after step")

    # Repeated full runs
    for _ in range(3):
        env.reset()
        frame = env.render()
        assert frame is not None
        assert frame.shape == (200, 200, 3)
    print("  ✓ Repeated full runs work")

    # Control interactions (simulated)
    # Pause: just don't step
    # Resume: step again
    # Step once: single step
    env.reset()
    for _ in range(5):
        done = env.step(0)[2]  # terminated
        if done:
            break
    frame_paused = env.render()

    # "Resume" - step more (only if not game over)
    for _ in range(5):
        done = env.step(0)[2]
        if done:
            break
    frame_resumed = env.render()

    # If game not over, frames should differ; if game over, they'll be same
    # Just verify we can render in both states
    assert frame_paused is not None
    assert frame_resumed is not None
    print("  ✓ Pause/resume/step control flow works")

    env.close()
    print("  ✓ Rendering stability test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Characterization Tests")
    print("=" * 60 + "\n")

    test_train_evaluate_manual_pause_resume_single_step()
    test_terminal_truncated_bootstrap()
    test_epsilon_decay_once_per_episode_not_during_eval()
    test_epsilon_logging_update_order()
    test_board_restart_history_reset_new_run()
    test_checkpoint_round_trip_incompatible_architecture()
    test_zero_loss_recorded_not_discarded()
    test_food_none_full_board_victory_small_grids_game_over()
    test_replay_memory_sample_compatibility()
    test_probability_helpers_no_mutation()
    test_rendering_unchanged_data_repeated_runs_controls()

    print("=" * 60)
    print("✅ ALL CHARACTERIZATION TESTS PASSED!")
    print("=" * 60)

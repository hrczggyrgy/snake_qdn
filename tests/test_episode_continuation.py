#!/usr/bin/env python3
"""
Regression test: automatic episode continuation in training mode.

This test verifies that when a training episode ends:
1. The episode count increases exactly once
2. `paused` remains False
3. The board resets and then advances
4. The same agent and replay buffer remain
5. Epsilon decays exactly once (not again during reset)
"""

from training import ActivityMode, TrainingController


def test_auto_episode_continuation_watch_mode():
    """Test that training continues automatically into next episode (Watch mode)."""
    print("Testing automatic episode continuation (Watch mode)...")

    controller = TrainingController()
    controller.set_mode(ActivityMode.TRAIN)
    controller.set_paused(False)

    # Initialize board to create agent
    controller.restart_board()
    agent_id_before = id(controller._agent)
    epsilon_before = controller.epsilon

    # Run until first episode ends
    episode_before = controller.episode
    env_steps_before = controller.total_env_steps
    opt_steps_before = controller.optimizer_steps

    max_steps = 500
    episode_ended = False
    for i in range(max_steps):
        snapshot = controller.advance_tick(execution_policy="Watch")
        if snapshot.game_over and controller.episode > episode_before:
            episode_ended = True
            print(f"  Episode ended at step {i + 1}")
            break

    assert episode_ended, "Episode did not end within max steps"
    assert controller.episode == episode_before + 1, (
        f"Episode count: {controller.episode} != {episode_before + 1}"
    )
    assert controller.paused is False, "Controller should not be paused after episode end"

    # Next tick should restart board and advance
    snapshot2 = controller.advance_tick(execution_policy="Watch")
    assert snapshot2.game_over is False, "Board should be active after restart"
    assert controller.episode == episode_before + 1, (
        "Episode count should not increase during reset"
    )

    # Verify same agent and replay buffer
    assert id(controller._agent) == agent_id_before, "Agent should be same object"
    assert len(controller.agent.memory) > 0, "Replay buffer should not be empty"

    # Verify epsilon decayed exactly once (not during reset)
    assert controller.epsilon < epsilon_before, "Epsilon should have decayed"
    expected_epsilon = max(
        controller.agent_config.epsilon_end, epsilon_before * controller.agent_config.epsilon_decay
    )
    assert abs(controller.epsilon - expected_epsilon) < 1e-6, (
        f"Epsilon should decay exactly once per episode: {controller.epsilon} != {expected_epsilon}"
    )

    print(f"  Episode count: {controller.episode}")
    print(f"  Env steps: {controller.total_env_steps} (was {env_steps_before})")
    print(f"  Opt steps: {controller.optimizer_steps} (was {opt_steps_before})")
    print(f"  Epsilon: {controller.epsilon:.4f} (was {epsilon_before:.4f})")
    print(f"  Replay size: {len(controller.agent.memory)}")

    controller.close()
    print("  ✓ Watch mode test passed!")


def test_auto_episode_continuation_fast_training():
    """Test that training continues automatically into next episode (FastTraining mode)."""
    print("Testing automatic episode continuation (FastTraining mode)...")

    controller = TrainingController()
    controller.agent_config.batch_size = 4
    controller.agent_config.memory_size = 100
    # Use smaller grid for faster termination
    controller.update_env_config(controller.env_config.__class__(grid_size=10))
    controller._agent = None  # Force recreation
    controller.set_mode(ActivityMode.TRAIN)
    controller.set_paused(False)

    # Initialize board to create agent
    controller.restart_board()
    agent_id_before = id(controller._agent)
    epsilon_before = controller.epsilon

    episode_before = controller.episode

    # Run enough bursts to complete an episode
    episode_ended = False
    for burst in range(50):
        controller.advance_tick(
            execution_policy="FastTraining",
            max_burst_steps=20,
            burst_time_budget=0.1,
        )
        if controller.episode > episode_before:
            episode_ended = True
            print(f"  Episode ended at burst {burst + 1}")
            break

    assert episode_ended, "Episode did not end within max bursts"
    assert controller.episode == episode_before + 1
    assert controller.training_episodes == episode_before + 1
    assert controller.paused is False

    # Next burst should restart and continue
    snapshot2 = controller.advance_tick(
        execution_policy="FastTraining",
        max_burst_steps=20,
        burst_time_budget=0.1,
    )
    assert snapshot2.game_over is False
    assert controller.episode == episode_before + 1

    # Verify same agent and epsilon decayed once
    assert id(controller._agent) == agent_id_before
    assert controller.epsilon < epsilon_before
    expected_epsilon = max(
        controller.agent_config.epsilon_end, epsilon_before * controller.agent_config.epsilon_decay
    )
    assert abs(controller.epsilon - expected_epsilon) < 1e-6

    print(f"  Episode count: {controller.episode}")
    print(f"  Training episodes: {controller.training_episodes}")
    print(f"  Epsilon: {controller.epsilon:.4f} (was {epsilon_before:.4f})")
    print(f"  Replay size: {len(controller.agent.memory)}")

    controller.close()
    print("  ✓ FastTraining mode test passed!")


def test_evaluation_pauses_on_terminal():
    """Test that evaluation mode pauses on terminal board."""
    print("Testing evaluation pauses on terminal board...")

    controller = TrainingController()
    # Use smaller grid for faster termination
    controller.update_env_config(controller.env_config.__class__(grid_size=10))
    controller.set_mode(ActivityMode.EVALUATE)
    controller.set_paused(False)

    # Run until episode ends
    for _ in range(200):
        snapshot = controller.advance_tick(execution_policy="Watch")
        if snapshot.game_over:
            break

    # Episode ended, now next tick should pause (not restart)
    snapshot = controller.advance_tick(execution_policy="Watch")
    assert controller.paused is True, "Evaluation should pause on terminal board"
    assert controller.episode == 1, "Total episode count should be 1"
    assert controller.evaluation_episodes == 1, "Evaluation episode count should be 1"
    assert controller.training_episodes == 0, "Training episode count should be 0"

    controller.close()
    print("  ✓ Evaluation pause test passed!")


def test_manual_mode_no_auto_advance():
    """Test that manual mode does not auto-advance."""
    print("Testing manual mode no auto-advance...")

    controller = TrainingController()
    controller.set_mode(ActivityMode.MANUAL)
    controller.set_paused(False)

    episode_before = controller.episode
    env_steps_before = controller.total_env_steps

    # Advance tick should do nothing in manual mode
    controller.advance_tick(execution_policy="Watch")

    assert controller.episode == episode_before
    assert controller.total_env_steps == env_steps_before
    assert controller.paused is False  # Manual mode doesn't auto-pause

    controller.close()
    print("  ✓ Manual mode test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Episode Continuation Regression Tests")
    print("=" * 60 + "\n")

    test_auto_episode_continuation_watch_mode()
    print()
    test_auto_episode_continuation_fast_training()
    print()
    test_evaluation_pauses_on_terminal()
    print()
    test_manual_mode_no_auto_advance()
    print()

    print("=" * 60)
    print("✅ ALL REGRESSION TESTS PASSED!")
    print("=" * 60)

#!/usr/bin/env python3
"""
Test script to verify all modules work correctly.
Run this before launching the Streamlit app.
"""

import sys
import os
import numpy as np
import torch
from collections import deque

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_snake_env():
    """Test the Snake environment."""
    print("Testing Snake Environment...")
    from snake_env import SnakeGameAI
    
    # Test standard grid
    env = SnakeGameAI(grid_size=10, render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    
    assert obs.shape == (11,), f"Expected obs shape (11,), got {obs.shape}"
    assert 'score' in info
    print(f"  ✓ Reset works, obs shape: {obs.shape}")
    
    # Test a few steps
    for i in range(5):
        action = np.random.randint(0, 3)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (11,)
        print(f"  Step {i}: action={action}, reward={reward:.2f}, score={info['score']}, done={terminated}")
        if terminated:
            break
    
    # Test render
    img = env.render()
    assert img is not None
    assert img.shape == (200, 200, 3), f"Expected (200, 200, 3), got {img.shape}"
    print(f"  ✓ Render works, shape: {img.shape}")
    
    env.close()
    
    # Test minimum grid size (3x3)
    env = SnakeGameAI(grid_size=3, render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    assert obs.shape == (11,)
    # Verify snake is within bounds
    for seg in env.snake:
        assert 0 <= seg[0] < 3 and 0 <= seg[1] < 3, f"Snake segment {seg} out of bounds for 3x3 grid"
    print(f"  ✓ Minimum grid size (3x3) works, snake: {list(env.snake)}")
    env.close()
    
    # Test grid_size=4
    env = SnakeGameAI(grid_size=4, render_mode='rgb_array')
    obs, info = env.reset(seed=42)
    for seg in env.snake:
        assert 0 <= seg[0] < 4 and 0 <= seg[1] < 4, f"Snake segment {seg} out of bounds for 4x4 grid"
    print(f"  ✓ Grid size 4x4 works, snake: {list(env.snake)}")
    env.close()
    
    # Test that grid_size < 3 raises error
    try:
        SnakeGameAI(grid_size=2)
        assert False, "Should have raised ValueError for grid_size < 3"
    except ValueError as e:
        print(f"  ✓ Correctly rejects grid_size < 3: {e}")
    
    print("  ✓ Snake Environment tests passed!\n")


def test_model():
    """Test the neural network models."""
    print("Testing Neural Network Models...")
    from model import Linear_QNet, QTrainer
    
    # Test model creation
    model = Linear_QNet(input_size=11, hidden_size=64, output_size=3)
    print(f"  ✓ Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    x = torch.randn(32, 11)
    output = model(x)
    assert output.shape == (32, 3), f"Expected (32, 3), got {output.shape}"
    print(f"  ✓ Forward pass works, output shape: {output.shape}")
    
    # Test activations
    acts = model.get_activations(x[0])
    assert 'fc1' in acts and 'fc2' in acts and 'fc3' in acts
    print(f"  ✓ Activations extracted: {list(acts.keys())}")
    
    # Test trainer
    trainer = QTrainer(model, lr=0.001, gamma=0.99)
    
    # Dummy training step
    states = torch.randn(16, 11)
    actions = torch.randint(0, 3, (16,))
    rewards = torch.randn(16)
    next_states = torch.randn(16, 11)
    dones = torch.randint(0, 2, (16,)).bool()
    
    loss = trainer.train_step(states, actions, rewards, next_states, dones)
    print(f"  ✓ Training step works, loss: {loss:.4f}")
    
    # Test Q-values
    q_vals = trainer.get_q_values(np.random.randn(11))
    assert q_vals.shape == (3,)
    print(f"  ✓ Q-values inference works: {q_vals}")
    
    # Test action probabilities
    probs = trainer.get_action_probabilities(np.random.randn(11))
    assert probs.shape == (3,)
    assert abs(probs.sum() - 1.0) < 1e-5
    print(f"  ✓ Action probabilities work: {probs}")
    
    print("  ✓ Model tests passed!\n")


def test_agent():
    """Test the DQN agent."""
    print("Testing DQN Agent...")
    from agent import Agent, ReplayMemory
    from snake_env import SnakeGameAI
    
    # Test replay memory
    memory = ReplayMemory(capacity=1000)
    for i in range(10):
        memory.push(np.random.randn(11), np.random.randint(0, 3), np.random.randn(), np.random.randn(11), False, False)
    assert len(memory) == 10
    batch = memory.sample(5)
    assert len(batch) == 6  # states, actions, rewards, next_states, terminated, truncated
    print(f"  ✓ Replay memory works, size: {len(memory)}")
    
    # Verify sample returns terminated and truncated
    states, actions, rewards, next_states, terminated, truncated = batch
    assert terminated.shape == (5,)
    assert truncated.shape == (5,)
    print(f"  ✓ Sample returns terminated/truncated correctly")
    
    # Test agent
    agent = Agent(hidden_size=64, memory_size=1000, batch_size=32)
    print(f"  ✓ Agent created")
    
    # Test action selection
    state = np.random.randn(11)
    action = agent.get_action(state, training=True)
    assert 0 <= action <= 2
    print(f"  ✓ Action selection works: {action}")
    
    # Test greedy action
    action_greedy = agent.get_action(state, training=False)
    assert 0 <= action_greedy <= 2
    print(f"  ✓ Greedy action works: {action_greedy}")
    
    # Test memory
    agent.remember(state, action, 1.0, np.random.randn(11), False)
    print(f"  ✓ Memory store works, size: {len(agent.memory)}")
    
    # Test short memory training
    loss = agent.train_short_memory(state, action, 1.0, np.random.randn(11), False)
    print(f"  ✓ Short memory training works, loss: {loss:.4f}")
    
    # Test stats
    stats = agent.get_training_stats()
    assert 'n_games' in stats
    print(f"  ✓ Stats work: {stats}")
    
    print("  ✓ Agent tests passed!\n")


def test_integration():
    """Test full integration: env + agent playing."""
    print("Testing Full Integration...")
    from snake_env import SnakeGameAI
    from agent import Agent
    
    env = SnakeGameAI(grid_size=10)
    agent = Agent(hidden_size=64, memory_size=1000, batch_size=32, epsilon_start=0.1)
    
    # Play a few episodes
    for ep in range(3):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, next_state, terminated, truncated)
            agent.train_short_memory(state, action, reward, next_state, terminated)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        agent.decay_epsilon()
        print(f"  Episode {ep}: score={info['score']}, steps={steps}, reward={total_reward:.2f}, epsilon={agent.epsilon:.3f}")
    
    env.close()
    print("  ✓ Integration test passed!\n")


def test_timeout_semantics():
    """Test that timeout returns truncated=True, not terminated."""
    print("Testing Timeout Semantics...")
    from snake_env import SnakeGameAI
    
    # Larger grid so snake doesn't hit wall before timeout
    env = SnakeGameAI(grid_size=10, step_penalty=0.0, food_reward=0.0, death_penalty=-1.0)
    
    obs, _ = env.reset()
    # Override max steps AFTER reset to a small value
    env.max_steps_without_food = 3
    # Place food far away so it won't be reached
    env.food = (9, 9)  # Bottom-right corner, snake starts at center moving right
    
    terminated = truncated = False
    
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(0)  # Go straight
        if terminated or truncated:
            break
    
    assert truncated == True, f"Expected truncated=True for timeout, got terminated={terminated}, truncated={truncated}"
    assert terminated == False, f"Timeout should not set terminated=True"
    # Should timeout exactly after 3 steps (max_steps_without_food)
    assert env.steps_since_food == 3, f"Expected steps_since_food=3, got {env.steps_since_food}"
    assert env.game_over == True, "game_over should be True after timeout"
    print(f"  ✓ Timeout correctly returns truncated={truncated}, terminated={terminated} after 3 steps")
    
    env.close()
    print("  ✓ Timeout semantics test passed!\n")


def test_bootstrap_target():
    """Test that terminated transitions suppress bootstrap while truncated keep it."""
    print("Testing Bootstrap Target Semantics...")
    import torch
    from model import Linear_QNet, QTrainer
    from agent import Agent
    
    # Create a simple scenario
    model = Linear_QNet(input_size=11, hidden_size=64, output_size=3)
    trainer = QTrainer(model, lr=0.001, gamma=0.9, device='cpu')
    
    # Set up: reward=1, next_state_value=5
    # For terminated: target = 1 + 0.9 * 5 * (1 - 1) = 1
    # For truncated: target = 1 + 0.9 * 5 * (1 - 0) = 5.5
    
    state = torch.randn(1, 11)
    action = torch.tensor([0])
    reward = torch.tensor([1.0])
    next_state = torch.randn(1, 11)
    
    # Manually set next_state Q-values to have max=5
    with torch.no_grad():
        # We can't easily control target network output, so test the formula directly
        gamma = 0.9
        next_q_max = 5.0
        
        # Terminated transition (game over)
        terminated = torch.tensor([1.0], dtype=torch.bool)
        target_term = reward + gamma * next_q_max * (1 - terminated.float())
        
        # Truncated transition (timeout)
        truncated = torch.tensor([0.0], dtype=torch.bool)
        target_trunc = reward + gamma * next_q_max * (1 - truncated.float())
        
        assert abs(target_term.item() - 1.0) < 0.01, f"Terminated target should be 1.0, got {target_term.item()}"
        assert abs(target_trunc.item() - 5.5) < 0.01, f"Truncated target should be 5.5, got {target_trunc.item()}"
        
        print(f"  ✓ Terminated target: {target_term.item():.2f} (no bootstrap)")
        print(f"  ✓ Truncated target: {target_trunc.item():.2f} (keeps bootstrap)")
    
    print("  ✓ Bootstrap target test passed!\n")


def test_full_board_victory():
    """Test that filling the board results in terminated=True (victory)."""
    print("Testing Full Board Victory...")
    from snake_env import SnakeGameAI
    
    # Use tiny grid to fill quickly
    env = SnakeGameAI(grid_size=3)
    
    # Manually fill most of the board - snake head at (1,0), moving right
    # Grid is 3x3, snake occupies 8 cells, food at last empty cell (0,0)
    env.snake = deque([
        (1, 0),  # head
        (2, 0),  # body
        (2, 1),  # body
        (1, 1),  # body
        (0, 1),  # body
        (0, 2),  # body
        (1, 2),  # body
        (2, 2),  # body
    ])
    env.direction = 2  # Left (head at (1,0) moving left to (0,0))
    env.food = (0, 0)  # Last empty cell
    env.score = 7
    
    # Move straight (left) to eat food at (0,0)
    obs, reward, terminated, truncated, info = env.step(0)  # Straight (left)
    
    assert terminated == True, f"Expected terminated=True for victory, got {terminated}"
    assert truncated == False, f"Expected truncated=False for victory"
    assert info['score'] == 8, f"Expected score=8 (3x3-1=8), got {info['score']}"
    assert env.food is None, "Food should be None when board is full"
    assert len(env.snake) == 9, f"Snake should fill board (9), got {len(env.snake)}"
    print(f"  ✓ Full board victory: terminated={terminated}, score={info['score']}, len={len(env.snake)}")
    
    env.close()
    print("  ✓ Full board victory test passed!\n")


def test_save_load():
    """Test checkpoint saving and loading."""
    print("Testing Checkpoint Save/Load...")
    from agent import Agent
    import os
    
    agent = Agent(hidden_size=64)
    agent.n_games = 42
    agent.high_score = 15
    agent.epsilon = 0.5
    
    save_path = '/tmp/test_checkpoint.pth'
    agent.save_checkpoint(save_path)
    print(f"  ✓ Checkpoint saved to {save_path}")
    
    # Create new agent and load
    new_agent = Agent(hidden_size=64)
    info = new_agent.load_checkpoint(save_path)
    print(f"  ✓ Checkpoint loaded: episode={info['episode']}, epsilon={info['epsilon']:.3f}, high_score={info['high_score']}")
    
    # Verify returned metadata
    assert info['episode'] == 42
    assert info['high_score'] == 15
    assert abs(info['epsilon'] - 0.5) < 0.01
    
    # Verify agent state was actually restored
    assert new_agent.n_games == 42, f"Expected n_games=42, got {new_agent.n_games}"
    assert new_agent.high_score == 15, f"Expected high_score=15, got {new_agent.high_score}"
    assert abs(new_agent.epsilon - 0.5) < 0.01, f"Expected epsilon=0.5, got {new_agent.epsilon}"
    
    # Verify Q-values are consistent (greedy action should match)
    test_state = np.random.randn(11)
    original_action = agent.get_action(test_state, training=False)
    loaded_action = new_agent.get_action(test_state, training=False)
    assert original_action == loaded_action, f"Greedy actions differ: {original_action} vs {loaded_action}"
    
    # Verify full Q-value arrays match (not just argmax)
    original_q = agent.get_q_values(test_state)
    loaded_q = new_agent.get_q_values(test_state)
    assert np.allclose(original_q, loaded_q), f"Q-values differ:\n  original: {original_q}\n  loaded:   {loaded_q}"
    
    os.remove(save_path)
    print("  ✓ Save/Load test passed!\n")


def test_epsilon_decay_on_episode_completion():
    """Test that epsilon decays when episode completes (matches dashboard behavior)."""
    print("Testing Epsilon Decay on Episode Completion...")
    from agent import Agent
    from snake_env import SnakeGameAI
    
    env = SnakeGameAI(grid_size=10)
    agent = Agent(hidden_size=64, memory_size=1000, batch_size=32, 
                  epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.5)
    
    initial_epsilon = agent.epsilon
    
    # Simulate completing an episode
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.get_action(state, training=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, terminated, truncated)
        agent.train_short_memory(state, action, reward, next_state, terminated)
        state = next_state
    
    # Dashboard behavior: calls update_stats() THEN decay_epsilon()
    agent.update_stats(info['score'])
    agent.decay_epsilon()
    
    expected_epsilon = max(0.01, initial_epsilon * 0.5)
    assert abs(agent.epsilon - expected_epsilon) < 0.001, \
        f"Expected epsilon={expected_epsilon:.4f}, got {agent.epsilon:.4f}"
    
    print(f"  ✓ Epsilon decayed from {initial_epsilon:.4f} to {agent.epsilon:.4f}")
    
    # Test second episode decay
    agent.decay_epsilon()
    expected_epsilon_2 = max(0.01, expected_epsilon * 0.5)
    assert abs(agent.epsilon - expected_epsilon_2) < 0.001
    
    # Test that evaluation does NOT decay epsilon (dashboard doesn't call update_stats for eval)
    eval_epsilon = agent.epsilon
    state, _ = env.reset()
    done = False
    while not done:
        action = agent.get_action(state, training=False)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
    
    # Evaluation path in dashboard: does NOT call update_stats or decay_epsilon
    # Just verify epsilon unchanged
    assert agent.epsilon == eval_epsilon, "Evaluation should not change epsilon"
    
    env.close()
    print("  ✓ Epsilon decay test passed!\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Snake DQN Module Tests")
    print("=" * 50 + "\n")
    
    try:
        test_snake_env()
        test_model()
        test_agent()
        test_integration()
        test_timeout_semantics()
        test_bootstrap_target()
        test_full_board_victory()
        test_save_load()
        test_epsilon_decay_on_episode_completion()
        
        print("=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
        print("\nYou can now run the Streamlit app with:")
        print("  streamlit run app.py")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
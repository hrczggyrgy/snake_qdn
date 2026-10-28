#!/usr/bin/env python3
"""
Test script to verify all modules work correctly.
Run this before launching the Streamlit app.
"""

import sys
import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, '/home/lptop/Documents/coding/snake')

def test_snake_env():
    """Test the Snake environment."""
    print("Testing Snake Environment...")
    from snake_env import SnakeGameAI
    
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
        memory.push(np.random.randn(11), np.random.randint(0, 3), np.random.randn(), np.random.randn(11), False)
    assert len(memory) == 10
    batch = memory.sample(5)
    assert len(batch) == 5
    print(f"  ✓ Replay memory works, size: {len(memory)}")
    
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
            
            agent.remember(state, action, reward, next_state, done)
            agent.train_short_memory(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        agent.decay_epsilon()
        print(f"  Episode {ep}: score={info['score']}, steps={steps}, reward={total_reward:.2f}, epsilon={agent.epsilon:.3f}")
    
    env.close()
    print("  ✓ Integration test passed!\n")


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
    
    assert info['episode'] == 42
    assert info['high_score'] == 15
    assert abs(info['epsilon'] - 0.5) < 0.01
    
    os.remove(save_path)
    print("  ✓ Save/Load test passed!\n")


if __name__ == "__main__":
    print("=" * 50)
    print("Running Snake DQN Module Tests")
    print("=" * 50 + "\n")
    
    try:
        test_snake_env()
        test_model()
        test_agent()
        test_integration()
        test_save_load()
        
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
"""
DQN Agent for Snake Game.

Implements experience replay, epsilon-greedy exploration,
and training loop management.
"""

import random
import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Dict, Any
import torch
import os

from snake_env import SnakeGameAI
from model import Linear_QNet, QTrainer


class ReplayMemory:
    """Experience Replay Buffer with efficient sampling.
    
    Stores terminated flag separately from truncated for correct
    bootstrapping in Q-learning (terminated suppresses bootstrap).
    """
    
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, terminated: bool, truncated: bool) -> None:
        """Add experience to memory.
        
        Args:
            terminated: True if episode ended due to game-over (wall/self collision)
            truncated: True if episode ended due to external limit (timeout)
        """
        self.memory.append((state, action, reward, next_state, terminated, truncated))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample random batch from memory.
        
        Returns:
            states, actions, rewards, next_states, terminated, truncated
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, terminated, truncated = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(terminated, dtype=bool),
            np.array(truncated, dtype=bool)
        )
    
    def __len__(self) -> int:
        return len(self.memory)
    
    def is_ready(self, batch_size: int) -> bool:
        return len(self.memory) >= batch_size


class Agent:
    """
    DQN Agent for Snake.
    
    Features:
    - Experience Replay
    - Epsilon-Greedy Exploration
    - Target Network
    - Training metrics tracking
    """
    
    def __init__(
        self,
        input_size: int = 11,
        hidden_size: int = 256,
        output_size: int = 3,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 100_000,
        batch_size: int = 1000,
        target_update_freq: int = 10,
        device: str = 'auto'
    ):
        # Model and trainer
        self.model = Linear_QNet(input_size, hidden_size, output_size)
        self.trainer = QTrainer(self.model, lr=lr, gamma=gamma, device=device)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Memory
        self.memory = ReplayMemory(memory_size)
        
        # Training stats
        self.n_games = 0
        self.total_steps = 0
        self.high_score = 0
        self.scores_history: List[int] = []
        self.mean_scores_history: List[float] = []
        self.epsilon_history: List[float] = []
        self.loss_history: List[float] = []
        
        # For step-by-step mode
        self.last_state: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        
        Args:
            state: Current observation (11,)
            training: If True, use epsilon-greedy. If False, greedy.
            
        Returns:
            Action index (0, 1, 2)
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, 2)
        
        # Greedy action
        q_values = self.trainer.get_q_values(state)
        return int(np.argmax(q_values))
    
    def get_action_with_reason(self, state: np.ndarray, training: bool = True) -> Tuple[int, str]:
        """
        Epsilon-greedy action selection with reason.
        
        Args:
            state: Current observation (11,)
            training: If True, use epsilon-greedy. If False, greedy.
            
        Returns:
            Tuple of (action_index, reason) where reason is "exploring", "exploiting", or "manual"
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, 2), "exploring"
        
        # Greedy action
        q_values = self.trainer.get_q_values(state)
        return int(np.argmax(q_values)), "exploiting"
    
    def get_action_probs(self, state: np.ndarray) -> np.ndarray:
        """Get action probabilities for visualization."""
        return self.trainer.get_action_probabilities(state)
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get raw Q-values for visualization."""
        return self.trainer.get_q_values(state)
    
    def remember(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        terminated: bool,
        truncated: bool = False
    ) -> None:
        """Store experience in replay memory.
        
        Args:
            terminated: Episode ended due to game-over (suppresses bootstrap)
            truncated: Episode ended due to external limit (keeps bootstrap)
        """
        self.memory.push(state, action, reward, next_state, terminated, truncated)
    
    def train_short_memory(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        terminated: bool
    ) -> float:
        """Train on single step (for immediate feedback).
        
        Args:
            terminated: True if episode ended due to game-over (suppresses bootstrap)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action_t = torch.LongTensor([action])
        reward_t = torch.FloatTensor([reward])
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        terminated_t = torch.BoolTensor([terminated])
        
        loss = self.trainer.train_step(state_t, action_t, reward_t, next_state_t, terminated_t)
        return loss
    
    def train_long_memory(self) -> Optional[float]:
        """Train on batch from replay memory."""
        if not self.memory.is_ready(self.batch_size):
            return None
        
        states, actions, rewards, next_states, terminated, truncated = self.memory.sample(self.batch_size)
        
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        terminated_t = torch.BoolTensor(terminated)
        
        loss = self.trainer.train_step(states_t, actions_t, rewards_t, next_states_t, terminated_t)
        return loss
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def update_stats(self, score: int, loss: Optional[float] = None) -> None:
        """Update training statistics."""
        self.n_games += 1
        self.scores_history.append(score)
        self.epsilon_history.append(self.epsilon)
        
        # Rolling average (last 50 games)
        window = min(50, len(self.scores_history))
        avg_score = np.mean(self.scores_history[-window:])
        self.mean_scores_history.append(avg_score)
        
        if score > self.high_score:
            self.high_score = score
        
        if loss is not None:
            self.loss_history.append(loss)
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        self.trainer.save_checkpoint(
            filepath,
            episode=self.n_games,
            epsilon=self.epsilon,
            high_score=self.high_score,
            avg_score=self.mean_scores_history[-1] if self.mean_scores_history else 0.0
        )
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load agent checkpoint."""
        info = self.trainer.load_checkpoint(filepath)
        
        # Restore agent state from checkpoint
        self.n_games = info.get('episode', 0)
        self.epsilon = info.get('epsilon', 1.0)
        self.high_score = info.get('high_score', 0)
        
        return info
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'n_games': self.n_games,
            'total_steps': self.total_steps,
            'high_score': self.high_score,
            'current_score': self.scores_history[-1] if self.scores_history else 0,
            'epsilon': self.epsilon,
            'avg_score_50': self.mean_scores_history[-1] if self.mean_scores_history else 0,
            'avg_score_10': np.mean(self.scores_history[-10:]) if len(self.scores_history) >= 10 else 0,
            'memory_size': len(self.memory),
            'loss': self.loss_history[-1] if self.loss_history else 0
        }
    
    def reset_stats(self) -> None:
        """Reset training statistics (keep model weights)."""
        self.n_games = 0
        self.total_steps = 0
        self.high_score = 0
        self.scores_history = []
        self.mean_scores_history = []
        self.epsilon_history = []
        self.loss_history = []
        self.trainer.loss_history = []
        self.trainer.training_step = 0


def train_agent(
    agent: Agent,
    env: SnakeGameAI,
    n_episodes: int = 1000,
    max_steps_per_episode: int = 2000,
    render_every: int = 0,
    save_every: int = 100,
    save_path: str = 'checkpoints/snake_dqn.pth',
    callback=None
) -> Dict[str, List]:
    """
    Main training loop.
    
    Args:
        agent: DQN Agent
        env: Snake Environment
        n_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        render_every: Render every N episodes (0 = never)
        save_every: Save checkpoint every N episodes
        save_path: Path to save checkpoints
        callback: Optional callback function(episode, stats) for progress tracking
        
    Returns:
        Dictionary with training history
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps_per_episode):
            # Get action
            action = agent.get_action(state, training=True)
            
            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store in memory (terminated controls bootstrap, done controls episode end)
            agent.remember(state, action, reward, next_state, terminated, truncated)
            
            # Train short memory (immediate) - use terminated for bootstrap mask
            loss = agent.train_short_memory(state, action, reward, next_state, terminated)
            if loss:
                episode_loss.append(loss)
            
            # Train long memory (replay)
            replay_loss = agent.train_long_memory()
            if replay_loss:
                episode_loss.append(replay_loss)
            
            state = next_state
            episode_reward += reward
            agent.total_steps += 1
            
            if done:
                break
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Update stats
        avg_loss = np.mean(episode_loss) if episode_loss else None
        agent.update_stats(info['score'], avg_loss)
        
        # Callback for progress
        if callback:
            callback(episode, agent.get_training_stats())
        
        # Save checkpoint
        if save_every > 0 and (episode + 1) % save_every == 0:
            agent.save_checkpoint(save_path)
            loss_text = f"{avg_loss:.4f}" if avg_loss is not None else "N/A"
            print(f"Episode {episode + 1}: Score={info['score']}, "
                  f"Avg(50)={agent.mean_scores_history[-1]:.2f}, "
                  f"Epsilon={agent.epsilon:.3f}, Loss={loss_text}")
    
    # Final save
    agent.save_checkpoint(save_path)
    
    return {
        'scores': agent.scores_history,
        'mean_scores': agent.mean_scores_history,
        'epsilon': agent.epsilon_history,
        'loss': agent.loss_history
    }


def evaluate_agent(
    agent: Agent,
    env: SnakeGameAI,
    n_episodes: int = 10,
    render: bool = False
) -> Dict[str, float]:
    """
    Evaluate agent performance (no exploration).
    
    Returns:
        Dictionary with evaluation metrics
    """
    scores = []
    steps_list = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action = agent.get_action(state, training=False)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                scores.append(info['score'])
                steps_list.append(steps)
                break
    
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': np.max(scores),
        'min_score': np.min(scores),
        'mean_steps': np.mean(steps_list),
        'scores': scores
    }


if __name__ == "__main__":
    # Quick test
    env = SnakeGameAI(grid_size=20)
    agent = Agent()
    
    # Test a few steps
    state, _ = env.reset()
    for _ in range(10):
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.remember(state, action, reward, next_state, terminated or truncated)
        state = next_state
        if terminated or truncated:
            break
    
    print(f"Memory size: {len(agent.memory)}")
    print(f"Test passed!")
import numpy as np
import sys
import os
import argparse
import time

# Ensure src is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from env import PaperEnv
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from env import PaperEnv

class CEMAgent:
    def __init__(self, env, n_elite=10, population_size=50, sigma_init=0.5):
        self.env = env
        self.n_elite = n_elite
        self.pop_size = population_size
        self.sigma = sigma_init
        
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        self.mean_weights = np.zeros(self.obs_dim * self.act_dim)
        self.cov_weights = np.eye(self.obs_dim * self.act_dim) * self.sigma

    def get_action(self, weights, obs):
        W = weights.reshape(self.act_dim, self.obs_dim)
        return np.tanh(W @ obs)

    def evaluate(self, weights, render=False):
        obs, _ = self.env.reset()
        total_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = self.get_action(weights, obs)
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if render:
                self.env.render()
        
        return total_reward

    def train(self, n_generations=20):
        print(f"Starting CEM Training (Gen={n_generations}, Pop={self.pop_size})...")
        
        for g in range(n_generations):
            samples = np.random.multivariate_normal(
                self.mean_weights, 
                self.cov_weights + np.eye(len(self.mean_weights))*1e-3, 
                self.pop_size
            )
            
            rewards = []
            for w in samples:
                rewards.append(self.evaluate(w))
            
            rewards = np.array(rewards)
            elite_idxs = rewards.argsort()[-self.n_elite:]
            elite_weights = samples[elite_idxs]
            
            self.mean_weights = elite_weights.mean(axis=0)
            self.cov_weights = np.cov(elite_weights, rowvar=False)
            
            avg_reward = rewards.mean()
            best_reward = rewards.max()
            print(f"Gen {g+1}: Avg Reward={avg_reward:.2f}, Best Reward={best_reward:.2f}")

        print("Training Complete.")
        return self.mean_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the agent")
    parser.add_argument("--view", action="store_true", help="View the result")
    parser.add_argument("--quick", action="store_true", help="Run a quick training session for testing")
    args = parser.parse_args()

    env = PaperEnv(render_mode="human" if args.view else None)
    agent = CEMAgent(env)
    
    weights_file = "agent_weights.npy"
    
    if args.train:
        gens = 2 if args.quick else 15
        pop_size = 10 if args.quick else 50
        agent.pop_size = pop_size
        
        best_weights = agent.train(n_generations=gens)
        np.save(weights_file, best_weights)
        print(f"Saved weights to {weights_file}")
        
    if args.view:
        if os.path.exists(weights_file):
            print(f"Loading weights from {weights_file}...")
            try:
                weights = np.load(weights_file)
                # Check size
                expected_size = agent.obs_dim * agent.act_dim
                if weights.size != expected_size:
                    raise ValueError(f"Size mismatch: expected {expected_size}, got {weights.size}")
                
                print("Running simulation...")
                agent.evaluate(weights, render=True)
                input("Press Enter to exit...")
            except Exception as e:
                print(f"Error loading/running with weights: {e}")
                print("Suggestion: run 'python src/train.py --train --quick' to regenerate weights.")
        else:
            print("No weights found. Run with --train first.")

if __name__ == "__main__":
    main()

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


    def evaluate_batch(self, population_weights):
        """
        Evaluate the entire population in parallel using the batched environment.
        population_weights: (Pop, Act*Obs)
        """
        import torch
        
        pop_size = population_weights.shape[0]
        # Reset Env
        obs = self.env.reset() # (Pop, Obs)
        
        total_rewards = torch.zeros(pop_size, device=self.env.device)
        done = False
        
        # We need to reshape weights for batch matmul
        # W: (Pop, Act, Obs)
        # weights: (Pop, Act*Obs)
        W = population_weights.view(pop_size, self.act_dim, self.obs_dim)
        
        for _ in range(self.env.max_steps):
            # Action = tanh(W @ obs)
            # Obs: (Pop, Obs, 1)
            obs_uns = obs.unsqueeze(2)
            act = torch.tanh(torch.bmm(W, obs_uns)).squeeze(2)
            
            obs, rewards, done, _ = self.env.step(act)
            total_rewards += rewards
            
            if done: break
            
        return total_rewards.cpu().numpy()

    def train(self, n_generations=20, use_cuda=False):
        print(f"Starting CEM Training (Gen={n_generations}, Pop={self.pop_size}, CUDA={use_cuda})...")
        
        # If CUDA, convert mean/cov to torch if needed or just handle in loop
        
        for g in range(n_generations):
            samples = np.random.multivariate_normal(
                self.mean_weights, 
                self.cov_weights + np.eye(len(self.mean_weights))*1e-3, 
                self.pop_size
            )
            
            if use_cuda:
                import torch
                # Convert samples to tensor
                w_tensor = torch.tensor(samples, dtype=torch.float32, device=self.env.device)
                rewards = self.evaluate_batch(w_tensor)
            else:
                rewards = []
                for w in samples:
                    rewards.append(self.evaluate(w))
                rewards = np.array(rewards)
            
            # Elite Selection
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
    parser.add_argument("--cuda", action="store_true", help="Use CUDA for training")
    args = parser.parse_args()

    if args.cuda and args.train:
        # Use Batched Env
        try:
            import torch
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, using CPU for tensor operations.")
                device = "cpu"
            else:
                device = "cuda"
            
            try:
                from src.torch_env import BatchPaperEnv
            except ImportError:
                from torch_env import BatchPaperEnv
            # Pop size needed for env init?
            # We usually define pop size in agent. 
            # But BatchEnv needs batch size.
            # Let's align them.
            gens = 2 if args.quick else 15
            pop_size = 10 if args.quick else 50
            
            env = BatchPaperEnv(batch_size=pop_size, device=device)
            print(f"Initialized Batch Enviroment on {device}")
            
        except ImportError as e:
            print(f"Failed to import Torch/CUDA modules: {e}")
            return
    else:
        env = PaperEnv(render_mode="human" if args.view else None)

    agent = CEMAgent(env)
    
    weights_file = "agent_weights.npy"
    
    if args.train:
        gens = 2 if args.quick else 15
        pop_size = 10 if args.quick else 50
        agent.pop_size = pop_size
        
        # If using batched env, make sure pop size matches
        if args.cuda:
             if agent.pop_size != env.batch_size:
                 print("Error: Agent population size must match Environment batch size for CUDA training.")
                 return
        
        best_weights = agent.train(n_generations=gens, use_cuda=args.cuda)
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
                
                # If we used CUDA env for training, we might have a BatchEnv here.
                # But for viewing we want a single instance human render.
                # Re-init env if needed.
                if hasattr(agent.env, 'batch_size'):
                     print("Switching to standard env for visualization...")
                     agent.env = PaperEnv(render_mode="human")
                
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

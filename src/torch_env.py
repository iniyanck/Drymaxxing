import torch
import numpy as np
try:
    from src.torch_sim import TorchPaperSim
except ImportError:
    from torch_sim import TorchPaperSim

class BatchPaperEnv:
    def __init__(self, batch_size=1, device="cpu", L=10.0, W=5.0, n_controls=5):
        self.batch_size = batch_size
        self.device = device
        self.L = L
        self.W = W
        self.n_controls = n_controls
        
        self.sim = TorchPaperSim(L=L, W=W, batch_size=batch_size, device=device)
        
        # Action/Observation Dimensions
        self.action_dim = 6 + n_controls
        self.obs_dim = 9 + n_controls # pos(3), euler(3), k(n), rain(3)
        
        # Compatibility with Gym/CEMAgent
        class Space:
             def __init__(self, shape): self.shape = shape
             
        self.action_space = Space((self.action_dim,))
        self.observation_space = Space((self.obs_dim,))
        
        # State
        self.pos = torch.zeros((batch_size, 3), device=device)
        self.euler = torch.zeros((batch_size, 3), device=device)
        self.curvatures = torch.zeros((batch_size, n_controls), device=device)
        self.rain_dir = torch.tensor([0.0, 0.0, -1.0], device=device).expand(batch_size, 3)
        
        self.max_linear_speed = 0.5 
        self.max_angular_speed = 0.1 
        self.max_k_speed = 0.1 
        
        self.workspace_bounds = 12.0
        
        self.current_step = 0
        self.max_steps = 200
        
    def reset(self):
        self.current_step = 0
        
        # Random Initialization
        # Pos near center
        self.pos = (torch.rand((self.batch_size, 3), device=self.device) * 4.0 - 2.0)
        self.euler = (torch.rand((self.batch_size, 3), device=self.device) * 0.4 - 0.2)
        self.curvatures = (torch.rand((self.batch_size, self.n_controls), device=self.device) * 0.02 - 0.01)
        
        # Random Rain Direction per batch or same?
        # Typically environment has one condition, but for robustness we can randomize per agent or same for all.
        # Let's randomize per agent to learn generic policy.
        
        v = torch.randn((self.batch_size, 3), device=self.device)
        v = v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)
        
        # Flip Z if pointing up
        mask_up = v[:, 2] > 0
        v[mask_up, 2] = -v[mask_up, 2]
        
        # Bias down
        v[:, 2] -= 0.5
        self.rain_dir = v / (torch.norm(v, dim=1, keepdim=True) + 1e-8)
        
        self.sim.reset_wetness()
        self.sim.update(self.pos, self.euler, self.curvatures)
        
        return self._get_obs()

    def _get_obs(self):
        # [pos, euler, curv, rain]
        return torch.cat([self.pos, self.euler, self.curvatures, self.rain_dir], dim=1)

    def step(self, actions):
        # actions: (B, action_dim)
        actions = torch.clamp(actions, -1.0, 1.0)
        
        d_pos = actions[:, 0:3] * self.max_linear_speed
        d_euler = actions[:, 3:6] * self.max_angular_speed
        d_k = actions[:, 6:] * self.max_k_speed
        
        self.pos += d_pos
        self.euler += d_euler
        
        # Wrap phases? Or keep continuous? 
        # Original env wraps: (val + pi) % 2pi - pi
        pi = 3.14159265359
        self.euler = (self.euler + pi) % (2 * pi) - pi
        
        # Boundary Check
        self.pos = torch.clamp(self.pos, -self.workspace_bounds, self.workspace_bounds)
        
        # Floor Constraint (Approximate: check Center Z)
        # Detailed check in sim update? Sim update builds vertices.
        # Let's just update and let penalties handle it, or hard clamp.
        # Hard clamp floor for center
        floor_limit = -8.0
        self.pos[:, 2] = torch.clamp(self.pos[:, 2], min=floor_limit)
        
        self.curvatures += d_k
        self.curvatures = torch.clamp(self.curvatures, -2.0, 2.0)
        
        self.sim.update(self.pos, self.euler, self.curvatures)
        
        # Apply Rain
        self.sim.apply_rain(rain_dir=self.rain_dir, n_drops=100)
        
        # Calculate Reward
        wet_penalty = self.sim.calculate_wet_area(self.rain_dir) / (self.L * self.W) # (B,)
        
        # Clip Penalty
        # Max extent
        # We can check vertices if we want precise
        # Or just use pos
        clip_penalty = torch.zeros(self.batch_size, device=self.device)
        
        # Simple box check on center for speed, maybe check vertices occasionally?
        # Vertices are (B, Np, Nr, 3)
        # Max abs coord
        
        # This can be heavy. Let's trust center pos + approx size
        # Paper is ~10x5. Radius ~6
        # If pos > workspace - 6, start penalizing
        
        dist = torch.max(torch.abs(self.pos), dim=1)[0]
        clip_mask = dist > (self.workspace_bounds - 2.0)
        clip_penalty[clip_mask] = (dist[clip_mask] - (self.workspace_bounds - 2.0)) * 1.0
        
        reward = 1.0 - wet_penalty - clip_penalty
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, done, {}

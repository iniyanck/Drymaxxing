import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from src.paper import PaperSim
except ImportError:
    try:
        from paper import PaperSim
    except ImportError:
        # Fallback
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from paper import PaperSim

class PaperEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, L=10.0, W=5.0, n_controls=5, rain_mode='dynamic', render_rain_effects=True):
        self.L = L
        self.W = W
        self.n_controls = n_controls
        self.render_mode = render_mode
        self.rain_mode = rain_mode
        self.render_rain_effects = render_rain_effects
        self.sim = PaperSim(L=L, W=W, n_profile=60, n_rulings=50)
        
        # Action Space: 
        # [v_x, v_y, v_z, v_roll, v_pitch, v_yaw, v_curl, d_k1...d_kn]
        # Size = 7 + n_controls
        self.action_dim = 7 + n_controls
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(self.action_dim,), 
            dtype=np.float32
        )
        
        # Observation Space:
        # [x, y, z, roll, pitch, yaw, curl_angle, k1...kn, rx, ry, rz, lx, ly, lz]
        # Added 3 for global rain direction + 3 for local rain direction
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(13 + n_controls,),
            dtype=np.float32
        )
        
        # State
        self.pos = np.zeros(3)
        self.euler = np.zeros(3)
        self.curl_angle = 0.0
        self.curvatures = np.zeros(n_controls)
        self.rain_dir = np.array([0.0, 0.0, -1.0])
        
        self.max_linear_speed = 0.5 
        self.max_angular_speed = 0.25
        self.max_curl_speed = 0.1 # Slow rotation for bending axis
        self.max_k_speed = 0.2 
        
        self.max_steps = 200 # Increased for RL
        self.current_step = 0
        self.workspace_bounds = 12.0 # Slightly tighter to keep it visible
        
        # Camera State
        self.cam_pos = np.array([0.0, -15.0, 10.0])
        self.cam_elev = 30
        self.cam_azim = -90
        self.cam_speed = 0.5
        self.look_speed = 5
        
        # Background stars (Relative offsets)
        self.n_stars = 300
        # Create a sphere of stars far away
        r = 50.0
        theta = np.random.uniform(0, 2*np.pi, self.n_stars)
        phi = np.random.uniform(0, np.pi, self.n_stars)
        self.stars_offset_x = r * np.sin(phi) * np.cos(theta)
        self.stars_offset_y = r * np.sin(phi) * np.sin(theta)
        self.stars_offset_z = r * np.cos(phi)

        self.fig = None
        self.ax = None
        self.key_cid = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = np.random.uniform(-2, 2, 3) # Start near center
        self.euler = np.random.uniform(-0.2, 0.2, 3)
        self.curl_angle = np.random.uniform(-np.pi, np.pi) # Any initial orientation
        self.curvatures = np.random.uniform(-0.01, 0.01, self.n_controls)
        
        # Randomize Rain Direction
        # Random vector in lower hemisphere (z < 0)
        # We can just pick a random point on sphere and flip z if needed
        v = np.random.normal(0, 1, 3)
        v /= np.linalg.norm(v)
        if v[2] > 0: v[2] = -v[2]
        # Bias towards down slightly
        v[2] -= 0.5
        v /= np.linalg.norm(v)
        self.rain_dir = v

        self.current_step = 0
        self.sim.reset_wetness()
        self.sim.reset_wetness()
        self.sim.update(self.pos, self.euler, self.curvatures, self.curl_angle)
        return self._get_obs(), {}

    def _get_obs(self):
        # Calculate local rain direction
        # R maps local -> global
        # R.T maps global -> local
        # If sim.R is not set (first reset), it might be identity
        local_rain = self.sim.R.T @ self.rain_dir
        
        return np.concatenate((
            self.pos, 
            self.euler,
            [self.curl_angle],
            self.curvatures, 
            self.rain_dir,
            local_rain
        )).astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        
        d_pos = action[0:3] * self.max_linear_speed
        d_euler = action[3:6] * self.max_angular_speed
        d_curl = action[6] * self.max_curl_speed
        d_k = action[7:] * self.max_k_speed
        
        self.pos += d_pos
        self.euler += d_euler
        self.euler = (self.euler + np.pi) % (2 * np.pi) - np.pi
        
        self.curl_angle += d_curl
        self.curl_angle = (self.curl_angle + np.pi) % (2 * np.pi) - np.pi
        
        # Boundary Check / penalty
        # We allow it to move but penalize if it goes too far
        # Hard clip center
        self.pos = np.clip(self.pos, -self.workspace_bounds, self.workspace_bounds)
        
        # Floor Constraint (Hard)
        # Check lowest vertex z
        floor_level = -10.0
        if self.sim.vertices is not None:
             min_z = np.min(self.sim.vertices[:, :, 2])
             if min_z < floor_level:
                 # Push up
                 diff = floor_level - min_z
                 self.pos[2] += diff
                 # Update sim immediately so next step acts on valid state
                 self.sim.update(self.pos, self.euler, self.curvatures, self.curl_angle)

        # Try Update
        # Update always returns True now (Soft Constraint)
        prev_curvatures = self.curvatures.copy()
        self.curvatures += d_k
        self.curvatures = np.clip(self.curvatures, -2.0, 2.0)
        
        self.sim.update(self.pos, self.euler, self.curvatures, self.curl_angle)
        
        collision_penalty = 0.0
        if self.sim.self_intersecting:
            collision_penalty = 5.0 # Penalty for self-intersection
        
        if self.rain_mode == 'dynamic':
            # Drift rain direction
            # Small random rotation or just noise + normalize
            noise = np.random.normal(0, 0.05, 3) # Small drift
            self.rain_dir += noise
            
            # Bias towards down slightly to prevent it flipping up too easily
            self.rain_dir[2] -= 0.005 
            
            # Normalize
            self.rain_dir /= np.linalg.norm(self.rain_dir)
            
        # Apply Rain
        # We still apply rain for visualization/wetness mask (optional now for reward but good for vis)
        # Using a small number of drops for Vis is fine.
        if self.render_rain_effects:
            newly_wet_count = self.sim.apply_rain(rain_dir=self.rain_dir, n_drops=100)
        
        # Calculate Reward
        # Uses Robust Monte Carlo Projected Area
        wet_area = self.sim.calculate_wet_area(rain_dir=self.rain_dir)
        max_area = self.L * self.W
        
        # Normalized wetness penalty (0 to 1 ideally, but area can be slightly > L*W if strictly convex? No, projected area <= Surface Area)
        current_wet_fraction = wet_area / max_area
        
        # Reward Function:
        # Minimize Projected Area directly.
        # Stronger gradient.
        
        # Weights
        w_area = 20.0 
        
        wet_penalty = current_wet_fraction * w_area
        
        # Boundary Penalty
        # Check if any vertex is out of bounds
        clip_penalty = collision_penalty # Include self-collision here
        if self.sim.vertices is not None:
            # We check if absolute value of any vertex coordinate exceeds workspace_bounds
            max_extent = np.max(np.abs(self.sim.vertices))
            if max_extent > self.workspace_bounds:
                clip_penalty += (max_extent - self.workspace_bounds) * 5.0 
            
            # Floor proximity penalty (optional, but good for learning)
            min_z = np.min(self.sim.vertices[:, :, 2])
            if min_z < floor_level + 0.5:
                clip_penalty += (floor_level + 0.5 - min_z) * 5.0

        # Stability Penalty: Penalize high angular velocity
        # d_euler is in rang [-max_angular_speed, max_angular_speed]
        # We want to encourage it to be 0 when optimal.
        stability_penalty = np.sum(np.square(action[3:6])) * 0.005
        
        # Total Reward
        # We want to minimize wetness and minimize clipping.
        # Max reward = 1.0 (perfectly dry/edge-on, inside bounds)
        
        reward = 1.0 - wet_penalty - clip_penalty - stability_penalty
        
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, {"wet_area": wet_area, "clip_penalty": clip_penalty}

    def on_key_press(self, event):
        # View Rotation (Arrows)
        if event.key == 'up':
            self.cam_elev += self.look_speed
        elif event.key == 'down':
            self.cam_elev -= self.look_speed
        elif event.key == 'left':
            self.cam_azim -= self.look_speed
        elif event.key == 'right':
            self.cam_azim += self.look_speed
        
        rad = np.radians(self.cam_azim)
        dx = -np.sin(rad) * self.cam_speed
        dy = np.cos(rad) * self.cam_speed
        
        if event.key == 'w':
            self.cam_pos[0] += dx
            self.cam_pos[1] += dy
        elif event.key == 's':
            self.cam_pos[0] -= dx
            self.cam_pos[1] -= dy
        elif event.key == 'a':
            self.cam_pos[0] -= dy
            self.cam_pos[1] += dx
        elif event.key == 'd':
            self.cam_pos[0] += dy
            self.cam_pos[1] -= dx
        elif event.key == 'q':
            self.cam_pos[2] += self.cam_speed
        elif event.key == 'e':
            self.cam_pos[2] -= self.cam_speed
            
        self.cam_elev = np.clip(self.cam_elev, -90, 90)

    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.key_cid = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        self.ax.clear()
        
        self.ax.view_init(elev=self.cam_elev, azim=self.cam_azim)
        
        verts = self.sim.vertices
        if verts is None: return
        
        X = verts[:, :, 0]
        Y = verts[:, :, 1]
        Z = verts[:, :, 2]
        
        # Draw skybox
        sx = self.stars_offset_x + self.cam_pos[0]
        sy = self.stars_offset_y + self.cam_pos[1]
        sz = self.stars_offset_z + self.cam_pos[2]
        self.ax.scatter(sx, sy, sz, c='white', s=2, alpha=0.3)
        
        self.ax.set_facecolor('#000010') 
        self.ax.xaxis.set_pane_color((0.0, 0.0, 0.05, 1.0))
        self.ax.yaxis.set_pane_color((0.0, 0.0, 0.05, 1.0))
        self.ax.zaxis.set_pane_color((0.0, 0.0, 0.05, 1.0))
        self.ax.grid(False) 
        self.ax.set_axis_off() 
        
        # Rendering: Smooth Paper
        # Map wet_mask to colors
        # wet_mask is (n_profile, n_rulings) vertices
        # Face colors need to be (n_profile-1, n_rulings-1)
        # We'll just map vertex colors for simplicity if surface plot supports it, 
        # but plot_surface expects facecolors to match face dimensions or fit.
        
        # Let's simple create a color array based on Z or just uniform Cyan
        # But for wetness we want wet parts Blue.
        
        # wet_mask is at vertices.
        mask = self.sim.wet_mask
        # Create an RGBA array
        # Dry: Cyan (0, 1, 1, 0.9)
        # Wet: Blue (0, 0, 1, 0.9)
        
        colors = np.zeros(mask.shape + (4,))
        colors[~mask] = [0, 1, 1, 0.9] # Cyan
        colors[mask] = [0.1, 0.1, 0.8, 0.9] # Dark Blue
        
        self.ax.plot_surface(X, Y, Z, facecolors=colors, shade=False) # shade=False to use our colors exactly
        # Note: plot_surface facecolors expects the color array to match the grid shape (Z.shape)
        # It uses the color of the top-left vertex for the face usually? Or interpolates?
        # Actually for facecolors=Z.shape, it maps each patch to the interaction. 
        # Let's hope dimensionality works out (X, Y, Z have same shape as mask).
        
        # Set bounds
        R = 20.0 # Increased view range
        cx, cy, cz = self.cam_pos
        self.ax.set_xlim(cx - R, cx + R)
        self.ax.set_ylim(cy - R, cy + R)
        self.ax.set_zlim(cz - R, cz + R)
        
        self.ax.set_title(f"Rain: {self.rain_dir}", color='white')
        
        # Rain visualization (Directional)
        # We draw lines along -rain_dir
        # Rain falls along rain_dir
        d = self.rain_dir
        
        # Spawn lines around camera or paper
        center = self.pos
        for _ in range(10):
            # Random pt in box
            rx = np.random.uniform(center[0]-5, center[0]+5)
            ry = np.random.uniform(center[1]-5, center[1]+5)
            rz = np.random.uniform(center[2]+5, center[2]+15)
            
            p1 = np.array([rx, ry, rz])
            p2 = p1 + d * 5.0 # Line of length 5 along rain direction
            
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='white', alpha=0.1, linewidth=1)
            
        # Visible Wet Blots
        contacts_uv, contacts_w = self.sim.get_rain_contacts(n_drops=300, rain_dir=self.rain_dir)
        if len(contacts_w) > 0:
            self.ax.scatter(contacts_w[:, 0], contacts_w[:, 1], contacts_w[:, 2], c='blue', s=20, alpha=0.8)
            
        plt.pause(0.01)

    def close(self):
        if self.fig:
            plt.close(self.fig)

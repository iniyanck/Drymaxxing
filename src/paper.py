import numpy as np
from scipy.interpolate import interp1d

class PaperSim:
    def __init__(self, L=10.0, W=10.0, n_profile=50, n_rulings=50):
        """
        Initialize the Paper Simulation.
        
        Args:
            L (float): Length of the paper (along the bending profile).
            W (float): Width of the paper (along the straight rulings).
            n_profile (int): Number of discretization points along the profile magnitude.
            n_rulings (int): Number of discretization points along the width.
        """
        self.L = L
        self.W = W
        self.n_profile = n_profile
        self.n_rulings = n_rulings
        # Discretize arc length s from -L/2 to L/2
        self.s = np.linspace(-L/2, L/2, n_profile)
        # Discretize width w from -W/2 to W/2
        self.r = np.linspace(-W/2, W/2, n_rulings)
        
        self.vertices = None
        self.faces = None
        
        # State
        self.position = np.zeros(3)
        self.euler = np.zeros(3) # roll, pitch, yaw
        self.x_loc = None # Local profile X
        self.z_loc = None # Local profile Z
        self.R = np.eye(3) # Rotation matrix
        
        # Wetness State
        self.wet_mask = np.zeros((n_profile, n_rulings), dtype=bool)

    def _check_self_intersection(self, x, z):
        """
        Check if the profile curve (x, z) intersects itself.
        Simple O(N^2) segment intersection check.
        Adjacent segments are ignored.
        """
        n = len(x)
        if n < 4: return False
        
        # Pre-compute segments
        # P[i] = (x[i], z[i])
        # Segment i is P[i] -> P[i+1]
        
        # Vectorized approach or nested loop. Since N=50, nested loop is fine (2500 iter).
        # We can optimize slightly.
        
        def ccw(A, B, C):
            # Counter-clockwise check
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        def intersect(A, B, C, D):
            # Check if segment AB intersects CD
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

        points = np.column_stack((x, z))
        
        # Check all pairs of non-adjacent segments
        # Segment i: points[i] to points[i+1]
        # Segment j: points[j] to points[j+1]
        # Check if i and j are not adjacent (i+1 != j and i != j+1 and i!=j)
        
        for i in range(n - 1):
            A = points[i]
            B = points[i+1]
            # Start from i+2 to skip adjacent segment
            for j in range(i + 2, n - 1):
                C = points[j]
                D = points[j+1]
                
                if intersect(A, B, C, D):
                    return True
        return False

    def update(self, position, euler_angles, curvature_controls):
        """
        Update the paper geometry based on pose and curvature controls.
        
        Args:
            position (array-like): [x, y, z] coordinates of the center.
            euler_angles (array-like): [roll, pitch, yaw] in radians.
            curvature_controls (array-like): Control points for curvature along s.
            
        Returns:
            bool: True if update was successful (no collision), False otherwise.
        """
        # Calculate candidates first without updating self state
        
        # 1. Reconstruct curvature function k(s)
        controls = np.array(curvature_controls).flatten()
        n_controls = len(controls)
        
        if n_controls < 2:
            k_s = np.full_like(self.s, controls[0] if n_controls > 0 else 0)
        else:
            control_s = np.linspace(self.s[0], self.s[-1], n_controls)
            f_k = interp1d(control_s, controls, kind='linear', fill_value="extrapolate")
            k_s = f_k(self.s)
            
        # 2. Integrate to get profile curve in local XZ frame
        ds = self.s[1] - self.s[0]
        theta = np.cumsum(k_s) * ds
        # Fix tangent at center to be flat
        theta = theta - theta[len(theta)//2] 
        
        dx = np.cos(theta) * ds
        dz = np.sin(theta) * ds
        
        x_loc = np.cumsum(dx)
        z_loc = np.cumsum(dz)
        
        # Center the profile
        x_loc -= x_loc[len(x_loc)//2]
        z_loc -= z_loc[len(z_loc)//2]
        
        # Check Collision
        if self._check_self_intersection(x_loc, z_loc):
            return False

        # If valid, commit state
        self.x_loc = x_loc
        self.z_loc = z_loc
        self.position = np.array(position, dtype=np.float32)
        self.euler = np.array(euler_angles, dtype=np.float32)
        
        # 3. Generate 3D Vertices
        # Local Grid Construction (Paper Frame)
        # Paper extends along X (profile) and Y (width/rulings), Z is normal-ish
        # Actually our profile is in XZ plane, width along Y.
        
        X_local = x_loc[:, np.newaxis]  # (N_p, 1)
        Z_local = z_loc[:, np.newaxis]  # (N_p, 1)
        Y_local = self.r[np.newaxis, :] # (1, N_r)
        
        # Broadcast to full mesh points
        # shape (N_p, N_r)
        X_mesh = np.repeat(X_local, self.n_rulings, axis=1)
        Y_mesh = np.repeat(Y_local, self.n_profile, axis=0)
        Z_mesh = np.repeat(Z_local, self.n_rulings, axis=1)
        
        # Stack to (N_p, N_r, 3)
        local_verts = np.stack((X_mesh, Y_mesh, Z_mesh), axis=-1)
        
        # 4. Apply 6-DoF Transformation
        # Rotate
        roll, pitch, yaw = self.euler
        
        # Rotation Matrices
        # Rx (Roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        # Ry (Pitch)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        # Rz (Yaw)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined R = Rz * Ry * Rx
        self.R = Rz @ Ry @ Rx
        
        # Reshape for matrix multiplication: (N, 3)
        N_pts = self.n_profile * self.n_rulings
        flat_verts = local_verts.reshape(N_pts, 3)
        
        # Apply Rotation: v_rot = R . v_local^T  => (3, 3) . (3, N) = (3, N) -> Transpose to (N, 3)
        rotated_verts = (self.R @ flat_verts.T).T
        
        # Apply Translation
        world_verts = rotated_verts + self.position
        
        self.vertices = world_verts.reshape(self.n_profile, self.n_rulings, 3)
        return True
        
    def calculate_wet_area(self, rain_dir=None, resolution=100):
        """
        Approximate the projected area (wet area) relative to the rain direction using Monte Carlo.
        Minimizing this is the goal.
        """
        if self.vertices is None:
            return 0.0
            
        if rain_dir is None:
            rain_dir = np.array([0, 0, -1])
        
        # Normalize rain direction
        rain_dir = np.array(rain_dir, dtype=np.float64)
        norm = np.linalg.norm(rain_dir)
        if norm > 0:
            rain_dir = rain_dir / norm
        else:
            rain_dir = np.array([0, 0, -1])

        return self.get_wet_area()

    def get_wet_area(self):
        """
        Calculate total wet area based on wet_mask.
        """
        # Approximate area per cell
        # This is a bit rough since cells stretch, but for optimization it's okay.
        # Total area = L * W
        # N_cells = (n_profile-1) * (n_rulings-1) roughly
        # Let's simple count fraction of vertices wet * Total Area
        
        fraction_wet = np.mean(self.wet_mask)
        return fraction_wet * (self.L * self.W)

    def reset_wetness(self):
        self.wet_mask.fill(False)
        
    def apply_rain(self, rain_dir, n_drops=100):
        """
        Simulate rain drops and update wet_mask.
        """
        # Instantaneous Wetness: Reset mask
        self.wet_mask.fill(False)
        
        contacts_uv, _ = self.get_rain_contacts(n_drops=n_drops, rain_dir=rain_dir)

        
        if len(contacts_uv) > 0:
            # UV are normalized 0..1
            # Convert to indices
            # u -> profile index (0..n_profile-1)
            # v -> ruling index (0..n_rulings-1)
            
            us = contacts_uv[:, 0]
            vs = contacts_uv[:, 1]
            
            idx_u = (us * (self.n_profile - 1)).astype(int)
            idx_v = (vs * (self.n_rulings - 1)).astype(int)
            
            # Clip safe
            idx_u = np.clip(idx_u, 0, self.n_profile - 1)
            idx_v = np.clip(idx_v, 0, self.n_rulings - 1)
            
            # Count newly wet
            # Current mask at these indices
            current_wetness = self.wet_mask[idx_u, idx_v]
            # Convert to int (0 or 1), sum is existing wet count in this batch
            
            # We want to know how many traverse from False -> True
            # Let's perform the update and check difference or just check before
            
            # Simple way: just count how many False in current_wetness
            newly_wet = np.sum(~current_wetness)
            
            self.wet_mask[idx_u, idx_v] = True
            
            return newly_wet
            
        return 0


    def _monte_carlo_area(self, rain_dir, n_samples=2000):
        try:
            from matplotlib.path import Path
        except ImportError:
            return 0.0

        # We need to look at the paper from the perspective of the rain.
        # This means rotating the world so that rain_dir aligns with -Z (0, 0, -1).
        
        target = np.array([0, 0, -1])
        source = rain_dir
        
        # Rotation taking source to target
        v = np.cross(source, target)
        c = np.dot(source, target)
        s = np.linalg.norm(v)
        
        if s < 1e-9:
            if c > 0: R_view = np.eye(3)
            else: R_view = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            K = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R_view = np.eye(3) + K + (K @ K) * ((1 - c) / (s**2))

        # Rotate all vertices
        shape = self.vertices.shape
        flat_verts = self.vertices.reshape(-1, 3)
        view_verts = (R_view @ flat_verts.T).T
        view_verts = view_verts.reshape(shape)
        
        # Now we project to XY plane (which is perpendicular to Z, i.e., perpendicular to rain)
        pts = view_verts[:, :, :2].reshape(-1, 2)
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        
        area_bbox = (max_x - min_x) * (max_y - min_y)
        if area_bbox < 1e-9: return 0.0
        
        # 2. Random points in this view plane
        rand_x = np.random.uniform(min_x, max_x, n_samples)
        rand_y = np.random.uniform(min_y, max_y, n_samples)
        points = np.column_stack((rand_x, rand_y))
        
        # 3. Check collision
        hit_mask = np.zeros(n_samples, dtype=bool)
        
        for i in range(self.n_profile - 1):
            p1 = view_verts[i, 0, :2]
            p2 = view_verts[i+1, 0, :2]
            p3 = view_verts[i+1, -1, :2]
            p4 = view_verts[i, -1, :2]
            
            poly_verts = np.array([p1, p2, p3, p4, p1])
            path = Path(poly_verts)
            
            in_poly = path.contains_points(points)
            hit_mask = np.logical_or(hit_mask, in_poly)
            
        n_hits = np.sum(hit_mask)
        return (n_hits / n_samples) * area_bbox

        if len(valid_contacts_uv) > 0:
            # Also return the world coordinates for visualization
            # We already computed hit_y (local Y) and we can compute hit_x, hit_z from t_val
            # Then transform back to world.
            
            # Re-collect world points
            # We need to do this inside the loop or re-calculate. 
            # Let's just store them in the loop to avoid duplicate work.
            pass
            
        return np.array(valid_contacts_uv), np.array(valid_contacts_world)

    def get_rain_contacts(self, n_drops=100, bounds=None, rain_dir=None):
        """
        Simulate rain drops falling from rain_dir. 
        """
        if self.vertices is None:
            return np.empty((0, 2)), np.empty((0, 3))
            
        if rain_dir is None:
            rain_dir = np.array([0, 0, -1])
            
        # Normalize rain dir
        rain_dir = np.array(rain_dir, dtype=np.float64)
        norm = np.linalg.norm(rain_dir)
        if norm > 0: rain_dir /= norm
        else: rain_dir = np.array([0, 0, -1])

        # 1. Define a source plane perpendicular to rain_dir
        # Rotate world so rain_dir is -Z.
        target = np.array([0, 0, -1])
        source = rain_dir
        
        v = np.cross(source, target)
        c = np.dot(source, target)
        s = np.linalg.norm(v)
        
        if s < 1e-9:
            if c > 0: R_view = np.eye(3)
            else: R_view = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
        else:
            K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            R_view = np.eye(3) + K + (K @ K) * ((1-c)/(s**2))
            
        # Rotate vertices to view frame
        flat_verts = self.vertices.reshape(-1, 3)
        view_verts = (R_view @ flat_verts.T).T
        
        # Bounding box in view frame
        pts = view_verts[:, :2] # XY only
        min_x, min_y = np.min(pts, axis=0)
        max_x, max_y = np.max(pts, axis=0)
        
        pad = 5.0
        min_x -= pad; max_x += pad
        min_y -= pad; max_y += pad
        
        rand_x = np.random.uniform(min_x, max_x, n_drops)
        rand_y = np.random.uniform(min_y, max_y, n_drops)
        
        # These are origins in VIEW frame (where rain falls down -Z)
        z_start_view = np.max(view_verts[:, 2]) + 10.0
        
        origins_view = np.column_stack((rand_x, rand_y, np.full(n_drops, z_start_view)))
        dirs_view = np.tile(np.array([0.0, 0.0, -1.0]), (n_drops, 1))
        
        # Transform Rays back to World Frame
        origins_world = (R_view.T @ origins_view.T).T
        dirs_world = (R_view.T @ dirs_view.T).T 
        
        # Transform rays to Local Frame of the paper
        R_inv = self.R.T # Paper rotation inverse
        
        # Add directional noise to dirs_view before transforming
        # We want to perturb the direction (originally [0, 0, -1]) slightly for each drop
        # This makes the rain "randomized" / spray-like.
        # Noise level (standard deviation of angle roughly)
        noise_scale = 0.1 
        
        # We can add noise to x and y components of the direction in View Space.
        # Since main dir is -Z, adding small x, y gives angle deviation.
        noise = np.random.normal(0, noise_scale, (n_drops, 2))
        dirs_view_noisy = dirs_view.copy()
        dirs_view_noisy[:, 0] += noise[:, 0]
        dirs_view_noisy[:, 1] += noise[:, 1]
        
        # Re-normalize
        norms = np.linalg.norm(dirs_view_noisy, axis=1, keepdims=True)
        dirs_view_noisy /= norms
        
        # Use noisy dirs for transformation
        dirs_world = (R_view.T @ dirs_view_noisy.T).T
        
        origins_local = (R_inv @ (origins_world - self.position).T).T
        dirs_local = (R_inv @ dirs_world.T).T
        
        valid_contacts_uv = []
        valid_contacts_world = []
        
        loc_x = self.x_loc
        loc_z = self.z_loc
        total_s = self.s[-1] - self.s[0]
        s_start = self.s[0]

        for i in range(n_drops):
            O = origins_local[i]
            D = dirs_local[i]
            Ox, Oy, Oz = O
            Dx, Dy, Dz = D
            
            best_t = float('inf')
            hit_uv = None
            found_hit = False
            
            for k in range(len(loc_x) - 1):
                Ax, Az = loc_x[k], loc_z[k]
                Bx, Bz = loc_x[k+1], loc_z[k+1]
                
                DeltaX = Bx - Ax
                DeltaZ = Bz - Az
                
                det = Dx * DeltaZ - Dz * DeltaX
                if abs(det) < 1e-9: continue
                
                start_x = Ax - Ox
                start_z = Az - Oz
                
                t_val = (DeltaZ * start_x - DeltaX * start_z) / det
                u_seg = (Dz * start_x - Dx * start_z) / det 
                
                if 0.0 <= u_seg <= 1.0:
                    if t_val > 0 and t_val < best_t:
                        hit_y = Oy + t_val * Dy
                        if -self.W/2 <= hit_y <= self.W/2:
                            best_t = t_val
                            seg_len = self.s[k+1] - self.s[k]
                            current_s = self.s[k] + u_seg * seg_len
                            norm_u = (current_s - s_start) / total_s
                            norm_v = (hit_y - (-self.W/2)) / self.W
                            hit_uv = [norm_u, norm_v]
                            found_hit = True
                            
            if found_hit:
                valid_contacts_uv.append(hit_uv)
                # World Hit
                P_local = O + best_t * D
                P_world = (self.R @ P_local) + self.position
                valid_contacts_world.append(P_world)

        if len(valid_contacts_uv) > 0:
            return np.array(valid_contacts_uv), np.array(valid_contacts_world)
        else:
            return np.empty((0, 2)), np.empty((0, 3))

if __name__ == "__main__":
    # Test script
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    sim = PaperSim(L=10, W=5)
    
    # Test 6-DoF
    pos = [0, 0, 0]
    euler = [0, 0, 0]
    k = np.pi / 20.0
    
    sim.update(position=pos, euler_angles=euler, curvature_controls=[k, k])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = sim.vertices[:, :, 0]
    Y = sim.vertices[:, :, 1]
    Z = sim.vertices[:, :, 2]
    
    ax.plot_surface(X, Y, Z, alpha=0.5)
    
    # Test Slanted Rain
    rain_dir = np.array([1.0, 0.0, -1.0]) # Rain from +X +Z
    contacts_uv, contacts_w = sim.get_rain_contacts(n_drops=200, rain_dir=rain_dir)
    
    if len(contacts_w) > 0:
        ax.scatter(contacts_w[:,0], contacts_w[:,1], contacts_w[:,2], c='red')
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1])
    
    print(f"Wet Area: {sim.calculate_wet_area(rain_dir=rain_dir)}")
        
    plt.show()

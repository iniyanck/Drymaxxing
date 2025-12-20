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
        self.curl_angle = 0.0 # Angle of the bending axis in XY plane
        self.x_loc = None # Local profile X
        self.z_loc = None # Local profile Z
        self.R = np.eye(3) # Rotation matrix
        
        # Grid points of the flat paper (mesh material coordinates)
        # Defined once since paper is rigid
        xv, yv = np.meshgrid(self.s, self.r, indexing='ij')
        self.mat_x = xv # Material X coord
        self.mat_y = yv # Material Y coord
        
        # Wetness State
        self.wet_mask = np.zeros((n_profile, n_rulings), dtype=bool)
        self.self_intersecting = False

    def _check_self_intersection(self, x, z):
        """
        Check if the profile curve (x, z) intersects itself.
        Vectorized O(N^2) segment intersection check.
        Adjacent segments are ignored.
        """
        points = np.column_stack((x, z))
        n = len(points)
        if n < 4: return False
        
        # Create segments: (N-1) segments
        # each segment is defined by p1, p2
        # segments shape: (N-1, 2, 2) where last dim is (x,y)
        P1 = points[:-1]
        P2 = points[1:]
        
        # We need to compare segment i with segment j where j > i + 1
        n_seg = n - 1
        
        # Generate indices of pairs (i, j)
        idx_i, idx_j = np.triu_indices(n_seg, k=2)
        
        if len(idx_i) == 0:
            return False
            
        # Extract segment points
        # Segment 1: A->B
        A = P1[idx_i]
        B = P2[idx_i]
        
        # Segment 2: C->D
        C = P1[idx_j]
        D = P2[idx_j]
        
        # Vectorized CCW
        # ccw(P1, P2, P3): (y3-y1)*(x2-x1) > (y2-y1)*(x3-x1)
        
        def ccw(p1, p2, p3):
            val = (p3[:, 1] - p1[:, 1]) * (p2[:, 0] - p1[:, 0]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
            return val
            
        # CCW checks
        # intersect(A,B,C,D) is true if:
        # ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
        
        # Note: strict inequality for typical intersection check.
        # But for self-intersection of a continuous curve, touching is fine (it's connected).
        # We look for crossing.
        
        val_acd = ccw(A, C, D)
        val_bcd = ccw(B, C, D)
        val_abc = ccw(A, B, C)
        val_abd = ccw(A, B, D)
        
        # Check signs
        # intersect if signs differ: > 0 and < 0
        
        cond1 = (val_acd > 0) != (val_bcd > 0)
        cond2 = (val_abc > 0) != (val_abd > 0)
        
        # Also handle collinearity if needed, but for "crossing" simple sign check is usually enough.
        # Ideally check (val * val < 0) or similar.
        # Using strict sign comparison (True/False) works for crossing.
        
        # One edge case: if val is exactly 0. 
        # But with floats unlikely unless exact overlap.
        
        intersections = cond1 & cond2
        
        return np.any(intersections)

    def update(self, position, euler_angles, curvature_controls, curl_angle=0.0):
        """
        Update the paper geometry based on pose and curvature controls.
        
        Args:
            position (array-like): [x, y, z] coordinates of the center.
            euler_angles (array-like): [roll, pitch, yaw] in radians.
            curvature_controls (array-like): Control points for curvature along s.
            curl_angle (float): Angle of the curling axis. 0 = Curl along X (roll along Y).
            
        Returns:
            bool: True if update was successful (no collision), False otherwise.
        """
        # Save state
        self.position = np.array(position, dtype=np.float32)
        self.euler = np.array(euler_angles, dtype=np.float32)
        self.curl_angle = curl_angle
        
        # 1. Determine "Profile Space" coordinates
        # We rotate the material coordinates (mat_x, mat_y) by -curl_angle around Z.
        # u: component along bending direction (profile direction)
        # v: component along ruling direction (straight line)
        
        c = np.cos(curl_angle)
        s = np.sin(curl_angle)
        
        # u = x * cos(theta) + y * sin(theta)
        # v = -x * sin(theta) + y * cos(theta)
        # Note: This is a rotation of the coordinate system by theta.
        
        u_mesh = self.mat_x * c + self.mat_y * s
        v_mesh = -self.mat_x * s + self.mat_y * c
        
        # Determine necessary extent of profile generation
        u_min, u_max = np.min(u_mesh), np.max(u_mesh)
        
        # 2. Reconstruct curvature function k(s)
        controls = np.array(curvature_controls).flatten()
        n_controls = len(controls)
        
        # We need to generate the profile over [u_min, u_max]
        # But curvature controls are usually defined over the "length" of the paper.
        # If we rotate, the "effective length" changes.
        # Option A: Stretch curvature controls to cover new extent.
        # Option B: Curvature controls map to Material X, and we project? 
        #   No, that implies bending varies with X, which contradicts "generalized cylinder along angle".
        # Option C: Curvature is defined along the 'u' axis (bending axis).
        #   We assume controls cover [-L_eff/2, L_eff/2].
        #   Let's just map controls linearly to [u_min, u_max].
        
        # High-res profile generation for interpolation
        n_gen = int(self.n_profile * 1.5) # Higher res for interpolation
        s_gen = np.linspace(u_min, u_max, n_gen)
        
        if n_controls < 2:
            k_gen = np.full_like(s_gen, controls[0] if n_controls > 0 else 0)
        else:
            control_s = np.linspace(u_min, u_max, n_controls)
            f_k = interp1d(control_s, controls, kind='linear', fill_value="extrapolate")
            k_gen = f_k(s_gen)
            
        # 3. Integrate to get profile curve in (u, w) plane (where w is "up" in profile frame)
        ds = s_gen[1] - s_gen[0]
        theta = np.cumsum(k_gen) * ds
        # Center tangent? Maybe not necessary, but keeps it "flat" on average or at center.
        # Let's zero theta at u=0 (approx center of paper)
        zero_idx = np.argmin(np.abs(s_gen))
        theta = theta - theta[zero_idx]
        
        du = np.cos(theta) * ds
        dw = np.sin(theta) * ds
        
        u_profile = np.cumsum(du)
        w_profile = np.cumsum(dw)
        
        # Adjust u_profile so it matches s_gen scale approx (arc length parameterization)
        # u_profile starts at 0? No cumsum.
        
        # Re-center profile at u=0
        u_profile -= u_profile[zero_idx]
        w_profile -= w_profile[zero_idx]
        
        # Store profile s-coordinates for ray casting
        # s_gen is the variable along the profile curve (u in Surface Frame)
        self.profile_s = s_gen
        
        # Since u_mesh determines the ARC LENGTH parameter 's' for the profile,
        # We need X_profile(s) and Z_profile(s).
        # Wait, s_gen IS the arc length parameter.
        # The integration gives us position (u_prof, w_prof) in the 2D plane.
        # Ideally u_profile approx equals s_gen if curve is flat.
        
        # But we need to lookup (X_local, Z_local) given arc-length 's' (which is our u_mesh value).
        # Our integration results:
        # P(s) = ( \int cos(theta) ds, \int sin(theta) ds )
        # This P(s) is the position in the rotated frame's (X', Z') plane.
        # Let's call them (x_prime, z_prime).
        
        x_prime_gen = u_profile # This is the coordinate along the "ground" in rotated frame? No.
        # No, u_profile is the X-coordinate in the profile plane.
        # We integrated cos(theta) ds.
        
        z_prime_gen = w_profile
        
        # Interpolate x_prime(s), z_prime(s) for all s in u_mesh
        # u_mesh contains arc-length values.
        
        # We need to map s_gen to x_prime_gen using interpolation
        # s_gen is the arc length input.
        f_x = interp1d(s_gen, x_prime_gen, kind='linear', fill_value="extrapolate")
        f_z = interp1d(s_gen, z_prime_gen, kind='linear', fill_value="extrapolate")
        
        profile_x = f_x(u_mesh)
        profile_z = f_z(u_mesh)
        
        # Now we have surface points in the Rotated Frame:
        # X_rot = profile_x
        # Y_rot = v_mesh (Rulings are straight lines along Y_rot)
        # Z_rot = profile_z
        
        # 4. Transform back to Paper Frame (Material Frame Aligned?? No, just un-rotate curl angle)
        # We rotated Material Coords by -curl to get (u, v).
        # The surface is defined in this (u, v, z) space efficiently.
        # To get back to "Local Paper Frame" (where bounds are approx L x W), we rotate by +curl around Z.
        
        # P_local = R_curl @ P_rot
        # P_rot = [profile_x, v_mesh, profile_z]
        
        # Rotation around Z axis (Z_rot is actually Z_local)
        # x_local = x_rot * cos + y_rot * -sin  (Inverse rotation)
        # y_local = x_rot * sin + y_rot * cos
        
        # c, s are cos(curl), sin(curl)
        
        # X_rot = profile_x
        # Y_rot = v_mesh
        
        x_local = profile_x * c - v_mesh * s
        y_local = profile_x * s + v_mesh * c
        z_local = profile_z
        
        # Store for collision/rendering
        # Note: self.x_loc, self.z_loc are used for calculating bbox/collision in 2D profile.
        # With arbitrary rotation, the self-intersection check is tricky if we rely on 1D profile.
        # But since it IS a generalized cylinder, self-intersection only happens in the 2D profile plane.
        # So we can just check (x_prime_gen, z_prime_gen) for intersection!
        self.self_intersecting = self._check_self_intersection(x_prime_gen, z_prime_gen)
        
        self.x_loc = x_prime_gen
        self.z_loc = z_prime_gen
        
        # 5. Transform to World
        # Combined Rotation
        roll, pitch, yaw = self.euler
        
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        
        self.R = Rz @ Ry @ Rx
        
        # Reshape to (N, 3)
        local_verts = np.stack((x_local, y_local, z_local), axis=-1)
        flat_verts = local_verts.reshape(-1, 3)
        
        # Apply World Rotation
        rotated_verts = (self.R @ flat_verts.T).T
        
        # Apply Translation
        self.vertices = (rotated_verts + self.position).reshape(self.n_profile, self.n_rulings, 3)
        
        return True
        
    def calculate_wet_area(self, rain_dir=None, resolution=100):
        """
        Calculate robust projected area using Monte Carlo.
        This provides a smooth gradient for minimizing exposure.
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

        # Use Robust Monte Carlo
        return self._monte_carlo_area(rain_dir, n_samples=2000)

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
        # Optimization: Create one single Compound Path for all quads.
        # This replaces the loop over n_profile with a single vectorized check.
        
        # Extract quads
        # view_verts shape: (n_profile, n_rulings, 3)
        # We use the strip definition: Quads formed by profile edges at ruling 0 and -1
        
        P1 = view_verts[:-1, 0, :2]    # (N, 2)
        P2 = view_verts[1:, 0, :2]     # (N, 2)
        P3 = view_verts[1:, -1, :2]    # (N, 2)
        P4 = view_verts[:-1, -1, :2]   # (N, 2)
        
        # Ensure consistent winding (CCW) to prevents overlaps from canceling out (winding rule)
        # Calculate signed area (shoelace for quad) approx or cross product
        # Vectorized Cross Product (2D) for P1->P2 and P1->P4 (approx) or just check signed area
        # Area = 0.5 * |(x1y2 - y1x2) + (x2y3 - y2x3) + (x3y4 - y3x4) + (x4y1 - y4x1)|
        # We need the signed version.
        
        x1, y1 = P1[:,0], P1[:,1]
        x2, y2 = P2[:,0], P2[:,1]
        x3, y3 = P3[:,0], P3[:,1]
        x4, y4 = P4[:,0], P4[:,1]
        
        signed_area = 0.5 * (
            (x1*y2 - x2*y1) + 
            (x2*y3 - x3*y2) + 
            (x3*y4 - x4*y3) + 
            (x4*y1 - x1*y4)
        )
        
        # Identify CW quads (area < 0) and swap to make them CCW
        # We can just swap P2 and P4 for those indices
        swap_mask = signed_area < 0
        
        # Create final arrays
        # We'll build the array (N, 5, 2)
        quads = np.zeros((len(P1), 5, 2))
        
        # Standard: P1 -> P2 -> P3 -> P4 -> P1
        quads[~swap_mask, 0] = P1[~swap_mask]
        quads[~swap_mask, 1] = P2[~swap_mask]
        quads[~swap_mask, 2] = P3[~swap_mask]
        quads[~swap_mask, 3] = P4[~swap_mask]
        quads[~swap_mask, 4] = P1[~swap_mask]
        
        # Swapped: P1 -> P4 -> P3 -> P2 -> P1 (Reversed)
        quads[swap_mask, 0] = P1[swap_mask]
        quads[swap_mask, 1] = P4[swap_mask] # Swapped
        quads[swap_mask, 2] = P3[swap_mask]
        quads[swap_mask, 3] = P2[swap_mask] # Swapped
        quads[swap_mask, 4] = P1[swap_mask]
        
        # Flatten to (N*5, 2)
        all_verts = quads.reshape(-1, 2)
        
        # Create Codes
        n_quads = len(P1)
        codes = np.full(n_quads * 5, Path.LINETO, dtype=np.uint8)
        codes[0::5] = Path.MOVETO
        codes[4::5] = Path.CLOSEPOLY
        
        path = Path(all_verts, codes)
        
        # Single check
        hit_mask = path.contains_points(points)
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
        
        # Transform Local -> Surface Frame (Rotate by -curl angle)
        # We need this because the ray-surface intersection logic works in the Surface Frame (where profile is along X, rulings along Y)
        # Local Frame: Material Frame (rotated by curl angle) -> No, Local Frame is "Paper Centered" but arbitrarily rotated by curl.
        # Wait, self.vertices were constructed by:
        # 1. Material (x, y) -> Rotate(-curl) -> Surface (u, v) -> Profile(u) -> 3D Surface P_surf
        # 2. P_local = Rotate(+curl) @ P_surf
        # So to go Local -> Surface, we Rotate(-curl).
        
        c = np.cos(-self.curl_angle)
        s = np.sin(-self.curl_angle)
        # Matrix for Rotation(-curl)
        R_curl_inv = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        origins_surf = (R_curl_inv @ origins_local.T).T
        dirs_surf = (R_curl_inv @ dirs_local.T).T
        
        valid_contacts_uv = []
        valid_contacts_world = []
        
        # Profile in Surface Frame
        loc_x = self.x_loc # Profile X (in Surface Frame)
        loc_z = self.z_loc # Profile Z (in Surface Frame)
        profile_s = self.profile_s # S coordinate corresponding to loc_x/loc_z
        
        # Loop over drops
        for i in range(n_drops):
            O = origins_surf[i]
            D = dirs_surf[i]
            Ox, Oy, Oz = O
            Dx, Dy, Dz = D
            
            best_t = float('inf')
            hit_uv = None
            found_hit = False
            
            # Intersect with generalized cylinder profile
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
                        # Found intersection with infinite strip in Surface Frame
                        hit_v_surf = Oy + t_val * Dy
                        
                        # We need 'u' coordinate in Surface Frame (arc length along profile)
                        seg_len = profile_s[k+1] - profile_s[k]
                        hit_u_surf = profile_s[k] + u_seg * seg_len
                        
                        # Check Bounds in Material Frame!
                        # Transform Surface (u, v) -> Material (x, y)
                        # We rotated Material by -curl to get Surface.
                        # So Material = Rotate(+curl) @ Surface.
                        
                        # Note: hit_u_surf is the 'x' coord after rotation.
                        # hit_v_surf is the 'y' coord after rotation.
                        
                        cc = np.cos(self.curl_angle)
                        ss = np.sin(self.curl_angle)
                        
                        mat_x = hit_u_surf * cc - hit_v_surf * ss
                        mat_y = hit_u_surf * ss + hit_v_surf * cc
                        
                        # Material Bounds: [-L/2, L/2] x [-W/2, W/2]
                        if (-self.L/2 <= mat_x <= self.L/2) and (-self.W/2 <= mat_y <= self.W/2):
                            best_t = t_val
                            
                            norm_u = (mat_x - (-self.L/2)) / self.L
                            norm_v = (mat_y - (-self.W/2)) / self.W
                            hit_uv = [norm_u, norm_v]
                            found_hit = True
                            
            if found_hit:
                valid_contacts_uv.append(hit_uv)
                # World Hit
                # We can calculate P_world from t_val on the original ray
                # Ray was origin in World? No, Origins were Local (before surface transform).
                # Actually we have origins_world available.
                
                P_world = origins_world[i] + best_t * dirs_world[i]
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
    
    sim.update(position=pos, euler_angles=euler, curvature_controls=[k, k], curl_angle=np.pi/4)
    
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

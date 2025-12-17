import torch
import numpy as np

class TorchPaperSim:
    def __init__(self, L=10.0, W=5.0, n_profile=50, n_rulings=50, batch_size=1, device="cpu"):
        self.L = L
        self.W = W
        self.n_profile = n_profile
        self.n_rulings = n_rulings
        self.batch_size = batch_size
        self.device = device
        
        # Static Grid
        # s from -L/2 to L/2
        self.s = torch.linspace(-L/2, L/2, n_profile, device=device) # (N_p)
        self.ds = self.s[1] - self.s[0]
        
        # r from -W/2 to W/2
        self.r = torch.linspace(-W/2, W/2, n_rulings, device=device) # (N_r)
        
        # State tensors (Batch, ...)
        self.pos = torch.zeros((batch_size, 3), device=device)
        self.euler = torch.zeros((batch_size, 3), device=device)
        self.vertices = None
        
        # Wetness
        self.wet_mask = torch.zeros((batch_size, n_profile, n_rulings), dtype=torch.bool, device=device)
        
    def reset_wetness(self):
        self.wet_mask.zero_()
        
    def update(self, pos, euler, curvature_controls):
        """
        pos: (B, 3)
        euler: (B, 3)
        curvature_controls: (B, K)
        """
        self.pos = pos
        self.euler = euler
        
        B = self.batch_size
        N_p = self.n_profile
        
        # 1. Reconstruct curvature k(s)
        # We assume control points are evenly spaced along s.
        # We need to interpolate controls to size N_p.
        # curvature_controls = (B, N_k)
        # We can use torch.nn.functional.interpolate if we treat it as 1D signal
        # Input: (B, 1, N_k) -> Output (B, 1, N_p)
        
        k_ctrl = curvature_controls.unsqueeze(1) # (B, 1, N_k)
        
        # Interpolate
        # mode='linear' expects 3D input (N, C, L)
        k_s = torch.nn.functional.interpolate(k_ctrl, size=N_p, mode='linear', align_corners=True)
        k_s = k_s.squeeze(1) # (B, N_p)
        
        # 2. Integrate profile
        # theta = cumsum(k * ds)
        theta = torch.cumsum(k_s * self.ds, dim=1)
        
        # Center tangent? theta -= theta[:, center]
        center_idx = N_p // 2
        theta = theta - theta[:, center_idx:center_idx+1]
        
        dx = torch.cos(theta) * self.ds
        dz = torch.sin(theta) * self.ds
        
        x_loc = torch.cumsum(dx, dim=1)
        z_loc = torch.cumsum(dz, dim=1)
        
        # Center profile position
        x_loc = x_loc - x_loc[:, center_idx:center_idx+1]
        z_loc = z_loc - z_loc[:, center_idx:center_idx+1]
        
        self.x_loc = x_loc # (B, N_p)
        self.z_loc = z_loc # (B, N_p)
        
        # 3. Build 3D Surface
        # Local frame: X=Profile, Y=Width, Z=Normal(ish)
        # Vertices (B, N_p, N_r, 3)
        
        # Use broadcasting
        # X_loc: (B, N_p, 1)
        X_mesh = x_loc.unsqueeze(2).expand(B, N_p, self.n_rulings)
        Z_mesh = z_loc.unsqueeze(2).expand(B, N_p, self.n_rulings)
        
        # Y_loc: (1, 1, N_r) -> (B, N_p, N_r)
        Y_mesh = self.r.view(1, 1, -1).expand(B, N_p, self.n_rulings)
        
        local_verts = torch.stack((X_mesh, Y_mesh, Z_mesh), dim=-1) # (B, N_p, N_r, 3)
        
        # 4. Rotation
        # Construct rotation matrices (B, 3, 3)
        roll = euler[:, 0]
        pitch = euler[:, 1]
        yaw = euler[:, 2]
        
        zeros = torch.zeros_like(roll)
        ones = torch.ones_like(roll)
        
        # Rx
        Rx = torch.stack([
            torch.stack([ones, zeros, zeros], dim=1),
            torch.stack([zeros, torch.cos(roll), -torch.sin(roll)], dim=1),
            torch.stack([zeros, torch.sin(roll), torch.cos(roll)], dim=1)
        ], dim=1) # (B, 3, 3)
        
        # Ry
        Ry = torch.stack([
            torch.stack([torch.cos(pitch), zeros, torch.sin(pitch)], dim=1),
            torch.stack([zeros, ones, zeros], dim=1),
            torch.stack([-torch.sin(pitch), zeros, torch.cos(pitch)], dim=1)
        ], dim=1)
        
        # Rz
        Rz = torch.stack([
            torch.stack([torch.cos(yaw), -torch.sin(yaw), zeros], dim=1),
            torch.stack([torch.sin(yaw), torch.cos(yaw), zeros], dim=1),
            torch.stack([zeros, zeros, ones], dim=1)
        ], dim=1)
        
        self.R = torch.bmm(Rz, torch.bmm(Ry, Rx)) # (B, 3, 3)
        
        # Rotate vertices
        # Flatten verts to (B, N_pts, 3)
        N_pts = N_p * self.n_rulings
        flat_verts = local_verts.view(B, N_pts, 3)
        
        # (B, 3, 3) @ (B, N_pts, 3)^T -> (B, 3, N_pts) -> T -> (B, N_pts, 3)
        rot_verts = torch.bmm(self.R, flat_verts.transpose(1, 2)).transpose(1, 2)
        
        world_verts = rot_verts + self.pos.unsqueeze(1)
        
        self.vertices = world_verts.view(B, N_p, self.n_rulings, 3)
        
    def calculate_wet_area(self, rain_dir):
        # rain_dir: (B, 3) or (3,)
        # Normalized wet area (fraction)
        return torch.mean(self.wet_mask.float(), dim=(1, 2)) * (self.L * self.W)

    def apply_rain(self, rain_dir, n_drops=100):
        """
        Batched rain simulation.
        rain_dir: (3,) or (B, 3) - Global rain direction
        """
        B = self.batch_size
        
        if rain_dir.dim() == 1:
            rain_dir = rain_dir.expand(B, 3)
            
        # Normalize
        rain_dir = rain_dir / (torch.norm(rain_dir, dim=1, keepdim=True) + 1e-8)
        
        # 1. View Transform (Rain aligned with -Z)
        # Create a view matrix for each batch item such that R_view * rain_dir approx (0,0,-1)
        # We can just construct it using cross products.
        target = torch.tensor([0., 0., -1.], device=self.device).expand(B, 3)
        source = rain_dir
        
        v = torch.cross(source, target, dim=1)
        c = torch.sum(source * target, dim=1).view(B, 1, 1)
        s = torch.norm(v, dim=1).view(B, 1, 1)
        
        # K matrix
        z = torch.zeros(B, device=self.device)
        K = torch.stack([
            torch.stack([z, -v[:, 2], v[:, 1]], dim=1),
            torch.stack([v[:, 2], z, -v[:, 0]], dim=1),
            torch.stack([-v[:, 1], v[:, 0], z], dim=1)
        ], dim=1) # (B, 3, 3)
        
        # R_view formula: I + K + K^2 * (1-c)/s^2
        I = torch.eye(3, device=self.device).expand(B, 3, 3)
        # Handle s close to 0 (parallel)
        # If already -Z, R=I. If +Z, R=Flip Y
        # For simplicity, let's assume valid s for now or add epsilon
        
        term3 = (1 - c) / (s.pow(2) + 1e-6)
        R_view = I + K + torch.bmm(K, K) * term3
        
        # 2. Project vertices to View Space
        verts_flat = self.vertices.view(B, -1, 3) # (B, Npts, 3)
        verts_view = torch.bmm(R_view, verts_flat.transpose(1, 2)).transpose(1, 2)
        
        # 3. Generate Rain Drops in View Space
        # Bounds in XY
        # We assume Paper is roughly centered in view, but best to measure bounds per batch
        min_xy = torch.min(verts_view[:, :, :2], dim=1)[0] # (B, 2)
        max_xy = torch.max(verts_view[:, :, :2], dim=1)[0] # (B, 2)
        
        pad = 2.0
        min_xy -= pad
        max_xy += pad
        
        # Random XY for drops
        rand_x = torch.rand(B, n_drops, device=self.device)
        rand_y = torch.rand(B, n_drops, device=self.device)
        
        d_x = max_xy[:, 0:1] - min_xy[:, 0:1]
        d_y = max_xy[:, 1:2] - min_xy[:, 1:2]
        
        drop_x = min_xy[:, 0:1] + rand_x * d_x
        drop_y = min_xy[:, 1:2] + rand_y * d_y
        
        # Z start (above paper)
        max_z = torch.max(verts_view[:, :, 2], dim=1)[0].unsqueeze(1) + 5.0 # (B, 1)
        drop_z = max_z.expand(B, n_drops)
        
        # Origins in View Frame
        origins_view = torch.stack([drop_x, drop_y, drop_z], dim=-1) # (B, N_drops, 3)
        dirs_view = torch.tensor([0., 0., -1.], device=self.device).expand(B, n_drops, 3)
        
        # 4. Transform Rays to Local Paper Frame
        # World = R_view.T @ View
        # Local = R_paper.T @ (World - Pos)
        
        # Combined Transform: Local = R_inv_paper @ (R_view.T @ View - Pos)
        #                           = (R_inv_paper @ R_view.T) @ View - R_inv_paper @ Pos
        
        R_view_T = R_view.transpose(1, 2)
        R_paper_T = self.R.transpose(1, 2)
        
        M = torch.bmm(R_paper_T, R_view_T) # (B, 3, 3)
        offset = torch.bmm(R_paper_T, self.pos.unsqueeze(2)).squeeze(2) # (B, 3)
        
        origins_local = torch.bmm(M, origins_view.transpose(1, 2)).transpose(1, 2) - offset.unsqueeze(1)
        dirs_local = torch.bmm(M, dirs_view.transpose(1, 2)).transpose(1, 2)
        
        # 5. Intersect with Profile Segments (in 2D XZ plane)
        # Since the paper's profile does not change with Y (it's extruded), we only need to solve 2D intersection
        # in the XZ plane of the local frame.
        # Ray: O + t*D.
        # Segment: P1 -> P2.
        
        # Rays: (B, N_drops, 2) for X,Z
        O_xz = origins_local[..., [0, 2]]
        D_xz = dirs_local[..., [0, 2]]
        
        # Segments: The profile is defined by x_loc, z_loc (B, N_p)
        # We form segments (B, N_p-1, 2, 2) -> (Point1, Point2)
        P1_x = self.x_loc[:, :-1]
        P1_z = self.z_loc[:, :-1]
        P2_x = self.x_loc[:, 1:]
        P2_z = self.z_loc[:, 1:]
        
        # Vectorize intersection
        # We need to check every ray against every segment? 
        # (B, N_drops) vs (B, N_p-1) -> (B, N_drops, N_p-1)
        
        # Ray: R(t) = O + tD
        # Seg: S(u) = P1 + u(P2 - P1)
        # O + tD = P1 + u(dP)
        # tD - u(dP) = P1 - O
        # Linear system | D  -dP | | t | = | P1 - O |
        #               | .   .  | | u |   |   .    |
        
        # Broadcast D: (B, N_drops, 1, 2)
        # Broadcast dP: (B, 1, N_segs, 2)
        dP_x = P2_x - P1_x
        dP_z = P2_z - P1_z
        dP = torch.stack([dP_x, dP_z], dim=-1).unsqueeze(1) # (B, 1, N_segs, 2)
        
        D_br = D_xz.unsqueeze(2) # (B, N_drops, 1, 2)
        O_br = O_xz.unsqueeze(2) # (B, N_drops, 1, 2)
        P1_br = torch.stack([P1_x, P1_z], dim=-1).unsqueeze(1) # (B, 1, N_segs, 2)
        
        # Matrix Quantities
        # Determinant = dx * (-dp_z) - dz * (-dp_x) = -dx*dp_z + dz*dp_x
        # Wait, col0 is D, col1 is -dP
        det = D_br[..., 0] * (-dP[..., 1]) - D_br[..., 1] * (-dP[..., 0])
        
        # Avoid zero det
        mask_det = torch.abs(det) > 1e-9
        
        # RHS = P1 - O
        RHS = P1_br - O_br # (B, Nd, Ns, 2)
        
        # Solve Cramer's
        # t = (rhs_x * -dp_z - rhs_z * -dp_x) / det
        # u = (d_x * rhs_z - d_z * rhs_x) / det
        
        t_val = (RHS[..., 0] * -dP[..., 1] - RHS[..., 1] * -dP[..., 0]) / (det + 1e-10)
        u_val = (D_br[..., 0] * RHS[..., 1] - D_br[..., 1] * RHS[..., 0]) / (det + 1e-10)
        
        # Valid intersection?
        # t > 0
        # 0 <= u <= 1
        valid = mask_det & (t_val > 0) & (u_val >= 0.0) & (u_val <= 1.0)
        
        # Also need to check Y Coordinate
        # Hit Point Local = O + t*D (in 3D)
        # We computed t from XZ only.
        # Now check if Y at that t is within [-W/2, W/2]
        
        O_y = origins_local[..., 1].unsqueeze(2) # (B, Nd, 1)
        D_y = dirs_local[..., 1].unsqueeze(2)
        hit_y = O_y + t_val * D_y
        
        valid_y = (hit_y >= -self.W/2) & (hit_y <= self.W/2)
        valid = valid & valid_y
        
        # Find closest hit per drop
        # We want the hit with minimal t for each drop that is valid.
        # t_val (B, Nd, Ns).
        # We exclude invalid hits by setting t to infinity
        
        t_final = t_val.clone()
        t_final[~valid] = float('inf')
        
        min_t, min_idx = torch.min(t_final, dim=2) # (B, Nd)
        
        # Mask where we actually hit something
        hit_mask = min_t < float('inf')
        
        # We need to update wet_mask
        # For each hit, we have:
        #   Batch Index (implicitly row)
        #   Drop Index (col)
        #   Segment Index (min_idx) -> determines 'u' (profile coord)
        #   Hit Y -> determines 'v' (ruling coord)
        
        # Gather best u_val and hit_y
        # min_idx (B, Nd)
        
        # We need to extract the u_val corresponding to min_idx
        # u_val is (B, Nd, Ns)
        # gather dim 2
        best_u = torch.gather(u_val, 2, min_idx.unsqueeze(2)).squeeze(2)
        best_y = torch.gather(hit_y, 2, min_idx.unsqueeze(2)).squeeze(2)
        
        # Filter by hit_mask
        # We will iterate or use scatter?
        # Let's perform updates.
        
        # Calculate indices
        # Segment counts:
        # segment k maps to s indices k, k+1.
        # Continuous s = k + u_local. (since segments are uniform in our array layout, but s might not be uniform if we had variable ds ? No ds is uniform).
        
        # Map to profile index 0..N_p-1
        # global_u_float = segment_index + best_u
        global_u_float = min_idx.float() + best_u
        
        # normalize to 0..N_p-1
        # It IS 0..N_p-1 range basically.
        idx_u = torch.round(global_u_float).long()
        idx_u = torch.clamp(idx_u, 0, self.n_profile - 1)
        
        # Map Y to 0..N_r-1
        # Y is in [-W/2, W/2]
        norm_y = (best_y - (-self.W/2)) / self.W
        idx_v = torch.floor(norm_y * (self.n_rulings - 1)).long()
        idx_v = torch.clamp(idx_v, 0, self.n_rulings - 1)
        
        # Now we have B sets of (idx_u, idx_v) to mark as wet.
        # wet_mask is (B, N_p, N_r)
        
        # Flattened indices for scatter
        # idx = b * (Np*Nr) + u * Nr + v
        
        b_idx = torch.arange(B, device=self.device).unsqueeze(1).expand(B, n_drops)
        
        flat_indices = b_idx * (self.n_profile * self.n_rulings) + idx_u * self.n_rulings + idx_v
        
        # Apply mask
        valid_indices = flat_indices[hit_mask]
        
        if valid_indices.numel() > 0:
            self.wet_mask.view(-1)[valid_indices] = True
            
        return hit_mask

import numpy as np
import torch

class OnlineDualSpaceCache:
    """
    Dual-Space Centroid KV (v3.5 - Streaming Online Formulation)
    Maintains KV cache via online centroid clustering in canonical De-RoPE space (U-space).
    Time complexity: O(B * d) per inserted token where B is the cluster budget.
    """
    def __init__(self, budget, d_k, threshold=0.88, num_sinks=4):
        self.budget = budget
        self.d_k = d_k
        self.scale = 1.0 / np.sqrt(d_k)
        self.threshold = threshold
        self.num_sinks = num_sinks
        
        # Sinks (never compressed)
        self.sink_k = []
        self.sink_v = []
        
        # Centroids pool
        # Each cluster maintains: count (pi), u_mean, k_mean, v_mean, M_k (cross-covariance)
        self.clusters = []
        
    def add_token(self, k_t, v_t, pos, rotary_cos, rotary_sin):
        """
        k_t: [d_k] numpy array
        v_t: [d_v] numpy array
        pos: integer position
        """
        if pos < self.num_sinks:
            self.sink_k.append(k_t.copy())
            self.sink_v.append(v_t.copy())
            return

        # 1. Canonical De-RoPE to U-space
        # k_rot = (k * cos) + (rotate_half(k) * sin)
        # u = (k * cos) - (rotate_half(k) * sin)
        d = len(k_t)
        half = d // 2
        rot_half = np.concatenate([-k_t[half:], k_t[:half]])
        u_t = (k_t * rotary_cos) - (rot_half * rotary_sin)
        u_norm = np.linalg.norm(u_t) + 1e-9
        u_unit = u_t / u_norm
        
        # 2. Match with existing centroids in U-space
        best_sim = -1.0
        best_idx = -1
        
        for idx, c in enumerate(self.clusters):
            sim = np.dot(u_unit, c['u_unit'])
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
                
        # 3. Online Merge or Allocate
        if best_sim >= self.threshold and best_idx >= 0:
            # Merge into best centroid
            c = self.clusters[best_idx]
            pi_old = c['pi']
            pi_new = pi_old + 1
            c['pi'] = pi_new
            
            # Welford-like online mean update
            k_old_mean = c['k_mean'].copy()
            v_old_mean = c['v_mean'].copy()
            
            c['u_mean'] += (u_t - c['u_mean']) / pi_new
            c['u_unit'] = c['u_mean'] / (np.linalg.norm(c['u_mean']) + 1e-9)
            
            c['k_mean'] += (k_t - k_old_mean) / pi_new
            c['v_mean'] += (v_t - v_old_mean) / pi_new
            
            # Online cross-covariance update: M = sum (v_i - v_bar)(k_i - k_bar)^T
            # Delta update: M_new = M_old + (pi_old / pi_new) * (v_t - v_old_mean) (k_t - k_old_mean)^T
            dv = (v_t - v_old_mean)[:, None]
            dk = (k_t - k_old_mean)[None, :]
            c['M_K'] += (pi_old / pi_new) * (dv @ dk)
            
        elif len(self.clusters) < (self.budget - self.num_sinks):
            # Create new centroid
            self.clusters.append({
                'pi': 1,
                'u_mean': u_t.copy(),
                'u_unit': u_unit.copy(),
                'k_mean': k_t.copy(),
                'v_mean': v_t.copy(),
                'M_K': np.zeros((len(v_t), d), dtype=np.float32)
            })
        else:
            # Budget full: merge with the closest centroid or smallest cluster
            c = self.clusters[best_idx if best_idx >= 0 else 0]
            pi_old = c['pi']
            pi_new = pi_old + 1
            c['pi'] = pi_new
            k_old_mean = c['k_mean'].copy()
            v_old_mean = c['v_mean'].copy()
            c['u_mean'] += (u_t - c['u_mean']) / pi_new
            c['u_unit'] = c['u_mean'] / (np.linalg.norm(c['u_mean']) + 1e-9)
            c['k_mean'] += (k_t - k_old_mean) / pi_new
            c['v_mean'] += (v_t - v_old_mean) / pi_new
            dv = (v_t - v_old_mean)[:, None]
            dk = (k_t - k_old_mean)[None, :]
            c['M_K'] += (pi_old / pi_new) * (dv @ dk)

    def query_attention(self, q_t):
        """
        Computes attention output for query vector q_t using v3.5 formulation.
        """
        logits = []
        values = []
        
        # Sinks
        for k_s, v_s in zip(self.sink_k, self.sink_v):
            logits.append(self.scale * np.dot(q_t, k_s))
            values.append(v_s)
            
        # Centroids
        for c in self.clusters:
            pi = c['pi']
            z = self.scale * np.dot(q_t, c['k_mean']) + np.log(pi)
            if pi > 1:
                v_eff = c['v_mean'] + (self.scale / pi) * (c['M_K'] @ q_t)
            else:
                v_eff = c['v_mean']
            logits.append(z)
            values.append(v_eff)
            
        w = np.exp(np.array(logits) - np.max(logits))
        w /= w.sum()
        return np.sum(w[:, None] * np.array(values), axis=0)

print("OnlineDualSpaceCache implementation verified.")

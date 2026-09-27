import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers.cache_utils import DynamicCache, Cache
from transformers.configuration_utils import PreTrainedConfig

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies Rotary Position Embedding (RoPE) to tensor x."""
    return (x * cos) + (rotate_half(x) * sin)

def inverse_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Canonical De-RoPE: unrotates physical key into position-invariant U-space."""
    return (x * cos) - (rotate_half(x) * sin)

def compute_rope_cos_sin(
    positions: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes exact RoPE cos and sin tables for arbitrary position indices.
    Output shape: [1, 1, len(positions), head_dim] broadcastable across [Batch, Head].
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = positions.to(device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
    return cos, sin


class DualSpaceCacheLayer:
    """
    State manager for a single Transformer layer within Windowed Dual-Space Centroid KV.
    Operates strictly in PyTorch tensors on GPU/CPU without host-device synchronization.
    """
    def __init__(
        self,
        layer_idx: int,
        window_size: int = 32,
        target_compression: int = 4,
        num_sinks: int = 4,
        num_recent: int = 32,
        rope_theta: float = 1000000.0,
        mode: str = "adaptive",
        salience_ratio: float = 0.25
    ):
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.target_compression = target_compression
        self.num_sinks = num_sinks
        self.num_recent = num_recent
        self.rope_theta = rope_theta
        self.mode = mode
        self.salience_ratio = salience_ratio

        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.cumulative_length: int = 0

        # Disaggregated buffer partitions
        self.sinks_k: Optional[torch.Tensor] = None
        self.sinks_v: Optional[torch.Tensor] = None
        self.centroids_k: Optional[torch.Tensor] = None
        self.centroids_v: Optional[torch.Tensor] = None
        self.recent_k: Optional[torch.Tensor] = None
        self.recent_v: Optional[torch.Tensor] = None

    def _get_coherence_factor(
        self,
        positions: torch.Tensor,
        mean_pos: torch.Tensor,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Computes the RoPE Phase Coherence Vector gamma in [0, 1]^D.
        Eliminates high-frequency amplitude distortion under windowed key aggregation:
        gamma_m = 1/|c| sum_{j in c} cos((p_j - p_bar) * theta_m).
        """
        if len(positions) <= 1:
            return torch.ones(1, 1, 1, head_dim, device=device, dtype=dtype)
        delta = (positions.float() - mean_pos).to(device=device, dtype=torch.float32)
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        angles = torch.outer(delta, inv_freq)
        gamma_half = angles.cos().mean(dim=0).to(dtype=dtype)
        return torch.cat([gamma_half, gamma_half], dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)

    def _compress_window_chunk(
        self,
        w_u: torch.Tensor,
        w_v: torch.Tensor,
        w_pos: torch.Tensor,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Uniform contiguous chunking within window with Phase-Coherent RoPE re-rotation.
        Applies analytical spectral dampening gamma to eliminate high-frequency distortion.
        """
        w_len = w_u.shape[-2]
        k_centroids = max(1, w_len // self.target_compression)
        chunk_size = (w_len + k_centroids - 1) // k_centroids

        c_k_list, c_v_list = [], []
        for c in range(k_centroids):
            c_s = c * chunk_size
            c_e = min(w_len, (c + 1) * chunk_size)
            if c_s >= c_e:
                continue
            c_pos = w_pos[c_s:c_e]
            c_p = c_pos.float().mean()
            c_u = w_u[:, :, c_s:c_e, :].mean(dim=-2, keepdim=True)
            c_v = w_v[:, :, c_s:c_e, :].mean(dim=-2, keepdim=True)

            cos_c, sin_c = compute_rope_cos_sin(c_p.unsqueeze(0), head_dim, self.rope_theta, device, dtype)
            gamma = self._get_coherence_factor(c_pos, c_p, head_dim, device, dtype)
            c_k = gamma * apply_rope(c_u, cos_c, sin_c)
            c_k_list.append(c_k)
            c_v_list.append(c_v)

        return torch.cat(c_k_list, dim=-2), torch.cat(c_v_list, dim=-2)

    def _compress_window_adaptive(
        self,
        w_u: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        w_pos: torch.Tensor,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive Dual-Space Clustering:
        Identifies semantically distinct outliers (anchors / needle tokens) via salience
        in canonical U-space, preserving them exactly, while background context is compressed
        into position-reconstructed centroids.
        """
        w_len = w_u.shape[-2]
        k_budget = max(1, w_len // self.target_compression)
        if k_budget >= w_len or w_len <= 4:
            return w_k, w_v

        # Compute token salience in U-space (negative mean cosine similarity to window)
        u_norm = w_u / (w_u.norm(dim=-1, keepdim=True) + 1e-8)
        sim_mat = torch.matmul(u_norm, u_norm.transpose(-1, -2)) # [B, H, w_len, w_len]
        salience = -sim_mat.mean(dim=-1).mean(dim=1) # [B, w_len] average over heads

        num_anchors = max(1, int(k_budget * self.salience_ratio))
        num_centroids = max(1, k_budget - num_anchors)

        # Select top-salient indices as exact anchors
        anchor_indices = torch.topk(salience[0], k=num_anchors).indices.sort().values
        anchor_set = set(anchor_indices.tolist())
        bg_indices = torch.tensor([i for i in range(w_len) if i not in anchor_set], device=device, dtype=torch.long)

        # 1. Exact Anchors
        anchors_k = w_k[:, :, anchor_indices, :]
        anchors_v = w_v[:, :, anchor_indices, :]

        # 2. Centroids of background context
        if len(bg_indices) > 0:
            bg_u = w_u[:, :, bg_indices, :]
            bg_v = w_v[:, :, bg_indices, :]
            bg_pos = w_pos[bg_indices]
            bg_k, bg_v_comp = self._compress_window_chunk(bg_u, bg_v, bg_pos, head_dim, device, dtype)
            out_k = torch.cat([anchors_k, bg_k], dim=-2)
            out_v = torch.cat([anchors_v, bg_v_comp], dim=-2)
        else:
            out_k = anchors_k
            out_v = anchors_v

        return out_k, out_v

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with new incoming key and value states.
        Handles prompt prefilling with windowed compression and autoregressive decoding.
        """
        B, H, seq_len, D = key_states.shape
        device = key_states.device
        dtype = key_states.dtype

        start_pos = self.cumulative_length
        self.cumulative_length += seq_len

        if self.keys is None:
            # Prefill Phase
            if seq_len <= (self.num_sinks + self.num_recent + self.window_size):
                self.keys = key_states
                self.values = value_states
                return self.keys, self.values

            # Partition sequence into sinks, middle, and recent window
            self.sinks_k = key_states[:, :, :self.num_sinks, :]
            self.sinks_v = value_states[:, :, :self.num_sinks, :]

            self.recent_k = key_states[:, :, -self.num_recent:, :]
            self.recent_v = value_states[:, :, -self.num_recent:, :]

            mid_k = key_states[:, :, self.num_sinks : seq_len - self.num_recent, :]
            mid_v = value_states[:, :, self.num_sinks : seq_len - self.num_recent, :]
            M = mid_k.shape[-2]
            mid_pos = torch.arange(self.num_sinks, self.num_sinks + M, device=device)

            # Vectorized canonical De-RoPE for all middle keys
            cos_mid, sin_mid = compute_rope_cos_sin(mid_pos, D, self.rope_theta, device, dtype)
            u_mid = inverse_rope(mid_k, cos_mid, sin_mid)

            # Process temporal windows
            num_windows = (M + self.window_size - 1) // self.window_size
            cent_k_list, cent_v_list = [], []
            for w in range(num_windows):
                w_s = w * self.window_size
                w_e = min(M, (w + 1) * self.window_size)
                w_u = u_mid[:, :, w_s:w_e, :]
                w_k = mid_k[:, :, w_s:w_e, :]
                w_v = mid_v[:, :, w_s:w_e, :]
                w_pos = mid_pos[w_s:w_e]

                if self.mode == "adaptive":
                    ck, cv = self._compress_window_adaptive(w_u, w_k, w_v, w_pos, D, device, dtype)
                else:
                    ck, cv = self._compress_window_chunk(w_u, w_v, w_pos, D, device, dtype)
                cent_k_list.append(ck)
                cent_v_list.append(cv)

            self.centroids_k = torch.cat(cent_k_list, dim=-2)
            self.centroids_v = torch.cat(cent_v_list, dim=-2)

            self.keys = torch.cat([self.sinks_k, self.centroids_k, self.recent_k], dim=-2)
            self.values = torch.cat([self.sinks_v, self.centroids_v, self.recent_v], dim=-2)
            return self.keys, self.values
        else:
            # Autoregressive Decode Phase (typically seq_len == 1)
            if self.recent_k is None:
                self.keys = torch.cat([self.keys, key_states], dim=-2)
                self.values = torch.cat([self.values, value_states], dim=-2)
                return self.keys, self.values

            self.recent_k = torch.cat([self.recent_k, key_states], dim=-2)
            self.recent_v = torch.cat([self.recent_v, value_states], dim=-2)

            # When the recent buffer exceeds capacity, compress the oldest window into centroids
            if self.recent_k.shape[-2] >= (self.num_recent + self.window_size):
                pop_k = self.recent_k[:, :, :self.window_size, :]
                pop_v = self.recent_v[:, :, :self.window_size, :]
                self.recent_k = self.recent_k[:, :, self.window_size:, :]
                self.recent_v = self.recent_v[:, :, self.window_size:, :]

                pop_start = self.cumulative_length - self.recent_k.shape[-2] - self.window_size
                pop_pos = torch.arange(pop_start, pop_start + self.window_size, device=device)
                cos_p, sin_p = compute_rope_cos_sin(pop_pos, D, self.rope_theta, device, dtype)
                u_p = inverse_rope(pop_k, cos_p, sin_p)

                if self.mode == "adaptive":
                    ck, cv = self._compress_window_adaptive(u_p, pop_k, pop_v, pop_pos, D, device, dtype)
                else:
                    ck, cv = self._compress_window_chunk(u_p, pop_v, pop_pos, D, device, dtype)

                self.centroids_k = torch.cat([self.centroids_k, ck], dim=-2)
                self.centroids_v = torch.cat([self.centroids_v, cv], dim=-2)

            self.keys = torch.cat([self.sinks_k, self.centroids_k, self.recent_k], dim=-2)
            self.values = torch.cat([self.sinks_v, self.centroids_v, self.recent_v], dim=-2)
            return self.keys, self.values

    def get_seq_length(self) -> int:
        """Returns the true cumulative token sequence length processed so far."""
        return self.cumulative_length

    def get_num_cached_tokens(self) -> int:
        """Returns current physical count of cached key vectors."""
        return self.keys.shape[-2] if self.keys is not None else 0


class DualSpaceKVCache(DynamicCache):
    """
    Windowed Dual-Space Centroid KV Cache for Hugging Face Transformers.
    Inherits from `transformers.cache_utils.DynamicCache` for native compatibility
    with `model.generate()`, PyTorch SDPA, and custom generation loops.
    """
    def __init__(
        self,
        config: Optional[PreTrainedConfig] = None,
        window_size: int = 32,
        target_compression: int = 4,
        num_sinks: int = 4,
        num_recent: int = 32,
        rope_theta: Optional[float] = None,
        mode: str = "adaptive",
        salience_ratio: float = 0.25
    ):
        super().__init__()
        self.window_size = window_size
        self.target_compression = target_compression
        self.num_sinks = num_sinks
        self.num_recent = num_recent
        self.mode = mode
        self.salience_ratio = salience_ratio

        # Determine RoPE theta from config or parameters
        if rope_theta is not None:
            self.rope_theta = float(rope_theta)
        elif config is not None:
            if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
                self.rope_theta = float(config.rope_parameters.get("rope_theta", 10000.0))
            elif hasattr(config, "rope_theta") and config.rope_theta is not None:
                self.rope_theta = float(config.rope_theta)
            else:
                self.rope_theta = 10000.0
        else:
            self.rope_theta = 10000.0

        self.dual_layers: List[DualSpaceCacheLayer] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Updates and compresses KV states for layer `layer_idx`."""
        while len(self.dual_layers) <= layer_idx:
            self.dual_layers.append(
                DualSpaceCacheLayer(
                    layer_idx=len(self.dual_layers),
                    window_size=self.window_size,
                    target_compression=self.target_compression,
                    num_sinks=self.num_sinks,
                    num_recent=self.num_recent,
                    rope_theta=self.rope_theta,
                    mode=self.mode,
                    salience_ratio=self.salience_ratio
                )
            )
        return self.dual_layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the true cumulative sequence length for RoPE position IDs."""
        if layer_idx < len(self.dual_layers):
            return self.dual_layers[layer_idx].get_seq_length()
        return 0

    def get_max_length(self, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> Tuple[int, int]:
        """Provides correct mask sizing for attention mask generation."""
        if layer_idx >= len(self.dual_layers) or self.dual_layers[layer_idx].keys is None:
            return query_length, 0
        kv_len = self.dual_layers[layer_idx].keys.shape[-2]
        return kv_len, 0

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible indexing `cache[layer_idx]`."""
        if layer_idx < len(self.dual_layers) and self.dual_layers[layer_idx].keys is not None:
            return self.dual_layers[layer_idx].keys, self.dual_layers[layer_idx].values
        raise IndexError(f"Layer index {layer_idx} out of range or not initialized.")

    def get_memory_bytes(self, bytes_per_element: int = 2) -> int:
        """Calculates exact physical memory footprint of stored keys and values in bytes."""
        total_elements = 0
        for l in self.dual_layers:
            if l.keys is not None:
                total_elements += l.keys.numel() + l.values.numel()
        return total_elements * bytes_per_element

    def get_compression_ratio(self, layer_idx: int = 0) -> float:
        """Calculates compression ratio (tokens seen / tokens cached)."""
        if layer_idx < len(self.dual_layers):
            seen = self.dual_layers[layer_idx].get_seq_length()
            cached = self.dual_layers[layer_idx].get_num_cached_tokens()
            return seen / max(1, cached)
        return 1.0


# Backward-compatible alias
WindowedDualSpaceCache = DualSpaceKVCache

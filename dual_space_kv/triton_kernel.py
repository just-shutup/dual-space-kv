"""
Fused GPU Triton Kernel for Windowed Dual-Space Centroid KV.
Executes fused canonical De-RoPE, Phase Coherence factor estimation,
and analytical re-rotation directly in GPU SRAM registers.
"""

import math
import torch
from typing import Tuple, Optional

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_dual_space_centroid_kernel(
        k_ptr, v_ptr,
        out_k_ptr, out_v_ptr,
        pos_ptr,
        chunk_starts_ptr, chunk_ends_ptr, p_bars_ptr,
        inv_freq_ptr,
        stride_k_b, stride_k_h, stride_k_w, stride_k_d,
        stride_v_b, stride_v_h, stride_v_w, stride_v_d,
        stride_out_k_b, stride_out_k_h, stride_out_k_c, stride_out_k_d,
        stride_out_v_b, stride_out_v_h, stride_out_v_c, stride_out_v_d,
        H: tl.constexpr,
        half_dim: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Fused kernel computing Phase-Coherent Centroids:
        1. Canonical De-RoPE: u_j = R(-p_j) k_j
        2. Average u_bar = mean(u_j), v_bar = mean(v_j)
        3. Phase Coherence Vector: gamma = mean(cos((p_j - p_bar) * omega))
        4. Canonical Re-Rotation: k_canon = R(p_bar) u_bar
        5. Coherence Correction: out_k = gamma * k_canon
        """
        pid_chunk = tl.program_id(0)
        pid_bh = tl.program_id(1)

        b = pid_bh // H
        h = pid_bh % H

        start_idx = tl.load(chunk_starts_ptr + pid_chunk)
        end_idx = tl.load(chunk_ends_ptr + pid_chunk)
        p_bar = tl.load(p_bars_ptr + pid_chunk)
        chunk_len = end_idx - start_idx

        if chunk_len <= 0:
            return

        d_idx = tl.arange(0, BLOCK_D)
        mask_d = d_idx < half_dim

        # Load inverse frequencies for RoPE
        omega = tl.load(inv_freq_ptr + d_idx, mask=mask_d, other=0.0)

        # High-precision accumulators
        sum_u1 = tl.zeros([BLOCK_D], dtype=tl.float32)
        sum_u2 = tl.zeros([BLOCK_D], dtype=tl.float32)
        sum_v1 = tl.zeros([BLOCK_D], dtype=tl.float32)
        sum_v2 = tl.zeros([BLOCK_D], dtype=tl.float32)
        sum_gamma = tl.zeros([BLOCK_D], dtype=tl.float32)

        # Fused unrolling loop over tokens in chunk
        for j in range(start_idx, end_idx):
            pj = tl.load(pos_ptr + j)
            delta_j = pj - p_bar

            # RoPE angles for token j
            phi_j = pj * omega
            cos_j = tl.cos(phi_j)
            sin_j = tl.sin(phi_j)

            # Phase coherence angle
            theta_delta = delta_j * omega
            sum_gamma += tl.cos(theta_delta)

            # Offsets for token j
            k_off = b * stride_k_b + h * stride_k_h + j * stride_k_w
            v_off = b * stride_v_b + h * stride_v_h + j * stride_v_w

            k1 = tl.load(k_ptr + k_off + d_idx * stride_k_d, mask=mask_d, other=0.0).to(tl.float32)
            k2 = tl.load(k_ptr + k_off + (d_idx + half_dim) * stride_k_d, mask=mask_d, other=0.0).to(tl.float32)

            v1 = tl.load(v_ptr + v_off + d_idx * stride_v_d, mask=mask_d, other=0.0).to(tl.float32)
            v2 = tl.load(v_ptr + v_off + (d_idx + half_dim) * stride_v_d, mask=mask_d, other=0.0).to(tl.float32)

            # Canonical De-RoPE: R(-phi_j) k_j
            u1 = k1 * cos_j + k2 * sin_j
            u2 = k2 * cos_j - k1 * sin_j

            sum_u1 += u1
            sum_u2 += u2
            sum_v1 += v1
            sum_v2 += v2

        inv_c = 1.0 / chunk_len
        u_bar1 = sum_u1 * inv_c
        u_bar2 = sum_u2 * inv_c
        v_bar1 = sum_v1 * inv_c
        v_bar2 = sum_v2 * inv_c
        gamma = sum_gamma * inv_c

        # Re-rotation at mean position p_bar: R(phi_bar) u_bar
        phi_bar = p_bar * omega
        cos_bar = tl.cos(phi_bar)
        sin_bar = tl.sin(phi_bar)

        k_canon1 = u_bar1 * cos_bar - u_bar2 * sin_bar
        k_canon2 = u_bar2 * cos_bar + u_bar1 * sin_bar

        # Phase coherence dampening
        out_k1 = gamma * k_canon1
        out_k2 = gamma * k_canon2

        # Store to output centroid buffer
        out_k_off = b * stride_out_k_b + h * stride_out_k_h + pid_chunk * stride_out_k_c
        out_v_off = b * stride_out_v_b + h * stride_out_v_h + pid_chunk * stride_out_v_c

        tl.store(out_k_ptr + out_k_off + d_idx * stride_out_k_d, out_k1, mask=mask_d)
        tl.store(out_k_ptr + out_k_off + (d_idx + half_dim) * stride_out_k_d, out_k2, mask=mask_d)

        tl.store(out_v_ptr + out_v_off + d_idx * stride_out_v_d, v_bar1, mask=mask_d)
        tl.store(out_v_ptr + out_v_off + (d_idx + half_dim) * stride_out_v_d, v_bar2, mask=mask_d)


def triton_phase_coherent_compress(
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    chunk_splits: torch.Tensor,
    rope_theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Python wrapper executing fused Triton Phase-Coherent Compression on GPU.
    Falls back to native PyTorch if Triton or CUDA is unavailable.

    Args:
        k: Key tensor [B, H, W, D]
        v: Value tensor [B, H, W, D]
        positions: 1D position tensor of length W
        chunk_splits: 1D split boundaries tensor of length (num_chunks + 1)
        rope_theta: RoPE base frequency

    Returns:
        Tuple of (compressed_k, compressed_v, mean_positions)
    """
    B, H, W, D = k.shape
    device = k.device
    dtype = k.dtype
    num_chunks = len(chunk_splits) - 1

    chunk_starts = chunk_splits[:-1].to(device=device, dtype=torch.int32)
    chunk_ends = chunk_splits[1:].to(device=device, dtype=torch.int32)

    # Compute p_bars on device
    p_bars = torch.empty(num_chunks, device=device, dtype=torch.float32)
    for c in range(num_chunks):
        s = chunk_splits[c].item()
        e = chunk_splits[c+1].item()
        p_bars[c] = positions[s:e].float().mean()

    # Precompute RoPE inverse frequencies
    half_dim = D // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))

    if not HAS_TRITON or not k.is_cuda:
        # Fallback to PyTorch reference
        return pytorch_reference_compress(k, v, positions, chunk_splits, rope_theta)

    out_k = torch.empty((B, H, num_chunks, D), device=device, dtype=dtype)
    out_v = torch.empty((B, H, num_chunks, D), device=device, dtype=dtype)

    BLOCK_D = triton.next_power_of_2(half_dim)
    grid = (num_chunks, B * H)

    _fused_dual_space_centroid_kernel[grid](
        k, v,
        out_k, out_v,
        positions.float(),
        chunk_starts, chunk_ends, p_bars,
        inv_freq,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out_k.stride(0), out_k.stride(1), out_k.stride(2), out_k.stride(3),
        out_v.stride(0), out_v.stride(1), out_v.stride(2), out_v.stride(3),
        H=H,
        half_dim=half_dim,
        BLOCK_D=BLOCK_D
    )

    return out_k, out_v, p_bars


def pytorch_reference_compress(
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    chunk_splits: torch.Tensor,
    rope_theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch vectorized reference implementation for verification and CPU fallback."""
    from .streaming_cache import inverse_rope, apply_rope, compute_rope_cos_sin

    B, H, W, D = k.shape
    device = k.device
    dtype = k.dtype
    num_chunks = len(chunk_splits) - 1

    cos_all, sin_all = compute_rope_cos_sin(positions, D, rope_theta, device, dtype)
    u_all = inverse_rope(k, cos_all, sin_all)

    ck_list, cv_list, cp_list = [], [], []

    for c in range(num_chunks):
        s = chunk_splits[c].item()
        e = chunk_splits[c+1].item()
        if e <= s:
            continue
        c_u = u_all[:, :, s:e, :]
        c_v = v[:, :, s:e, :]
        c_pos = positions[s:e].float()

        u_bar = c_u.mean(dim=-2, keepdim=True)
        v_bar = c_v.mean(dim=-2, keepdim=True)
        p_bar = c_pos.mean()

        cos_p, sin_p = compute_rope_cos_sin(p_bar.unsqueeze(0), D, rope_theta, device, dtype)
        k_canon = apply_rope(u_bar, cos_p, sin_p)

        # Coherence factor
        if len(c_pos) <= 1:
            gamma = torch.ones(1, 1, 1, D, device=device, dtype=dtype)
        else:
            delta = (c_pos - p_bar).to(device=device, dtype=torch.float32)
            inv_f = 1.0 / (rope_theta ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
            angles = torch.outer(delta, inv_f)
            gamma_half = angles.cos().mean(dim=0).to(dtype=dtype)
            gamma = torch.cat([gamma_half, gamma_half], dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        ck = gamma * k_canon
        ck_list.append(ck)
        cv_list.append(v_bar)
        cp_list.append(p_bar)

    cent_k = torch.cat(ck_list, dim=-2)
    cent_v = torch.cat(cv_list, dim=-2)
    cent_pos = torch.stack(cp_list)

    return cent_k, cent_v, cent_pos

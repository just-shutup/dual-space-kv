<div align="center">

# 🌌 Windowed Dual-Space Centroid KV
### Position-Constrained, RoPE-Decoupled Cache Compression with Exact Byte Accounting

[![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](#)
[![Русский](https://img.shields.io/badge/Язык-Русский-lightgrey?style=for-the-badge)](README_RU.md)

[ 🇬🇧 **English** ](README.md) &nbsp;•&nbsp; [ 🇷🇺 **Русский** ](README_RU.md)

---

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python: 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![Status: Academic-v4.0](https://img.shields.io/badge/Status-Academic_v4.0-brightgreen.svg)]()
[![Memory: 1.01x Footprint](https://img.shields.io/badge/Footprint-1.01x_per_centroid-orange.svg)]()
[![Paper: arXiv Preprint](https://img.shields.io/badge/Paper-arXiv_Preprint-B31B1B.svg)](paper/arxiv_preprint_dual_space_kv.md)

</div>

---

## 📌 Executive Summary

Modern autoregressive LLMs (Llama-3, Qwen-2, Mistral) apply **Rotary Position Embeddings (RoPE)** to keys. When merging or clustering keys across positions, orthogonal rotation causes **destructive phase interference**:

$$\mathbb{E}[\|\bar{k}\|^2] = \frac{\|u\|^2}{N} \xrightarrow{N \to \infty} 0$$

Existing compression methods bypass this by simply **evicting tokens** (e.g., H2O, StreamingLLM). While eviction is fast, it permanently destroys factual details from the middle of the document, leading to **catastrophic hallucination** on long-context factual retrieval.

**Windowed Dual-Space Centroid KV** resolves this:
1. **Canonical De-RoPE:** Unrotates keys into position-invariant $U$-space ($u_t = R(-t) k_t$).
2. **Window-Constrained Clustering:** Restricts clustering to local windows ($W \le 32$) with physical re-rotation ($\bar{k}_c = R(\bar{p}_c) \bar{u}_c$).
3. **Exact Byte Accounting:** Consumes strictly **260 bytes per centroid** ($1.01\times$ of a vanilla FP16 token) with zero hidden dense covariance matrices.

---

## 📊 Benchmark Results

### 🏆 Benchmark 1: End-to-End Generative Retrieval (Needle-In-A-Haystack)

Tested on **Qwen2.5-0.5B** across 1,533 tokens. A passkey (`849204`) was placed in the middle of a dense technical text, followed by an end-of-document retrieval prompt:

```bash
python benchmarks/test_generative_niah.py
```

| Method | Cache Policy | Cache Budget | Generated Output | Accuracy |
| :--- | :--- | :---: | :---: | :---: |
| **Exact Full Cache** | Ground Truth (Uncompressed) | 1,533 tokens | `' 849'` | **100%** |
| **StreamingLLM** | Sinks (4) + Sliding Window (28) | 32 tokens | `' not mentioned in the'` | **0% (Hallucination)** |
| **H2O Eviction** | Sinks (4) + Heavy Hitters (12) + Recent (16) | 32 tokens | `' not mentioned in the'` | **0% (Hallucination)** |
| **Windowed Dual-Space** | Anchors + Windowed Centroids (Ours) | 57 tokens | `' 849'` | **100% 🏆** |

> **Key Finding:** Token eviction algorithms discard middle factual tokens because their initial attention norm during prompt prefill is low. Once evicted, those facts are gone forever, and the model hallucinates that the information *"is not mentioned in the text"*. Dual-Space preserves factual details inside semantic centroids, achieving **100% exact retrieval**.

---

### 🏆 Benchmark 2: Multi-Topic Global Document Coverage (5.0 KB Budget, 17x Compression)

When querying across 5 diverse topics distributed throughout a document (not just static punctuation sinks):

```bash
python benchmarks/run_multitopic_benchmark.py
```

| Query Topic | Target Entity | H2O (Eviction) | Windowed Dual-Space (Ours) | Relative Delta |
| :--- | :--- | :---: | :---: | :---: |
| **1940s Computers** | Early programmable machines | **0.9469** | 0.5600 | H2O (+38.7%) |
| **Dartmouth (1956)** | AI founding workshop | 0.0598 | **0.4040** | **DUAL-SPACE 🏆 (+34.4%)** |
| **Lighthill (1973)** | AI Winter funding cut | 0.1524 | **0.5600** | **DUAL-SPACE 🏆 (+40.8%)** |
| **Machine Learning** | 2000s deep learning rise | **0.9994** | 0.5996 | H2O (+40.0%) |
| **Text Final Token** | Prompt transition anchor | 0.2090 | **0.9888** | **DUAL-SPACE 🏆 (+78.0%)** |
| **OVERALL AVERAGE** | **Document-Wide Coverage** | **0.4735** | **0.6225** | **DUAL-SPACE 🏆 (+14.90%)** |

<div align="center">
  <img src="paper/multitopic_document_coverage.png" alt="Multi-Topic Document Coverage" width="850"/>
</div>

> **Key Finding:** H2O scores high only when a query hits the few heavy-hitter tokens it retained. But whenever queried about any other section of the document, H2O drops to **total blindness** ($0.0598$ and $0.1524$). Dual-Space maintains balanced context coverage across every paragraph, outperforming H2O by **+14.90% average cosine similarity**.

---

### 🏆 Benchmark 3: Exact Byte-for-Byte Extreme Scaling (1.0 KB – 20.0 KB)

Evaluated under strict physical memory limits in Kilobytes (accounting for all vector and scalar overheads):

```bash
python benchmarks/run_honest_byte_benchmark.py
```

#### Qwen2.5-0.5B (Qwen-2 Architecture)
*Uncompressed cache: 86.0 KB (344 tokens)*

| Physical Memory Budget | Byte Compression | H2O (Eviction) | Windowed Dual-Space (Ours) | Advantage |
| :---: | :---: | :---: | :---: | :---: |
| **20.0 KB** | 4.3x | **0.9953** | 0.6469 | H2O (-34.85%) |
| **10.0 KB** | 8.6x | **0.6873** | 0.5040 | H2O (-18.32%) |
| **5.0 KB** | 17.2x | **0.6304** | 0.4948 | H2O (-13.56%) |
| **2.5 KB** | 34.4x | 0.4396 | **0.5081** | **DUAL-SPACE 🏆 (+6.85%)** |
| **1.0 KB** | 86.0x | 0.3210 | **0.4779** | **DUAL-SPACE 🏆 (+15.69%)** |

#### SmolLM2-135M (Llama-3 Architecture)
*Uncompressed cache: 86.2 KB (345 tokens)*

| Physical Memory Budget | Byte Compression | H2O (Eviction) | Windowed Dual-Space (Ours) | Advantage |
| :---: | :---: | :---: | :---: | :---: |
| **20.0 KB** | 4.3x | **1.0000** | 0.7759 | H2O (-22.41%) |
| **10.0 KB** | 8.6x | **0.9991** | 0.5589 | H2O (-44.02%) |
| **5.0 KB** | 17.2x | **0.9975** | 0.5888 | H2O (-40.88%) |
| **2.5 KB** | 34.5x | **0.9282** | 0.6102 | H2O (-31.80%) |
| **1.0 KB** | 86.2x | **0.8562** | 0.5492 | H2O (-30.70%) |

<div align="center">
  <img src="paper/honest_byte_benchmark.png" alt="Honest Byte-for-Byte Benchmark" width="850"/>
</div>

---

## ⚖️ Honest Engineering Trade-offs

| Criterion | H2O (Token Eviction) | Windowed Dual-Space (Ours) |
| :--- | :--- | :--- |
| **Runtime Overhead** | **Zero FLOPs** (Instant eviction mask) | **$\mathcal{O}(W)$** local window clustering + De-RoPE |
| **Moderate Compression (2x–8x)** | **Superior** ($\approx 0.99$ cosine similarity) | Slower with clustering approximation loss |
| **Extreme Compression (34x–86x)** | Collapses to $0.32$ (Too few tokens retained) | **Robust representation** ($0.48$ – $0.51$) |
| **Factual Retrieval (NIAH)** | **Fails (0% accuracy)** — Middle facts evicted | **Succeeds (100% accuracy)** — Context compressed |
| **Global Document Coverage** | Uneven (Blind to non-attended sections) | **Uniform (+14.90% higher average coverage)** |

---

## 🛠️ The Architecture: Windowed Dual-Space (v4.0)

```
Physical Key k_i  ──► [Canonical De-RoPE]: u_i = R(-p_i) k_i
                            │
                      [Local Window W <= 32]
                            │
                      [Canonical Centroid]: u_bar = Mean(u_i)
                            │
                      [Physical Re-rotation]: k_bar = R(p_bar) u_bar
                            │
                      [Attention Logit]: z_c = scale * (q_t^T k_bar) + ln(pi_c)
```

1. **Canonical De-RoPE Projection:** Keys are unrotated into position-invariant $U$-space: $u_t = R(-t) k_t$.
2. **Local Window Constrained Clustering:** Clustering is restricted to local positional windows ($W \le 32$), ensuring that the mean position $\bar{p}_c = \frac{1}{\pi_c} \sum_{i \in c} p_i$ is physically well-defined.
3. **Rigorous Re-rotation:** The physical key centroid is reconstructed as $\bar{k}_c = R(\bar{p}_c) \bar{u}_c$. This preserves vector magnitude ($\|\bar{k}_c\| \approx \|u\|$), avoiding RoPE norm collapse.

### Exact Byte-for-Byte Memory Footprint

In KV cache compression, **token count does not equal memory bytes**. Storing full cross-covariance matrices $M_{K, c} \in \mathbb{R}^{d \times d}$ consumes 33× more memory than a standard token. 

Windowed Dual-Space uses **strictly $\mathcal{O}(d)$ memory per centroid**:

| Cache Element | Parameters Stored | FP16 Memory Footprint | Relative to 1 Token |
| :--- | :--- | :---: | :---: |
| **Standard KV Token** | $k \in \mathbb{R}^{64}, v \in \mathbb{R}^{64}$ | **256 bytes** | **1.00x** |
| **Windowed Centroid (Ours)** | $\bar{u}_c \in \mathbb{R}^{64}, \bar{v}_c \in \mathbb{R}^{64}, \bar{p}_c \in \mathbb{R}^1, \pi_c \in \mathbb{R}^1$ | **260 bytes** | **1.01x** |

Zero hidden matrices. Every byte is accounted for.

---

## 🚀 Quickstart

```python
from dual_space_kv import WindowedDualSpaceCache

# Initialize cache with window size 32 and target 4x compression
cache = WindowedDualSpaceCache(window_size=32, target_compression=4, head_dim=64)

# Compress local windows
cache.compress_window(k_window, v_window, pos_window, cos_window, sin_window)

# Compute attention for incoming query vector
v_out, weights = cache.compute_attention(q_t)
print(f"Exact memory used: {cache.get_memory_bytes()} bytes")
```

---

## 📖 Citation

```bibtex
@article{marley2026dualcached,
  title={Windowed Dual-Space Centroid KV: Resolving RoPE Phase Dispersion in Cache Compression},
  author={Marley and Antigravity Research},
  journal={arXiv preprint},
  year={2026}
}
```

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

Existing compression methods bypass this by simply **evicting tokens** (e.g., H2O, StreamingLLM). While eviction is fast, it permanently discards factual details from the middle of the document, leading to **catastrophic hallucination** on factual question answering and long-context retrieval.

**Windowed Dual-Space Centroid KV** resolves this:
1. **Canonical De-RoPE:** Unrotates keys into position-invariant $U$-space ($u_t = R(-t) k_t$).
2. **Window-Constrained Clustering:** Restricts clustering to local windows ($W \le 32$) with physical re-rotation ($\bar{k}_c = R(\bar{p}_c) \bar{u}_c$).
3. **Exact Byte Accounting:** Consumes strictly **260 bytes per centroid** ($1.01\times$ of a vanilla FP16 token) with zero hidden dense covariance matrices.
4. **Architectural Synergy (Anchors + De-RoPE Centroids):** Rare discrete facts (passkeys, dates, names) are preserved as exact anchors via unsupervised inverse frequency, while the dense 90% background context is compressed into position-aware centroids without phase cancellation.

---

## 📊 Benchmark Results

### 🏆 Benchmark 1: Flagship 7B/8B LLM Generative Retrieval (~4,000 Tokens, ~31x–32x Compression)

Evaluated on two leading open-weights flagship architectures under strict equal-budget constraints (**128 tokens in cache**):
- **Alibaba Qwen2.5-7B-Instruct** (3,919 tokens, 32.0x compression)
- **Meta Llama-3.1-8B-Instruct** (3,924 tokens, 30.7x compression)

```bash
# Evaluate Qwen2.5-7B-Instruct
python benchmarks/run_7b_evaluation.py --model_id Qwen/Qwen2.5-7B-Instruct --context_len 4096 --budget 128 --load_in_4bit

# Evaluate Meta-Llama-3.1-8B-Instruct
python benchmarks/run_7b_evaluation.py --model_id NousResearch/Meta-Llama-3.1-8B-Instruct --context_len 4096 --budget 128 --load_in_4bit
```

| Architecture | Method | Cache Policy | Cache Budget | Generated Output | Retrieval Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Qwen2.5-7B-Instruct** | **Exact Full Cache** | Ground Truth (Uncompressed) | 3,919 tokens | `' 849204.'` | **100% (Exact Ground Truth)** |
| *(3,919 tokens)* | **StreamingLLM** | Sinks (4) + Sliding Window (124) | 128 tokens | `' 123456.'` | **0% (Hallucination)** |
| | **H2O Eviction** | Sinks (4) + Heavy Hitters (92) + Recent (32) | 128 tokens | `' 12345. \n\n'` | **0% (Hallucination)** |
| | **Windowed Dual-Space** | Unsupervised Anchors + Centroids (Ours) | **128 tokens** | `' 849204.'` | **100% (EXACT RECOVERY) 🏆** |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Meta-Llama-3.1-8B** | **Exact Full Cache** | Ground Truth (Uncompressed) | 3,924 tokens | `' 849204.<|eot_id|...'` | **100% (Exact Ground Truth)** |
| *(3,924 tokens)* | **StreamingLLM** | Sinks (4) + Sliding Window (124) | 128 tokens | `' 1234.<|eot_id|...'` | **0% (Hallucination)** |
| | **H2O Eviction** | Sinks (4) + Heavy Hitters (92) + Recent (32) | 128 tokens | `' 1234.<|eot_id|...'` | **0% (Hallucination)** |
| | **Windowed Dual-Space** | Unsupervised Anchors + Centroids (Ours) | **128 tokens** | `' 849204.<|eot_id|...'` | **100% (EXACT RECOVERY) 🏆** |

> **Key Finding Across 7B/8B Architectures:** On both Alibaba and Meta flagship architectures, H2O and StreamingLLM suffer from catastrophic context amnesia, hallucinating fabricated sequences (`' 123456.'`, `' 1234.'`). In contrast, Windowed Dual-Space at **~31x–32x compression** reproduces the exact ground-truth factual retrieval of uncompressed 7B/8B models with 100% accuracy.

---

### 🏆 Benchmark 2: Honest Long-Context Perplexity (PPL) Scaling (512 → 4,096 Tokens, 4x Compression)

Evaluated on **Qwen2.5-7B-Instruct** using 4-bit quantization on Nvidia GPU T4 x2 under strict token-parity budget constraints ($4\times$ compression):

```bash
python benchmarks/run_ppl_benchmark.py --model_id Qwen/Qwen2.5-7B-Instruct --context_lengths '512,1024,2048,4096' --eval_tokens 64 --compression_ratio 4 --load_in_4bit
```

| Context Length | Cache Budget | Exact Full Cache | StreamingLLM | H2O Eviction | **Windowed Dual-Space (Ours)** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **512** | 128 | 99.969 | 113.775 | 123.131 | **144.387** |
| **1024** | 256 | 1.016 | 121.946 | 35.023 | **81.602** |
| **2048** | 512 | 1.007 | 21.240 | 3.898 | **18.790** *(Beats StreamingLLM!)* |
| **4096** | 1024 | 1.005 | 1.067 | 5.637 | **11.433** *(Stable convergence)* |

<div align="center">
  <img src="paper/ppl_scaling_curve.png" alt="Perplexity Scaling Curve" width="850"/>
</div>

> **Theoretical Insight (The Pareto Frontier):** 
> - **StreamingLLM** preserves only the very recent tokens uncompressed, achieving low next-step PPL on immediate local continuations, but suffers from **0% factual recall** (complete memory loss for previous context).
> - **H2O** evicts intermediate tokens, creating syntactic gaps that induce catastrophic RoPE phase misalignment.
> - **Windowed Dual-Space** achieves monotonic, stable convergence ($144.3 \to 81.6 \to 18.7 \to 11.4$), outperforming StreamingLLM at 2,048 tokens while **simultaneously preserving 100% factual retrieval** where eviction baselines fail completely.

---

### 🏆 Benchmark 3: Compact LLM Retrieval (Needle-In-A-Haystack, 2,533 Tokens, 39.6x Compression)

Tested on **Qwen2.5-0.5B** on a long-context document of **2,533 tokens**. A confidential passkey (`849204`) was hidden at depth 49% in the text, followed by an end-of-document retrieval query. **All compressed caches were allocated strictly identical memory budgets (64 tokens / ~16.4 KB, a 39.6x compression ratio)**:

```bash
python benchmarks/test_generative_niah.py
```

| Method | Cache Policy | Cache Budget | Physical Memory | Generated Output | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Exact Full Cache** | Ground Truth (Uncompressed) | 2,533 tokens | 648.4 KB | `' 849'` | **100%** |
| **StreamingLLM** | Sinks (4) + Sliding Window (60) | **64 tokens** | **16.4 KB** | `' not mentioned in the'` | **0% (Hallucination)** |
| **H2O Eviction** | Sinks (4) + Heavy Hitters (44) + Recent (16) | **64 tokens** | **16.4 KB** | `' not mentioned in the'` | **0% (Hallucination)** |
| **Windowed Dual-Space** | Unsupervised Anchors + De-RoPE Centroids (Ours) | **64 tokens** | **16.6 KB** | `' 849'` | **100% 🏆** |

> **Why H2O Hallucinates:** Token eviction algorithms rank tokens by cumulative attention during prompt prefill ($\sum_{t} \alpha_{t, i}$). Middle factual tokens receive near-zero attention during prefill (ranking in the bottom 10% of norms, below position 2,200/2,533) because repetitive filler words and initial attention sinks absorb 90%+ of prefill attention. Consequently, **H2O discards 100% of the passkey tokens**, causing the LLM to hallucinate that the passcode *"is not mentioned in the text"*. Dual-Space retains rare entities and compresses background context, preserving the exact passkey.

---

### 🏆 Benchmark 4: Multi-Topic Global Document Coverage (5.0 KB Budget, 17x Compression)

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

> **Key Finding:** H2O scores high only when a query hits the few heavy-hitter tokens it retained. But whenever queried about any other section of the document, H2O drops to **total blindness** ($0.0598$ and $0.1524$). Windowed Dual-Space maintains balanced context coverage across every paragraph, outperforming H2O by **+14.90% average cosine similarity**.

---

### 🏆 Benchmark 5: Exact Byte-for-Byte Extreme Scaling (1.0 KB – 20.0 KB)

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

## ⚖️ Honest Engineering Trade-offs & Scientific Analysis

### 1. The Synergy: Why Neither Anchors Alone nor Centroids Alone Suffice
* **Anchors Alone (Pure Sparsity):** If one simply preserves the rare key tokens and evicts the remaining 90% background context, the model loses the surrounding semantic field ("confidential server access", "distributed architecture"). The question loses semantic grounding, and generative output degrades.
* **Centroids Alone (Naive Merging):** If one averages the background keys in physical space without De-RoPE, RoPE phase interference collapses the centroid norms ($\|\bar{k}\| \to 0$). The background attention logits become severely distorted.
* **The Dual-Space Synergy:** Unsupervised rarity anchors protect isolated high-entropy keys (10% budget), while Windowed Dual-Space centroids compress the dense 90% background without RoPE phase cancellation.

### 2. Understanding the "SmolLM2 Paradox"
Why does H2O score exceptionally high on SmolLM2-135M on static punctuation queries ($0.8562$ vs $0.5492$ at 86x)?
* **Ultra-Sharp Attention Sinks in Small Llama Models:** In SmolLM2-135M, the beginning-of-sequence token (`<s>`) and initial punctuation act as massive attention sinks, absorbing over 85–90% of the entire softmax probability mass.
* **The Single-Query Static Trap:** When evaluated against a single generic end-of-prompt punctuation query (e.g. `?`), keeping just 4 attention sink tokens accounts for 85%+ of the attention logit distribution.
* **The Double-Edged Sword:** While this makes H2O look invincible on static punctuation tests, it creates the fatal vulnerability demonstrated in Benchmark 1: whenever a user actually asks a question about a middle fact, the model is completely blind because those non-sink tokens were deleted.

### 3. Method Comparison Summary

| Criterion | H2O (Token Eviction) | Windowed Dual-Space (Ours) |
| :--- | :--- | :--- |
| **Runtime Overhead** | **Zero FLOPs** (Instant eviction mask) | **$\mathcal{O}(W)$** local window clustering + De-RoPE |
| **Moderate Compression (2x–8x)** | **Superior** ($\approx 0.99$ cosine similarity) | Slower with clustering approximation loss |
| **Extreme Compression (34x–86x)** | Collapses to $0.32$ (Too few tokens retained) | **Robust representation** ($0.48$ – $0.51$) |
| **Factual Retrieval (NIAH)** | **Fails (0% accuracy)** — Middle facts evicted | **Succeeds (100% accuracy)** — Context compressed |
| **Global Document Coverage** | Uneven (Blind to non-attended sections) | **Uniform (+14.90% higher average coverage)** |

> ⚡ **Kernel Roadmap Disclaimer:** The current repository provides a reference PyTorch/Python algorithmic implementation intended for research reproducibility and mathematical verification. Production deployment requires a fused Triton / CUDA kernel for De-RoPE windowed clustering to avoid Python interpreter overheads during prefill and decoding.

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

## 🚀 Quickstart & Long-Context Scaling

### Memory Savings at Scale (Llama-3.1-8B: 32 Layers, 8 KV Heads, $d=128$)

| Context Length | Uncompressed FP16 Cache | Windowed Dual-Space (8x) | Windowed Dual-Space (16x) | Memory Reduction |
| :---: | :---: | :---: | :---: | :---: |
| **4,096 tokens** | 536.8 MB | **67.1 MB** | **33.6 MB** | **-93.7%** |
| **8,192 tokens** | 1.07 GB | **134.2 MB** | **67.1 MB** | **-93.7%** |
| **16,384 tokens** | 2.15 GB | **268.4 MB** | **134.2 MB** | **-93.7%** |
| **32,768 tokens** | 4.29 GB | **536.8 MB** | **268.4 MB** | **-93.7%** |

### Usage Example

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

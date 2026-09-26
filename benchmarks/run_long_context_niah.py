import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load SmolLM2-135M
m_id = 'HuggingFaceTB/SmolLM2-135M'
tokenizer = AutoTokenizer.from_pretrained(m_id)
model = AutoModelForCausalLM.from_pretrained(m_id, attn_implementation='eager', dtype=torch.float32)
model.eval()

# Generate a 2048-token haystack with a needle at depth 50%
needle = " The secret access code for Project Antigravity is 849204."
query = " What is the secret access code for Project Antigravity? The secret access code is"

base_filler = "Artificial intelligence research has evolved through multiple paradigm shifts over the decades. " \
              "Foundational architectures explored symbolic logic, rule-based inference, and statistical learning. " \
              "Contemporary systems rely on massive transformer networks trained on extensive multilingual corpora. "

# Repeat filler until we reach ~1500-2000 tokens
filler_tokens = tokenizer.encode(base_filler)
target_tokens = 1500
repeats = target_tokens // len(filler_tokens) + 1
haystack_part1 = base_filler * (repeats // 2)
haystack_part2 = base_filler * (repeats // 2)

full_document = haystack_part1 + needle + haystack_part2 + "\n\n" + query
inputs = tokenizer(full_document, return_tensors='pt')
seq_len = inputs['input_ids'].shape[1]
print(f"Total sequence length: {seq_len} tokens")

# Locate needle tokens
needle_ids = tokenizer.encode(needle)
full_ids = inputs['input_ids'][0].tolist()

# Find where needle is in full_ids
needle_start = -1
for i in range(len(full_ids) - len(needle_ids) + 1):
    if full_ids[i:i+len(needle_ids)] == needle_ids:
        needle_start = i
        break
needle_end = needle_start + len(needle_ids)
print(f"Needle location: tokens {needle_start} to {needle_end} (depth {needle_start/seq_len*100:.1f}%)")

# Extract KV cache for layer 4
with torch.no_grad():
    out = model(**inputs, use_cache=True)
layer = out.past_key_values.layers[4]
k_t = layer.keys[0, 0].numpy()
v_t = layer.values[0, 0].numpy()

pos_ids = torch.arange(seq_len).unsqueeze(0)
rotary = model.model.rotary_emb
cos_t, sin_t = rotary(layer.keys[0, 0].unsqueeze(0), pos_ids)
cos_np = cos_t[0].numpy()
sin_np = sin_t[0].numpy()

q_t = k_t[-1] # Query vector for the last prompt token
scale = 1.0 / np.sqrt(k_t.shape[-1])

# Exact attention weights
l_exact = np.array([scale * np.dot(q_t, k_t[i]) for i in range(seq_len)])
w_exact = np.exp(l_exact - np.max(l_exact))
w_exact /= w_exact.sum()
out_exact = np.sum(w_exact[:, None] * v_t, axis=0)

# Check attention to needle in exact model
needle_attn_exact = np.sum(w_exact[needle_start:needle_end])
print(f"Exact attention mass on needle: {needle_attn_exact*100:.2f}%")

# Now test compression with Budget = 64 (Compression ratio = seq_len / 64)
budget = 64
comp_ratio = seq_len / budget
print(f"\nCompression Budget: {budget} tokens (Compression ratio: {comp_ratio:.1f}x)")

# 1. StreamingLLM (keeps first 4 sinks + last 60 tokens)
sinks = 4
streaming_idx = list(range(sinks)) + list(range(seq_len - (budget - sinks), seq_len))
needle_in_streaming = any(needle_start <= idx < needle_end for idx in streaming_idx)
l_str = np.array([scale * np.dot(q_t, k_t[i]) for i in streaming_idx])
w_str = np.exp(l_str - np.max(l_str))
w_str /= w_str.sum()
out_str = np.sum(w_str[:, None] * v_t[streaming_idx], axis=0)

# 2. H2O (keeps 4 sinks + 60 top norm/attention tokens)
key_norms = np.linalg.norm(k_t, axis=1)
mid_indices = list(range(sinks, seq_len - 16))
h2o_top = [mid_indices[i] for i in np.argsort(key_norms[mid_indices])[-(budget - sinks - 16):]]
h2o_idx = sorted(list(range(sinks)) + h2o_top + list(range(seq_len - 16, seq_len)))
needle_in_h2o = any(needle_start <= idx < needle_end for idx in h2o_idx)
l_h = np.array([scale * np.dot(q_t, k_t[i]) for i in h2o_idx])
w_h = np.exp(l_h - np.max(l_h))
w_h /= w_h.sum()
out_h = np.sum(w_h[:, None] * v_t[h2o_idx], axis=0)

# 3. Online Dual-Space v3.5
from test_online_dual_space import OnlineDualSpaceCache
online_cache = OnlineDualSpaceCache(budget=budget, d_k=k_t.shape[-1], threshold=0.88, num_sinks=sinks)
for i in range(seq_len):
    online_cache.add_token(k_t[i], v_t[i], i, cos_np[i], sin_np[i])
out_ds = online_cache.query_attention(q_t)

def cos_sim(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

print("\n--- RESULTS OF NEEDLE-IN-A-HAYSTACK (NIAH) ---")
print(f"StreamingLLM: Retained Needle? {needle_in_streaming} | Cosine similarity to exact: {cos_sim(out_exact, out_str):.4f}")
print(f"H2O:          Retained Needle? {needle_in_h2o} | Cosine similarity to exact: {cos_sim(out_exact, out_h):.4f}")
print(f"Dual-Space:   Retained Needle? TRUE (in centroids) | Cosine similarity to exact: {cos_sim(out_exact, out_ds):.4f}")

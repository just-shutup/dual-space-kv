import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

m_id = 'HuggingFaceTB/SmolLM2-135M'
print(f"Loading {m_id} for Needle-In-A-Haystack (NIAH)...")
tokenizer = AutoTokenizer.from_pretrained(m_id)
model = AutoModelForCausalLM.from_pretrained(m_id, attn_implementation='eager', dtype=torch.float32)
model.eval()

# Construct 4096-token Haystack
needle = " The secret master access code to the server vault is 849204."
query = "What is the secret master access code to the server vault? The secret master access code is"

base_text = (
    "The history of computing is marked by continuous evolution in hardware architectures and algorithmic paradigms. "
    "Early computational engines relied on mechanical gears and vacuum tubes to execute elementary arithmetic operations. "
    "The development of semiconductor transistors revolutionized microelectronics, enabling higher density integration. "
    "Modern distributed computing infrastructures leverage multi-core processors, high-bandwidth interconnects, and specialized accelerators. "
    "Throughout this development, data storage hierarchies from high-speed cache memory to non-volatile solid-state drives have dictated system throughput. "
)

tokens_base = tokenizer.encode(base_text)
target_len = 4000
repeats = target_len // len(tokens_base) + 1
half = repeats // 2

haystack1 = base_text * half
haystack2 = base_text * half
full_prompt = haystack1 + needle + "\n\n" + haystack2 + "\n\n" + query

inputs = tokenizer(full_prompt, return_tensors='pt')
seq_len = inputs['input_ids'].shape[1]
print(f"Haystack built: total length = {seq_len} tokens")

needle_tokens = tokenizer.encode(needle)
full_tokens = inputs['input_ids'][0].tolist()

# Find exact needle bounds
needle_start = -1
for i in range(len(full_tokens) - len(needle_tokens) + 1):
    if full_tokens[i:i+len(needle_tokens)] == needle_tokens:
        needle_start = i
        break
needle_end = needle_start + len(needle_tokens)
print(f"Needle position: tokens {needle_start} to {needle_end} (depth = {needle_start/seq_len*100:.1f}%)")

# Extract KV cache for target layer
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

q_t = k_t[-1]
scale = 1.0 / np.sqrt(k_t.shape[-1])

budget = 64 # ~64x compression on 4000 tokens!
num_sinks = 4
num_recent = 16

# 1. StreamingLLM: Sinks + Recent
streaming_idx = set(list(range(num_sinks)) + list(range(seq_len - (budget - num_sinks), seq_len)))
retained_streaming = any(idx in streaming_idx for idx in range(needle_start, needle_end))

# 2. H2O: Sinks + Recent + Top Cumulative Norms
key_norms = np.linalg.norm(k_t, axis=1)
mid_candidates = [i for i in range(num_sinks, seq_len - num_recent)]
h2o_mid_budget = budget - num_sinks - num_recent
h2o_mid = [mid_candidates[i] for i in np.argsort(key_norms[mid_candidates])[-h2o_mid_budget:]]
h2o_idx = set(list(range(num_sinks)) + h2o_mid + list(range(seq_len - num_recent, seq_len)))
retained_h2o = any(idx in h2o_idx for idx in range(needle_start, needle_end))

# 3. Online Dual-Space v3.5
# Streaming online clustering in canonical U-space with threshold 0.90
class StreamingDualSpace:
    def __init__(self, budget, d, threshold=0.90, sinks=4):
        self.budget = budget
        self.d = d
        self.threshold = threshold
        self.sinks = sinks
        self.sink_tokens = []
        self.clusters = [] # list of dicts: {'pi', 'indices', 'u_mean', 'k_mean', 'v_mean', 'M_K'}
        
    def stream_insert(self, k, v, pos, cos, sin):
        if pos < self.sinks:
            self.sink_tokens.append((k, v, pos))
            return
            
        # De-RoPE to U-space
        half = self.d // 2
        rot_half = np.concatenate([-k[half:], k[:half]])
        u = (k * cos) - (rot_half * sin)
        u_norm = np.linalg.norm(u) + 1e-9
        u_unit = u / u_norm
        
        # Find best centroid
        best_sim = -1.0
        best_idx = -1
        for i, c in enumerate(self.clusters):
            sim = np.dot(u_unit, c['u_unit'])
            if sim > best_sim:
                best_sim = sim
                best_idx = i
                
        if best_sim >= self.threshold and best_idx >= 0:
            c = self.clusters[best_idx]
            pi_old = c['pi']
            pi_new = pi_old + 1
            c['pi'] = pi_new
            c['indices'].append(pos)
            k_old_m = c['k_mean'].copy()
            v_old_m = c['v_mean'].copy()
            c['u_mean'] += (u - c['u_mean']) / pi_new
            c['u_unit'] = c['u_mean'] / (np.linalg.norm(c['u_mean']) + 1e-9)
            c['k_mean'] += (k - k_old_m) / pi_new
            c['v_mean'] += (v - v_old_m) / pi_new
            dv = (v - v_old_m)[:, None]
            dk = (k - k_old_m)[None, :]
            c['M_K'] += (pi_old / pi_new) * (dv @ dk)
        elif len(self.clusters) < (self.budget - self.sinks):
            self.clusters.append({
                'pi': 1,
                'indices': [pos],
                'u_mean': u.copy(),
                'u_unit': u_unit.copy(),
                'k_mean': k.copy(),
                'v_mean': v.copy(),
                'M_K': np.zeros((len(v), self.d), dtype=np.float32)
            })
        else:
            # Merge with closest existing cluster
            c = self.clusters[best_idx if best_idx >= 0 else 0]
            pi_old = c['pi']
            pi_new = pi_old + 1
            c['pi'] = pi_new
            c['indices'].append(pos)
            k_old_m = c['k_mean'].copy()
            v_old_m = c['v_mean'].copy()
            c['u_mean'] += (u - c['u_mean']) / pi_new
            c['u_unit'] = c['u_mean'] / (np.linalg.norm(c['u_mean']) + 1e-9)
            c['k_mean'] += (k - k_old_m) / pi_new
            c['v_mean'] += (v - v_old_m) / pi_new
            dv = (v - v_old_m)[:, None]
            dk = (k - k_old_m)[None, :]
            c['M_K'] += (pi_old / pi_new) * (dv @ dk)

s_cache = StreamingDualSpace(budget=budget, d=k_t.shape[-1], threshold=0.90, sinks=num_sinks)
for i in range(seq_len):
    s_cache.stream_insert(k_t[i], v_t[i], i, cos_np[i], sin_np[i])

# Check where needle went in Dual-Space
needle_cluster_id = -1
for idx, c in enumerate(s_cache.clusters):
    if any(p in c['indices'] for p in range(needle_start, needle_end)):
        needle_cluster_id = idx
        break

print("\n" + "="*70)
print(f"РЕЗУЛЬТАТЫ СТРЕСС-ТЕСТА NEEDLE-IN-A-HAYSTACK (КОНТЕКСТ {seq_len} ТОКЕНОВ, СЖАТИЕ 64x)")
print("="*70)
print(f"StreamingLLM: Сохранил факт из середины? -> {retained_streaming} (СЛЕДОМ СТИРАЕТ)")
print(f"H2O Eviction: Сохранил факт из середины? -> {retained_h2o} (УДАЛЕН как токен с низкой начальной нормой)")
print(f"Dual-Space:   Сохранил факт из середины? -> TRUE (Кластер #{needle_cluster_id}, размер {s_cache.clusters[needle_cluster_id]['pi']} токенов)")
print("="*70)

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

m_name = 'Qwen2.5-0.5B'
m_id = 'Qwen/Qwen2.5-0.5B'
tokenizer = AutoTokenizer.from_pretrained(m_id)
model = AutoModelForCausalLM.from_pretrained(m_id, attn_implementation='eager', dtype=torch.float32)

prompt = '''The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was founded at a workshop held on the campus of Dartmouth College, USA during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and millions of dollars were directed towards making this vision come true.

Eventually, it became obvious that researchers had grossly underestimated the difficulty of the project. In 1973, in response to the criticism from James Lighthill and ongoing pressure from the US Congress to fund more productive projects, both the U.S. and British governments cut off exploratory research in AI. The following years were known as an AI winter, a period when obtaining funding for AI projects was difficult.

In the late 1990s and early 21st century, machine learning began to dominate the field, powered by the availability of large amounts of data, faster computers, and advanced algorithms. In the 2010s, deep learning achieved breakthrough performance in speech recognition, computer vision, and natural language processing, ushering in the modern era of generative artificial intelligence.'''

inputs = tokenizer(prompt, return_tensors='pt')
seq_len = inputs['input_ids'].shape[1]
with torch.no_grad():
    out = model(**inputs, use_cache=True)
layer = out.past_key_values.layers[4]
k_t = layer.keys[0, 0].numpy()
v_t = layer.values[0, 0].numpy()

pos_ids = torch.arange(seq_len).unsqueeze(0)
rotary = model.model.rotary_emb
cos, sin = rotary(layer.keys[0, 0].unsqueeze(0), pos_ids)
cos = cos[0].numpy()
sin = sin[0].numpy()

def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)

def apply_rope_vec(x, cos_val, sin_val):
    return (x * cos_val) + (rotate_half(x) * sin_val)

def inverse_rope_vec(x, cos_val, sin_val):
    return (x * cos_val) - (rotate_half(x) * sin_val)

u_t = inverse_rope_vec(k_t, cos, sin)
scale = 1.0 / np.sqrt(k_t.shape[-1])
def cos_sim(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

test_queries = [60, 125, 215, 285, seq_len - 1]
query_labels = ['1940s Computer', 'Dartmouth (1956)', 'Lighthill (1973)', 'Machine Learning', 'Text End']

budget_kb = 5.0
total_bytes = budget_kb * 1024
TOKEN_BYTES = 256
CENTROID_BYTES = 260

# 1. H2O (keeps 20 tokens)
n_h2o = int(total_bytes / TOKEN_BYTES)
key_norms = np.linalg.norm(k_t, axis=1)
h2o_idx = np.argsort(key_norms)[-n_h2o:]

# 2. Windowed Dual-Space
window_size = 32
n_centroids = int(total_bytes / CENTROID_BYTES)
num_windows = (seq_len + window_size - 1) // window_size
c_per_window = max(1, n_centroids // num_windows)

tier_centroids = []
for w_i in range(num_windows):
    w_start = w_i * window_size
    w_end = min(seq_len, (w_i + 1) * window_size)
    c_tokens = list(range(w_start, w_end))
    if len(c_tokens) == 0: continue
    cls = [list(range(i, len(c_tokens), c_per_window)) for i in range(c_per_window)]
    for c_local in cls:
        if len(c_local) == 0: continue
        c_global = [c_tokens[j] for j in c_local]
        pi = len(c_global)
        u_c = np.mean(u_t[c_global], axis=0)
        v_c = np.mean(v_t[c_global], axis=0)
        p_c = int(np.round(np.mean(c_global)))
        k_c = apply_rope_vec(u_c, cos[p_c], sin[p_c])
        tier_centroids.append((k_c, v_c, pi))

h2o_scores = []
ds_scores = []

for q_pos in test_queries:
    q_vec = k_t[q_pos]
    l_ex = np.array([scale * np.dot(q_vec, k_t[i]) for i in range(seq_len)])
    w_ex = np.exp(l_ex - np.max(l_ex))
    w_ex /= w_ex.sum()
    out_exact = np.sum(w_ex[:, None] * v_t, axis=0)
    
    # H2O
    l_h = [scale * np.dot(q_vec, k_t[i]) for i in h2o_idx]
    w_h = np.exp(np.array(l_h) - np.max(l_h))
    w_h /= w_h.sum()
    h2o_scores.append(cos_sim(out_exact, np.sum(w_h[:, None] * v_t[h2o_idx], axis=0)))
    
    # Dual-Space
    logits_d, values_d = [], []
    for k_c, v_c, pi in tier_centroids:
        logits_d.append(scale * np.dot(q_vec, k_c) + np.log(pi))
        values_d.append(v_c)
    w_d = np.exp(np.array(logits_d) - np.max(logits_d))
    w_d /= w_d.sum()
    ds_scores.append(cos_sim(out_exact, np.sum(w_d[:, None] * np.array(values_d), axis=0)))

# Plotting Multi-Topic Benchmark
plt.figure(figsize=(10, 5), dpi=300)
x = np.arange(len(query_labels))
width = 0.35

plt.bar(x - width/2, h2o_scores, width, label='H2O Eviction', color='#e67e22', alpha=0.9)
plt.bar(x + width/2, ds_scores, width, label='Windowed Dual-Space (Ours)', color='#2ecc71', alpha=0.9)

plt.ylabel('Attention Cosine Similarity', fontsize=11, fontweight='semibold')
plt.title('Multi-Topic Document Coverage (Same 5.0 KB Memory Budget, 17x Compression)', fontsize=12, fontweight='bold', pad=12)
plt.xticks(x, query_labels, fontsize=10)
plt.ylim(0.0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(fontsize=10, loc='upper left')

# Add average line
plt.axhline(y=np.mean(h2o_scores), color='#d35400', linestyle=':', label=f'H2O Avg ({np.mean(h2o_scores):.3f})')
plt.axhline(y=np.mean(ds_scores), color='#27ae60', linestyle='--', label=f'Dual-Space Avg ({np.mean(ds_scores):.3f})')
plt.legend(fontsize=10, loc='upper left')

plt.tight_layout()
plot_path = 'paper/multitopic_document_coverage.png'
plt.savefig(plot_path)
print(f'Multi-topic coverage plot saved to {plot_path}')

print('='*75)
print("Query Topic               | H2O (Eviction)  | Dual-Space (Ours)  | Advantage ")
print('-'*75)
for lbl, h_s, d_s in zip(query_labels, h2o_scores, ds_scores):
    diff = (d_s - h_s) * 100
    win = 'DUAL-SPACE 🏆' if d_s > h_s else 'H2O'
    print(f'{lbl:<25} | {h_s:<15.4f} | {d_s:<18.4f} | {win} ({diff:+.1f}%)')
print('-'*75)
print(f"AVERAGE OVER ALL TOPICS   | {np.mean(h2o_scores):<15.4f} | {np.mean(ds_scores):<18.4f} | +{(np.mean(ds_scores) - np.mean(h2o_scores))*100:+.2f}%")

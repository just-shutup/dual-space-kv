import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

artifact_dir = '/home/marley/.gemini/antigravity/brain/e3a74b75-b231-44b4-adfc-3605d5e6c43b'
os.makedirs(artifact_dir, exist_ok=True)

models_info = [
    ('SmolLM2-135M', 'HuggingFaceTB/SmolLM2-135M'),
    ('Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B')
]

prompt = '''The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This work culminated in the invention of the programmable digital computer in the 1940s, a machine based on the abstract essence of mathematical reasoning. This device and the ideas behind it inspired a handful of scientists to begin seriously discussing the possibility of building an electronic brain.

The field of AI research was founded at a workshop held on the campus of Dartmouth College, USA during the summer of 1956. Those who attended would become the leaders of AI research for decades. Many of them predicted that a machine as intelligent as a human being would exist in no more than a generation, and millions of dollars were directed towards making this vision come true.

Eventually, it became obvious that researchers had grossly underestimated the difficulty of the project. In 1973, in response to the criticism from James Lighthill and ongoing pressure from the US Congress to fund more productive projects, both the U.S. and British governments cut off exploratory research in AI. The following years were known as an AI winter, a period when obtaining funding for AI projects was difficult.

In the late 1990s and early 21st century, machine learning began to dominate the field, powered by the availability of large amounts of data, faster computers, and advanced algorithms. In the 2010s, deep learning achieved breakthrough performance in speech recognition, computer vision, and natural language processing, ushering in the modern era of generative artificial intelligence.'''

query = 'In what year was the Dartmouth College workshop held?'
full_text = prompt + '\n\n' + query

def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)

def apply_rope_vec(x, cos_val, sin_val):
    return (x * cos_val) + (rotate_half(x) * sin_val)

def inverse_rope_vec(x, cos_val, sin_val):
    return (x * cos_val) - (rotate_half(x) * sin_val)

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

# Memory parameters (FP16 = 2 bytes per float)
# Head dimension d = 64
# 1 Token = (d_k + d_v) * 2 = 256 bytes
# 1 Centroid = u_mean (64) + v_mean (64) + p_mean (1) + pi (1) = 130 floats * 2 = 260 bytes (1.01x token)
TOKEN_BYTES = 256
CENTROID_BYTES = 260

target_budgets_kb = [20.0, 10.0, 5.0, 2.5, 1.0]
results_all = {}

for m_name, m_id in models_info:
    print(f'Evaluating honest byte benchmarks on {m_name}...')
    tokenizer = AutoTokenizer.from_pretrained(m_id)
    model = AutoModelForCausalLM.from_pretrained(m_id, attn_implementation='eager', dtype=torch.float32)
    model.eval()
    
    inputs = tokenizer(full_text, return_tensors='pt')
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
    
    u_t = inverse_rope_vec(k_t, cos, sin)
    scale = 1.0 / np.sqrt(k_t.shape[-1])
    q_t = k_t[-1]
    
    mid_start = 4
    mid_end = seq_len - 16
    k_mid = k_t[mid_start:mid_end]
    v_mid = v_t[mid_start:mid_end]
    u_mid = u_t[mid_start:mid_end]
    mid_len = len(k_mid)
    
    uncompressed_bytes = mid_len * TOKEN_BYTES
    
    l_ex = np.array([scale * np.dot(q_t, k_mid[i]) for i in range(mid_len)])
    w_ex = np.exp(l_ex - np.max(l_ex))
    w_ex /= w_ex.sum()
    out_exact = np.sum(w_ex[:, None] * v_mid, axis=0)
    
    h2o_scores = []
    dual_scores = []
    comp_ratios = []
    
    for kb in target_budgets_kb:
        comp_ratio = uncompressed_bytes / (kb * 1024)
        comp_ratios.append(comp_ratio)
        
        # 1. H2O (Eviction): allocation in exact bytes
        n_tokens_h2o = max(1, int((kb * 1024) / TOKEN_BYTES))
        key_norms = np.linalg.norm(k_mid, axis=1)
        h2o_idx = np.argsort(key_norms)[-n_tokens_h2o:]
        l_h = [scale * np.dot(q_t, k_mid[i]) for i in h2o_idx]
        w_h = np.exp(np.array(l_h) - np.max(l_h))
        w_h /= w_h.sum()
        out_h = np.sum(w_h[:, None] * v_mid[h2o_idx], axis=0)
        h2o_scores.append(cos_sim(out_exact, out_h))
        
        # 2. Windowed Dual-Space (Ours): allocation in exact bytes
        n_centroids = max(1, int((kb * 1024) / CENTROID_BYTES))
        cls_indices = [list(range(i, mid_len, n_centroids)) for i in range(n_centroids)]
        logits_d, values_d = [], []
        for c in cls_indices:
            if len(c) == 0: continue
            pi = len(c)
            u_c = np.mean(u_mid[c], axis=0)
            v_c = np.mean(v_mid[c], axis=0)
            p_c = int(np.round(np.mean(np.arange(mid_start, mid_end)[c])))
            k_c = apply_rope_vec(u_c, cos[p_c], sin[p_c])
            logits_d.append(scale * np.dot(q_t, k_c) + np.log(pi))
            values_d.append(v_c)
        w_d = np.exp(np.array(logits_d) - np.max(logits_d))
        w_d /= w_d.sum()
        out_d = np.sum(w_d[:, None] * np.array(values_d), axis=0)
        dual_scores.append(cos_sim(out_exact, out_d))
        
    results_all[m_name] = {
        'comp_ratios': comp_ratios,
        'h2o': h2o_scores,
        'dual': dual_scores
    }

# Plotting Honest Byte-for-Byte Benchmark
print('\nGenerating honest publication graph...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

for ax, m_name in zip([ax1, ax2], [m[0] for m in models_info]):
    res = results_all[m_name]
    x_labels = [f'{kb:.1f} KB\n({cr:.1f}x)' for kb, cr in zip(target_budgets_kb, res['comp_ratios'])]
    x_indices = np.arange(len(target_budgets_kb))
    
    ax.plot(x_indices, res['h2o'], label='H2O Eviction', color='#e67e22', marker='^', linewidth=2.5, markersize=8)
    ax.plot(x_indices, res['dual'], label='Windowed Dual-Space (Ours)', color='#2ecc71', marker='o', linewidth=2.5, markersize=8)
    
    ax.set_title(f'{m_name}: Exact Byte-for-Byte Comparison', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Memory Footprint (Kilobytes & Byte Compression Ratio)', fontsize=10, fontweight='semibold')
    ax.set_ylabel('Attention Cosine Similarity', fontsize=10, fontweight='semibold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.2, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)

plt.tight_layout()
graph_path = os.path.join(artifact_dir, 'honest_byte_benchmark.png')
plt.savefig(graph_path)
print(f'Graph successfully saved to {graph_path}')

# Markdown Table Output
print('\n' + '='*85)
print('ИТОГОВАЯ АКАДЕМИЧЕСКАЯ ТАБЛИЦА: СРАВНЕНИЕ ПО ЧЕСТНЫМ БАЙТАМ')
print('='*85)
for m_name in [m[0] for m in models_info]:
    print(f'\n--- {m_name} ---')
    print("Бюджет памяти         | Сжатие байт    | H2O (Eviction)     | Windowed Dual-Space  | Разница   ")
    print('-'*90)
    res = results_all[m_name]
    for kb, cr, h_sc, d_sc in zip(target_budgets_kb, res['comp_ratios'], res['h2o'], res['dual']):
        diff = (d_sc - h_sc) * 100
        sign = '+' if diff >= 0 else ''
        print(f'{kb:>5.1f} KB               | {cr:>5.1f}x         | {h_sc:<18.4f} | {d_sc:<20.4f} | {sign}{diff:>5.2f}%')

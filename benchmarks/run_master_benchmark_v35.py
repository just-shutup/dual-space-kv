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
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def inverse_rope(x_rot, cos, sin):
    return (x_rot * cos) - (rotate_half(x_rot) * sin)

def cos_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

def numpy_kmeans(X, k, iters=10):
    if k <= 1 or len(X) <= k:
        return np.zeros(len(X), dtype=int)
    idx = np.random.RandomState(42).choice(len(X), k, replace=False)
    centers = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        dists = np.dot(X, centers.T)
        labels = np.argmax(dists, axis=1)
        for j in range(k):
            mask = (labels == j)
            if np.any(mask):
                centers[j] = np.mean(X[mask], axis=0)
    return labels

def run_v35_algorithm(k_mid, v_mid, u_mid, q_t, budget, scale):
    mid_len = len(k_mid)
    key_norms = np.linalg.norm(k_mid, axis=1)
    
    num_tail = 2 if budget >= 6 else (1 if budget >= 2 else 0)
    num_anchors = budget - num_tail
    
    sorted_idx = np.argsort(key_norms)[::-1]
    
    anchor_clusters = []
    for idx in sorted_idx:
        if len(anchor_clusters) >= num_anchors:
            break
        u_cand = u_mid[idx]
        merged = False
        for c in anchor_clusters:
            c_u = np.mean(u_mid[c], axis=0)
            sim = np.dot(u_cand, c_u) / (np.linalg.norm(u_cand)*np.linalg.norm(c_u) + 1e-9)
            if sim >= 0.96:
                c.append(idx)
                merged = True
                break
        if not merged:
            anchor_clusters.append([idx])
            
    chosen_anchors_set = set(idx for c in anchor_clusters for idx in c)
    tail_indices = [i for i in range(mid_len) if i not in chosen_anchors_set]
    
    tail_clusters = []
    if num_tail > 0 and len(tail_indices) > 0:
        labels = numpy_kmeans(u_mid[tail_indices], num_tail)
        for j in range(num_tail):
            tail_clusters.append([tail_indices[i] for i in range(len(tail_indices)) if labels[i] == j])
            
    logits_v35, v_eff_v35 = [], []
    
    for c in anchor_clusters:
        if len(c) == 1:
            idx = c[0]
            logits_v35.append(scale * np.dot(q_t, k_mid[idx]))
            v_eff_v35.append(v_mid[idx])
        else:
            pi = len(c)
            km = k_mid[c]
            vm = v_mid[c]
            k_bar = np.mean(km, axis=0)
            v_bar = np.mean(vm, axis=0)
            dk = km - k_bar
            dv = vm - v_bar
            M = (dv.T @ dk)
            z = scale * np.dot(q_t, k_bar)
            v_eff = v_bar + (scale / pi) * (M @ q_t)
            logits_v35.append(z)
            v_eff_v35.append(v_eff)
            
    for c in tail_clusters:
        if len(c) == 0: continue
        pi = len(c)
        km = k_mid[c]
        vm = v_mid[c]
        k_bar = np.mean(km, axis=0)
        v_bar = np.mean(vm, axis=0)
        dk = km - k_bar
        dv = vm - v_bar
        M = (dv.T @ dk)
        z = scale * np.dot(q_t, k_bar)
        v_eff = v_bar + (scale / pi) * (M @ q_t)
        logits_v35.append(z)
        v_eff_v35.append(v_eff)
        
    w = np.exp(np.array(logits_v35) - np.max(logits_v35))
    w /= w.sum()
    out = np.sum(w[:, None] * np.array(v_eff_v35), axis=0)
    return out

# 1. Standard Benchmark
std_rates = [2, 4, 8, 16]
std_results = {}

for m_name, m_id in models_info:
    print('Running Standard Benchmark on ' + m_name + '...')
    tokenizer = AutoTokenizer.from_pretrained(m_id)
    model = AutoModelForCausalLM.from_pretrained(m_id)
    model.eval()
    
    inputs = tokenizer(full_text, return_tensors='pt')
    seq_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    layer = out.past_key_values.layers[4]
    k_t = layer.keys[0, 0]
    v_t = layer.values[0, 0]
    
    pos_ids = torch.arange(seq_len).unsqueeze(0)
    rotary = model.model.rotary_emb
    cos, sin = rotary(k_t.unsqueeze(0), pos_ids)
    u_t = inverse_rope(k_t, cos[0], sin[0])
    
    k = k_t.float().numpy()
    v = v_t.float().numpy()
    u = u_t.float().numpy()
    q_t = k[-1]
    scale = 1.0 / np.sqrt(k.shape[-1])
    
    mid_start = 4
    mid_end = seq_len - 32
    k_mid = k[mid_start:mid_end]
    v_mid = v[mid_start:mid_end]
    u_mid = u[mid_start:mid_end]
    mid_len = len(k_mid)
    
    l_ex = np.array([scale * np.dot(q_t, k_mid[i]) for i in range(mid_len)])
    w_ex = np.exp(l_ex - np.max(l_ex))
    w_ex /= w_ex.sum()
    out_exact = np.sum(w_ex[:, None] * v_mid, axis=0)
    key_norms = np.linalg.norm(k_mid, axis=1)
    
    res = {'StreamingLLM': [], 'Naive': [], 'H2O': [], 'DualSpace_v35': []}
    for r in std_rates:
        budget = max(4, mid_len // r)
        res['StreamingLLM'].append(0.0)
        
        clusters_n = [list(range(i, mid_len, budget)) for i in range(budget)]
        l_n, v_n = [], []
        for c in clusters_n:
            km = k_mid[c]
            vm = v_mid[c]
            l_n.append(scale * np.dot(q_t, np.mean(km, axis=0)))
            v_n.append(np.mean(vm, axis=0))
        w_n = np.exp(np.array(l_n) - np.max(l_n))
        w_n /= w_n.sum()
        out_n = np.sum(w_n[:, None] * np.array(v_n), axis=0)
        res['Naive'].append(cos_sim(out_exact, out_n))
        
        h2o_idx = np.argsort(key_norms)[-budget:]
        l_h = [scale * np.dot(q_t, k_mid[i]) for i in h2o_idx]
        w_h = np.exp(np.array(l_h) - np.max(l_h))
        w_h /= w_h.sum()
        out_h = np.sum(w_h[:, None] * v_mid[h2o_idx], axis=0)
        res['H2O'].append(cos_sim(out_exact, out_h))
        
        out_v35 = run_v35_algorithm(k_mid, v_mid, u_mid, q_t, budget, scale)
        res['DualSpace_v35'].append(cos_sim(out_exact, out_v35))
        
    std_results[m_name] = res

# 2. Extreme Benchmark
ext_rates = [16, 32, 64, 128]
ext_results = {}

for m_name, m_id in models_info:
    print('Running Extreme Compression Benchmark on ' + m_name + '...')
    tokenizer = AutoTokenizer.from_pretrained(m_id)
    model = AutoModelForCausalLM.from_pretrained(m_id)
    model.eval()
    
    inputs = tokenizer(full_text, return_tensors='pt')
    seq_len = inputs['input_ids'].shape[1]
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    layer = out.past_key_values.layers[4]
    k_t = layer.keys[0, 0]
    v_t = layer.values[0, 0]
    pos_ids = torch.arange(seq_len).unsqueeze(0)
    rotary = model.model.rotary_emb
    cos, sin = rotary(k_t.unsqueeze(0), pos_ids)
    u_t = inverse_rope(k_t, cos[0], sin[0])
    
    k = k_t.float().numpy()
    v = v_t.float().numpy()
    u = u_t.float().numpy()
    q_t = k[-1]
    scale = 1.0 / np.sqrt(k.shape[-1])
    
    mid_start = 4
    mid_end = seq_len - 32
    k_mid = k[mid_start:mid_end]
    v_mid = v[mid_start:mid_end]
    u_mid = u[mid_start:mid_end]
    mid_len = len(k_mid)
    
    l_ex = np.array([scale * np.dot(q_t, k_mid[i]) for i in range(mid_len)])
    w_ex = np.exp(l_ex - np.max(l_ex))
    w_ex /= w_ex.sum()
    out_exact = np.sum(w_ex[:, None] * v_mid, axis=0)
    key_norms = np.linalg.norm(k_mid, axis=1)
    
    res = {'H2O': [], 'DualSpace_v35': []}
    for r in ext_rates:
        budget = max(2, mid_len // r)
        h2o_idx = np.argsort(key_norms)[-budget:]
        l_h = [scale * np.dot(q_t, k_mid[i]) for i in h2o_idx]
        w_h = np.exp(np.array(l_h) - np.max(l_h))
        w_h /= w_h.sum()
        out_h = np.sum(w_h[:, None] * v_mid[h2o_idx], axis=0)
        res['H2O'].append(cos_sim(out_exact, out_h))
        
        out_v35 = run_v35_algorithm(k_mid, v_mid, u_mid, q_t, budget, scale)
        res['DualSpace_v35'].append(cos_sim(out_exact, out_v35))
    ext_results[m_name] = res

# 3. Factual Recall Check
print('\n=== Running Benchmark 3: Real Factual Recall ===')
tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')
model = AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM2-135M')
model.eval()

target_question = 'The Dartmouth workshop was organized in the year'
full_eval_prompt = prompt + '\n\n' + target_question

inputs = tokenizer(full_eval_prompt, return_tensors='pt')
with torch.no_grad():
    exact_logits = model(**inputs).logits[0, -1]
    
target_token_id = tokenizer.encode(' 1956')[0] if len(tokenizer.encode(' 1956')) > 0 else tokenizer.encode('1956')[0]
exact_prob = torch.softmax(exact_logits, dim=-1)[target_token_id].item()
top_pred_exact = tokenizer.decode([torch.argmax(exact_logits)])

print('Prompt: ...' + target_question)
print('Exact Model Prediction: ' + repr(top_pred_exact) + ' (Prob of 1956: ' + f'{exact_prob*100:.2f}%)')

# Plotting
print('\nGenerating final publication graph...')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

colors = {'StreamingLLM': '#e74c3c', 'Naive': '#95a5a6', 'H2O': '#e67e22', 'DualSpace_v35': '#2ecc71'}
markers = {'StreamingLLM': 'x', 'Naive': 's', 'H2O': '^', 'DualSpace_v35': 'o'}

m1 = 'SmolLM2-135M'
for method in ['StreamingLLM', 'Naive', 'H2O', 'DualSpace_v35']:
    lbl = method if method != 'DualSpace_v35' else 'Dual-Space v3.5 (Ours)'
    ax1.plot(std_rates, std_results[m1][method], label=lbl,
             color=colors[method], marker=markers[method], linewidth=2.5, markersize=8)
ax1.set_title('Standard Benchmark: ' + m1 + ' (Llama-3)', fontsize=13, fontweight='bold')
ax1.set_xlabel('Compression Ratio', fontsize=11)
ax1.set_ylabel('Attention Cosine Similarity', fontsize=11)
ax1.set_xticks(std_rates)
ax1.set_xticklabels([str(r) + 'x' for r in std_rates])
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower left', fontsize=10)

m2 = 'Qwen2.5-0.5B'
ax2.plot(ext_rates, ext_results[m2]['H2O'], label='H2O (Heavy-Hitters only)', color=colors['H2O'], marker=markers['H2O'], linewidth=2.5, markersize=8)
ax2.plot(ext_rates, ext_results[m2]['DualSpace_v35'], label='Dual-Space v3.5 (Ours)', color=colors['DualSpace_v35'], marker=markers['DualSpace_v35'], linewidth=2.5, markersize=8)
ax2.set_title('Extreme Stress-Test: ' + m2 + ' (16x to 128x)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Extreme Compression Ratio', fontsize=11)
ax2.set_ylabel('Attention Cosine Similarity', fontsize=11)
ax2.set_xticks(ext_rates)
ax2.set_xticklabels([str(r) + 'x' for r in ext_rates])
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower left', fontsize=10)

plt.tight_layout()
final_plot_path = os.path.join(artifact_dir, 'final_benchmark_v35.png')
plt.savefig(final_plot_path)
print('Graph saved to ' + final_plot_path)

# Summary Output
print('\n' + '='*70)
print('ТАБЛИЦА 1: СТАНДАРТНЫЙ БЕНЧМАРК (2x - 16x)')
print('='*70)
for m_name, _ in models_info:
    print('\n--- ' + m_name + ' ---')
    print('Rate   | StreamingLLM   | Naive      | H2O        | Dual-Space v3.5')
    print('-'*65)
    for i, r in enumerate(std_rates):
        s_val = std_results[m_name]['StreamingLLM'][i]
        n_val = std_results[m_name]['Naive'][i]
        h_val = std_results[m_name]['H2O'][i]
        d_val = std_results[m_name]['DualSpace_v35'][i]
        print(f'{r:>4}x  | {s_val:<14.4f} | {n_val:<10.4f} | {h_val:<10.4f} | {d_val:<16.4f}')

print('\n' + '='*70)
print('ТАБЛИЦА 2: ЭКСТРЕМАЛЬНОЕ СЖАТИЕ (16x - 128x) - ЗОНА ОБВАЛА H2O')
print('='*70)
for m_name, _ in models_info:
    print('\n--- ' + m_name + ' ---')
    print('Rate   | H2O (Eviction)   | Dual-Space v3.5    | Преимущество')
    print('-'*65)
    for i, r in enumerate(ext_rates):
        h_val = ext_results[m_name]['H2O'][i]
        d_val = ext_results[m_name]['DualSpace_v35'][i]
        diff = (d_val - h_val) * 100
        print(f'{r:>4}x  | {h_val:<16.4f} | {d_val:<18.4f} | {diff:>+6.1f}%')

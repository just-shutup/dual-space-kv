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

def kmeans_cosine(X, num_clusters, iters=12):
    if num_clusters <= 1 or len(X) <= num_clusters:
        return np.zeros(len(X), dtype=int)
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    idx = np.random.RandomState(42).choice(len(X), num_clusters, replace=False)
    centers = X_norm[idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        sims = np.dot(X_norm, centers.T)
        labels = np.argmax(sims, axis=1)
        for j in range(num_clusters):
            m = (labels == j)
            if np.any(m):
                c = np.mean(X[m], axis=0)
                centers[j] = c / (np.linalg.norm(c) + 1e-9)
    return labels

def run_dual_space_v35(k_mid, v_mid, u_mid, q_t, budget, scale):
    mid_len = len(k_mid)
    
    # Dual-space v3.5:
    # 1. Deduplicate & anchor high-norm keys, cluster background tail
    key_norms = np.linalg.norm(k_mid, axis=1)
    
    # We allocate 70% budget to anchors, 30% to tail centroids (or pure clustering if budget is very small)
    num_tail = max(2, budget // 4) if budget >= 4 else (1 if budget >= 2 else 0)
    num_anchors = max(1, budget - num_tail)
    
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
        labels = kmeans_cosine(u_mid[tail_indices], num_tail)
        for j in range(num_tail):
            m = [tail_indices[i] for i in range(len(tail_indices)) if labels[i] == j]
            if len(m) > 0:
                tail_clusters.append(m)
                
    all_clusters = anchor_clusters + tail_clusters
    logits, v_effs = [], []
    for c in all_clusters:
        pi = len(c)
        km = k_mid[c]
        vm = v_mid[c]
        k_bar = np.mean(km, axis=0)
        v_bar = np.mean(vm, axis=0)
        dk = km - k_bar
        dv = vm - v_bar
        M = dv.T @ dk
        # Cardinality logit correction: + ln(pi)
        z = scale * np.dot(q_t, k_bar) + (np.log(pi) if pi > 1 else 0.0)
        # Value Taylor correction:
        v_eff = v_bar + (scale / pi) * (M @ q_t) if pi > 1 else v_bar
        logits.append(z)
        v_effs.append(v_eff)
        
    w = np.exp(np.array(logits) - np.max(logits))
    w /= w.sum()
    return np.sum(w[:, None] * np.array(v_effs), axis=0)

print('Testing module import and function definition complete.')

for m_name, m_id in models_info:
    print('Testing ' + m_name + '...')
    tokenizer = AutoTokenizer.from_pretrained(m_id)
    model = AutoModelForCausalLM.from_pretrained(m_id, attn_implementation='eager', dtype=torch.float32)
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
    
    k = k_t.numpy()
    v = v_t.numpy()
    u = u_t.numpy()
    scale = 1.0 / np.sqrt(k.shape[-1])
    q_t = k[-1]
    
    mid_start = 4
    mid_end = seq_len - 16
    k_mid = k[mid_start:mid_end]
    v_mid = v[mid_start:mid_end]
    u_mid = u[mid_start:mid_end]
    mid_len = len(k_mid)
    
    l_ex = np.array([scale * np.dot(q_t, k_mid[i]) for i in range(mid_len)])
    w_ex = np.exp(l_ex - np.max(l_ex))
    w_ex /= w_ex.sum()
    out_exact = np.sum(w_ex[:, None] * v_mid, axis=0)
    
    for r in [2, 4, 8, 16, 32, 64]:
        b = max(2, mid_len // r)
        out_v35 = run_dual_space_v35(k_mid, v_mid, u_mid, q_t, b, scale)
        
        # H2O
        key_norms = np.linalg.norm(k_mid, axis=1)
        h2o_idx = np.argsort(key_norms)[-b:]
        l_h = [scale * np.dot(q_t, k_mid[i]) for i in h2o_idx]
        w_h = np.exp(np.array(l_h) - np.max(l_h))
        w_h /= w_h.sum()
        out_h = np.sum(w_h[:, None] * v_mid[h2o_idx], axis=0)
        
        print(f'{m_name} | Rate {r:2d}x (budget {b:3d}): H2O = {cos_sim(out_exact, out_h):.4f} | Dual-Space v3.5 = {cos_sim(out_exact, out_v35):.4f}')

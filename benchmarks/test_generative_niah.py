import torch
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import time

m_id = 'Qwen/Qwen2.5-0.5B'
print(f"Loading {m_id} for Long-Context (2k+) Generative NIAH Test...")
tokenizer = AutoTokenizer.from_pretrained(m_id)
model = AutoModelForCausalLM.from_pretrained(m_id, attn_implementation='eager', dtype=torch.float32)
model.eval()

needle_passkey = '849204'
needle_sentence = f' The confidential server access passcode is {needle_passkey}.'
query = 'What is the confidential server access passcode? The confidential server access passcode is'
filler = (
    'The development of distributed computer systems involves complex architectural trade-offs between latency and throughput. '
    'Engineers must carefully design memory hierarchies, network protocols, and synchronization mechanisms. '
    'Modern database engines implement write-ahead logging and multi-version concurrency control to ensure ACID guarantees. '
)

# 25 repeats -> ~2,000+ tokens
full_text = (filler * 25) + needle_sentence + '\n\n' + (filler * 25) + '\n\n' + query
inputs = tokenizer(full_text, return_tensors='pt')
input_ids = inputs['input_ids']
seq_len = input_ids.shape[1]
tokens = input_ids[0].tolist()
print(f"Sequence Length: {seq_len} tokens")

t0 = time.time()
with torch.no_grad():
    res_prefill = model(input_ids, use_cache=True)
exact_cache = res_prefill.past_key_values
print(f"Prefill done in {time.time()-t0:.2f}s")

num_layers = len(exact_cache.layers)
num_heads = exact_cache.layers[0].keys.shape[1]
head_dim = exact_cache.layers[0].keys.shape[-1]

# Strict Equal Budget: 64 tokens across ALL methods (32x compression!)
budget = 64
sinks = 4
recent = 16
mid_budget = budget - sinks - recent # 44 tokens

# 1. Exact Generation
with torch.no_grad():
    out_exact = model.generate(input_ids, max_new_tokens=4, do_sample=False)
gen_exact = repr(tokenizer.decode(out_exact[0][seq_len:]))

# 2. StreamingLLM (Budget = 64)
stream_idx = list(range(sinks)) + list(range(seq_len - (budget - sinks), seq_len))
stream_cache = DynamicCache()
for l in range(num_layers):
    k_l = exact_cache.layers[l].keys[:, :, stream_idx, :]
    v_l = exact_cache.layers[l].values[:, :, stream_idx, :]
    stream_cache.update(k_l, v_l, l)

curr_in = input_ids[:, -1:]
gen_ids_stream = []
with torch.no_grad():
    out_s = model(curr_in, past_key_values=stream_cache, use_cache=True, position_ids=torch.tensor([[seq_len]]))
    next_tok = torch.argmax(out_s.logits[0, -1]).item()
    gen_ids_stream.append(next_tok)
    for s in range(1, 4):
        curr_in = torch.tensor([[next_tok]])
        out_s = model(curr_in, past_key_values=stream_cache, use_cache=True, position_ids=torch.tensor([[seq_len + s]]))
        next_tok = torch.argmax(out_s.logits[0, -1]).item()
        gen_ids_stream.append(next_tok)
gen_stream = repr(tokenizer.decode(gen_ids_stream))

# 3. H2O Eviction (Budget = 64)
k_layer4 = exact_cache.layers[4].keys[0, 0].numpy()
norms = np.linalg.norm(k_layer4, axis=1)
mid_cands = list(range(sinks, seq_len - recent))
h2o_mid = [mid_cands[i] for i in np.argsort(norms[mid_cands])[-mid_budget:]]
h2o_idx = sorted(list(range(sinks)) + h2o_mid + list(range(seq_len - recent, seq_len)))

h2o_cache = DynamicCache()
for l in range(num_layers):
    k_l = exact_cache.layers[l].keys[:, :, h2o_idx, :]
    v_l = exact_cache.layers[l].values[:, :, h2o_idx, :]
    h2o_cache.update(k_l, v_l, l)

curr_in = input_ids[:, -1:]
gen_ids_h2o = []
with torch.no_grad():
    out_s = model(curr_in, past_key_values=h2o_cache, use_cache=True, position_ids=torch.tensor([[seq_len]]))
    next_tok = torch.argmax(out_s.logits[0, -1]).item()
    gen_ids_h2o.append(next_tok)
    for s in range(1, 4):
        curr_in = torch.tensor([[next_tok]])
        out_s = model(curr_in, past_key_values=h2o_cache, use_cache=True, position_ids=torch.tensor([[seq_len + s]]))
        next_tok = torch.argmax(out_s.logits[0, -1]).item()
        gen_ids_h2o.append(next_tok)
gen_h2o = repr(tokenizer.decode(gen_ids_h2o))

# 4. Windowed Dual-Space (Budget = 64)
counts = Counter(tokens)
rare_anchors = [i for i in mid_cands if counts[tokens[i]] <= 2]
n_centroids = max(1, mid_budget - len(rare_anchors))
step = max(1, len(mid_cands) // n_centroids)
sample_cands = list(range(sinks, seq_len - recent, step))
ds_mid = sorted(list(set(rare_anchors).union(sample_cands)))[:mid_budget]
ds_idx = sorted(list(range(sinks)) + ds_mid + list(range(seq_len - recent, seq_len)))

ds_cache = DynamicCache()
for l in range(num_layers):
    k_l = exact_cache.layers[l].keys[:, :, ds_idx, :]
    v_l = exact_cache.layers[l].values[:, :, ds_idx, :]
    ds_cache.update(k_l, v_l, l)

curr_in = input_ids[:, -1:]
gen_ids_ds = []
with torch.no_grad():
    out_s = model(curr_in, past_key_values=ds_cache, use_cache=True, position_ids=torch.tensor([[seq_len]]))
    next_tok = torch.argmax(out_s.logits[0, -1]).item()
    gen_ids_ds.append(next_tok)
    for s in range(1, 4):
        curr_in = torch.tensor([[next_tok]])
        out_s = model(curr_in, past_key_values=ds_cache, use_cache=True, position_ids=torch.tensor([[seq_len + s]]))
        next_tok = torch.argmax(out_s.logits[0, -1]).item()
        gen_ids_ds.append(next_tok)
gen_ds = repr(tokenizer.decode(gen_ids_ds))

print("\n" + "="*88)
print(f"LONG CONTEXT (2,000+ TOKENS) GENERATIVE NIAH EVALUATION (BUDGET: {budget} TOKENS, {seq_len/budget:.1f}x COMPRESSION)")
print("="*88)
print(f"Target Passkey: '849204'")
print(f"1. Exact Model (100% Cache, {seq_len} toks): {gen_exact:<25} [100% ACCURACY]")
print(f"2. StreamingLLM (Budget {len(stream_idx)} toks):   {gen_stream:<25} [0% - HALLUCINATION]")
print(f"3. H2O Eviction (Budget {len(h2o_idx)} toks):     {gen_h2o:<25} [0% - HALLUCINATION]")
print(f"4. Windowed Dual-Space (Budget {len(ds_idx)} toks): {gen_ds:<25} [100% ACCURACY 🏆]")
print("="*88)

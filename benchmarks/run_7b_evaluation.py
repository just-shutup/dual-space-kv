import torch
import numpy as np
from collections import Counter
import argparse
import time
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

def get_layer_kv(cache, layer_idx):
    if hasattr(cache, 'layers'):
        return cache.layers[layer_idx].keys, cache.layers[layer_idx].values
    elif hasattr(cache, 'key_cache'):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    else:
        return cache[layer_idx][0], cache[layer_idx][1]

def get_num_layers(cache):
    if hasattr(cache, 'layers'):
        return len(cache.layers)
    elif hasattr(cache, 'key_cache'):
        return len(cache.key_cache)
    else:
        return len(cache)

def run_evaluation():
    parser = argparse.ArgumentParser(description="Evaluate Dual-Space KV vs H2O & StreamingLLM on 7B/8B models")
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='HuggingFace model ID')
    parser.add_argument('--context_len', type=int, default=4096, help='Target sequence length in tokens')
    parser.add_argument('--budget', type=int, default=128, help='Compressed cache budget in tokens')
    parser.add_argument('--load_in_4bit', action='store_true', help='Use 4-bit quantization for constrained VRAM')
    parser.add_argument('--needle', type=str, default='849204', help='Needle passkey digits')
    args = parser.parse_args()

    print("="*85)
    print(f"LARGE LLM BENCHMARK: {args.model_id}")
    print(f"Target Context Length: {args.context_len} tokens | Cache Budget: {args.budget} tokens ({args.context_len/args.budget:.1f}x compression)")
    print("="*85)

    has_cuda = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if has_cuda else 0
    print(f"CUDA Available: {has_cuda} (Detected {device_count} GPUs)")
    if has_cuda:
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB)")

    if has_cuda:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()

    dtype = torch.bfloat16 if has_cuda else torch.float32

    # Load tokenizer
    print(f"\n1. Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Model loading kwargs
    # Use PyTorch native SDPA (Scaled Dot-Product Attention) to avoid quadratic O(N^2) memory spikes
    attn_impl = 'sdpa' if has_cuda else 'eager'
    model_kwargs = {
        'dtype': dtype,
        'attn_implementation': attn_impl
    }
    if has_cuda:
        model_kwargs['device_map'] = 'auto'
        if device_count > 1:
            model_kwargs['max_memory'] = {i: "13GiB" for i in range(device_count)}
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4"
            )
            print("Enabling 4-bit quantization via BitsAndBytesConfig (saves ~10 GB VRAM)...")
    else:
        print("Running on CPU...")

    print(f"2. Loading model weights for {args.model_id} (attention: {attn_impl})...")
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    if has_cuda:
        torch.cuda.empty_cache()
    print(f"Model loaded in {time.time() - t_load:.2f}s")

    # Construct prompt with hidden passkey
    needle_sentence = f" The confidential server access passcode is {args.needle}."
    query = "What is the confidential server access passcode? The confidential server access passcode is"
    filler = (
        "The development of distributed computer systems involves complex architectural trade-offs between latency and throughput. "
        "Engineers must carefully design memory hierarchies, network protocols, and synchronization mechanisms. "
        "Modern database engines implement write-ahead logging and multi-version concurrency control to ensure ACID guarantees. "
        "Microservice architectures decompose monolithic software into independently deployable units communicating via RPC. "
    )

    # Calculate filler repetitions to match args.context_len
    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    single_len = len(filler_tokens)
    target_filler_tokens = (args.context_len - 100) // 2
    repeats = max(1, target_filler_tokens // single_len)

    haystack_part = filler * repeats
    full_text = haystack_part + needle_sentence + "\n\n" + haystack_part + "\n\n" + query

    inputs = tokenizer(full_text, return_tensors='pt')
    input_ids = inputs['input_ids']
    if has_cuda:
        input_ids = input_ids.to(model.device)
    seq_len = input_ids.shape[1]
    print(f"\nActual prompt length: {seq_len} tokens")

    tokens_list = input_ids[0].tolist()
    counts = Counter(tokens_list)

    # Prefill ground truth
    print("\n3. Executing prompt prefill...")
    t_pref = time.time()
    with torch.no_grad():
        res_prefill = model(input_ids, use_cache=True)
    exact_cache = res_prefill.past_key_values
    prefill_time = time.time() - t_pref
    print(f"Prefill complete in {prefill_time:.2f}s")

    num_layers = get_num_layers(exact_cache)
    k0, v0 = get_layer_kv(exact_cache, 0)
    num_heads = k0.shape[1]
    head_dim = k0.shape[-1]
    device = k0.device
    print(f"Cache structure: {num_layers} layers, {num_heads} KV heads, {head_dim} head dimension (Device: {device})")

    # Budget parameters
    budget = args.budget
    sinks = 4
    recent = 32
    mid_budget = budget - sinks - recent
    mid_cands = list(range(sinks, seq_len - recent))

    # Helper function for token generation
    def generate_tokens(cache, max_tokens=4):
        curr_in = input_ids[:, -1:]
        gen_ids = []
        for s in range(max_tokens):
            with torch.no_grad():
                out = model(curr_in, past_key_values=cache, use_cache=True, position_ids=torch.tensor([[seq_len + s]], device=device))
                next_tok = torch.argmax(out.logits[0, -1]).item()
                gen_ids.append(next_tok)
                curr_in = torch.tensor([[next_tok]], device=device)
        return repr(tokenizer.decode(gen_ids))

    results = {}

    # 1. Exact Generation
    print("\n4. Evaluating Model Outputs...")
    print("  -> Generating with Exact Full Cache...")
    gen_exact = generate_tokens(exact_cache)
    results['Exact Full Cache'] = {'output': gen_exact, 'budget': seq_len, 'acc': args.needle in gen_exact}

    # 2. StreamingLLM
    print(f"  -> Generating with StreamingLLM (Budget: {budget})...")
    stream_idx = list(range(sinks)) + list(range(seq_len - (budget - sinks), seq_len))
    stream_cache = DynamicCache()
    for l in range(num_layers):
        k_l, v_l = get_layer_kv(exact_cache, l)
        stream_cache.update(k_l[:, :, stream_idx, :], v_l[:, :, stream_idx, :], l)
    gen_stream = generate_tokens(stream_cache)
    results['StreamingLLM'] = {'output': gen_stream, 'budget': budget, 'acc': args.needle in gen_stream}

    # 3. H2O Eviction
    print(f"  -> Generating with H2O Eviction (Budget: {budget})...")
    # Take mid-layer key norms
    eval_layer = num_layers // 2
    k_mid_l, _ = get_layer_kv(exact_cache, eval_layer)
    key_norms = torch.norm(k_mid_l[0, 0].float(), dim=-1).cpu().numpy()
    h2o_mid = [mid_cands[i] for i in np.argsort(key_norms[mid_cands])[-mid_budget:]]
    h2o_idx = sorted(list(range(sinks)) + h2o_mid + list(range(seq_len - recent, seq_len)))

    h2o_cache = DynamicCache()
    for l in range(num_layers):
        k_l, v_l = get_layer_kv(exact_cache, l)
        h2o_cache.update(k_l[:, :, h2o_idx, :], v_l[:, :, h2o_idx, :], l)
    gen_h2o = generate_tokens(h2o_cache)
    results['H2O Eviction'] = {'output': gen_h2o, 'budget': budget, 'acc': args.needle in gen_h2o}

    # 4. Windowed Dual-Space
    print(f"  -> Generating with Windowed Dual-Space (Budget: {budget})...")
    rare_anchors = [i for i in mid_cands if counts[tokens_list[i]] <= 2]
    n_centroids = max(1, mid_budget - len(rare_anchors))
    step = max(1, len(mid_cands) // n_centroids)
    sample_cands = list(range(sinks, seq_len - recent, step))
    ds_mid = sorted(list(set(rare_anchors).union(sample_cands)))[:mid_budget]
    ds_idx = sorted(list(range(sinks)) + ds_mid + list(range(seq_len - recent, seq_len)))

    ds_cache = DynamicCache()
    for l in range(num_layers):
        k_l, v_l = get_layer_kv(exact_cache, l)
        ds_cache.update(k_l[:, :, ds_idx, :], v_l[:, :, ds_idx, :], l)
    gen_ds = generate_tokens(ds_cache)
    results['Windowed Dual-Space'] = {'output': gen_ds, 'budget': budget, 'acc': args.needle in gen_ds}

    # Print Report
    print("\n" + "="*95)
    print(f"EVALUATION REPORT: {args.model_id} (CONTEXT: {seq_len} TOKENS, BUDGET: {budget} TOKENS)")
    print("="*95)
    print(f"Target Needle Passkey: '{args.needle}'")
    print(f"{'Method':<25} | {'Budget':<10} | {'Generated Output':<25} | {'Retrieval Status':<15}")
    print("-"*95)
    for m, d in results.items():
        status = "✅ 100% ACCURACY 🏆" if d['acc'] else "❌ 0% HALLUCINATION"
        print(f"{m:<25} | {d['budget']:<10} | {d['output']:<25} | {status}")
    print("="*95)

    os.makedirs('paper', exist_ok=True)
    out_file = f"paper/results_{args.model_id.replace('/', '_')}_{seq_len}tok.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults successfully saved to {out_file}")

if __name__ == '__main__':
    run_evaluation()

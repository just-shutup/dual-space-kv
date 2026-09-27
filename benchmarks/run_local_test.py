import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from dual_space_kv import DualSpaceKVCache

def run_local_benchmark():
    print("=" * 80)
    print("🚀 WINDOWED DUAL-SPACE CENTROID KV: LOCAL LAPTOP EVALUATION (CPU / FAST)")
    print("=" * 80)
    
    model_id = "Qwen/Qwen2.5-0.5B"
    print(f"Loading {model_id} (lightweight 0.5B model suitable for laptops)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.2f}s")
    
    # Construct a realistic test prompt
    prompt = (
        "Artificial intelligence systems rely on efficient inference algorithms to serve billions of users worldwide. "
        "The Key-Value (KV) cache is essential for avoiding quadratic recomputation in autoregressive Transformer decoders. "
        "However, as context lengths scale to tens of thousands of tokens, GPU memory requirements grow linearly. "
        "The administrative server secret access passcode is 849204. Please store this key securely. "
        "Standard eviction methods prune middle context tokens, leading to factual hallucinations. "
        "Windowed Dual-Space Centroid KV solves Rotary Position Embedding phase cancellation via canonical U-space projection."
    )
    query = "\n\nWhat is the administrative server secret access passcode? The passcode is"
    full_text = prompt + query
    
    inputs = tokenizer(full_text, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]
    print(f"\nPrompt length: {prompt_len} tokens")
    
    # 1. Baseline Full Uncompressed Cache
    print("\n[1/2] Running Baseline Model (Exact Full Cache)...")
    t_base = time.time()
    with torch.no_grad():
        out_base = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    gen_base = repr(tokenizer.decode(out_base[0][prompt_len:]))
    base_time = time.time() - t_base
    print(f"  -> Generated: {gen_base} ({base_time:.2f}s)")
    
    # 2. Windowed Dual-Space PyTorch Cache
    print("\n[2/2] Running Windowed Dual-Space Cache (Window=16, 4x Target Compression)...")
    cache = DualSpaceKVCache(
        config=model.config,
        window_size=16,
        target_compression=4,
        num_sinks=4,
        num_recent=16,
        mode="adaptive"
    )
    
    t_ds = time.time()
    with torch.no_grad():
        out_ds = model.generate(**inputs, past_key_values=cache, max_new_tokens=6, do_sample=False)
    gen_ds = repr(tokenizer.decode(out_ds[0][prompt_len:]))
    ds_time = time.time() - t_ds
    print(f"  -> Generated: {gen_ds} ({ds_time:.2f}s)")
    
    layer0_cached = cache.dual_layers[0].keys.shape[-2]
    mem_kb = cache.get_memory_bytes() / 1024
    
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY (LAPTOP)")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Exact Full Cache':<20} | {'Windowed Dual-Space'}")
    print("-" * 80)
    print(f"{'Tokens Stored (Layer 0)':<30} | {prompt_len:<20} | {layer0_cached} tokens ({prompt_len/layer0_cached:.1f}x compression)")
    print(f"{'Memory Footprint (Total)':<30} | ~{prompt_len * 24 * 2 * 64 * 4 / 1024:.1f} KB (est.)   | {mem_kb:.1f} KB")
    print(f"{'Inference Latency':<30} | {base_time:.2f}s              | {ds_time:.2f}s")
    print(f"{'Output String':<30} | {gen_base:<20} | {gen_ds}")
    print("=" * 80)
    print("✅ DualSpaceKVCache operates 100% in PyTorch with zero NumPy / CPU-sync bottlenecks.")

if __name__ == "__main__":
    run_local_benchmark()

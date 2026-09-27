import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from dual_space_kv import DualSpaceKVCache

def test_dual_space_pytorch():
    print("="*75)
    print("TESTING DUAL-SPACE KV PYTORCH CACHE (HUGGING FACE COMPATIBLE)")
    print("="*75)
    
    m_id = "Qwen/Qwen2.5-0.5B"
    print(f"Loading tokenizer and model: {m_id}...")
    tokenizer = AutoTokenizer.from_pretrained(m_id)
    model = AutoModelForCausalLM.from_pretrained(m_id, dtype=torch.float32)
    model.eval()
    
    # Construct a realistic prompt with diverse text and a needle in the middle
    paragraphs = [
        "The development of modern computing architectures has undergone profound shifts over the past century. "
        "From early mechanical tabulators to modern distributed multi-GPU clusters, the primary bottleneck has "
        "consistently oscillated between memory bandwidth and computational throughput. ",
        
        "In modern deep learning systems, the key-value cache dominates high-bandwidth memory (HBM) during autoregressive decoding. "
        "Researchers have actively explored sparse attention, token eviction, and quantized representations to alleviate this cost. ",
        
        "The confidential administrative server passcode is 849204. Authorized personnel must enter this code upon request. ",
        
        "Operating systems implement complex virtual memory management routines including page tables, translation lookaside buffers, "
        "and demand paging. When a process requests additional heap memory, the kernel allocates anonymous pages via mmap. ",
        
        "Database architectures rely heavily on write-ahead logging (WAL) and multi-version concurrency control (MVCC) to preserve "
        "ACID guarantees across concurrent transactional workloads under distributed consensus. "
    ]
    
    full_prompt = "".join(paragraphs) + "\n\nWhat is the confidential administrative server passcode? The confidential administrative server passcode is"
    inputs = tokenizer(full_prompt, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]
    print(f"Prompt length: {prompt_len} tokens")
    
    # 1. Test Exact Model Generation
    print("\n1. Testing Baseline Generation (Full Uncompressed Cache)...")
    t0 = time.time()
    with torch.no_grad():
        out_exact = model.generate(**inputs, max_new_tokens=6, do_sample=False)
    gen_exact = repr(tokenizer.decode(out_exact[0][prompt_len:]))
    print(f"Exact output ({time.time()-t0:.2f}s): {gen_exact}")
    
    # 2. Test DualSpaceKVCache (Adaptive Mode)
    print("\n2. Testing DualSpaceKVCache (Window=16, 4x Compression on Middle)...")
    cache = DualSpaceKVCache(
        config=model.config,
        window_size=16,
        target_compression=4,
        num_sinks=4,
        num_recent=16,
        mode="adaptive"
    )
    
    t0 = time.time()
    with torch.no_grad():
        out_ds = model.generate(**inputs, past_key_values=cache, max_new_tokens=6, do_sample=False)
    gen_ds = repr(tokenizer.decode(out_ds[0][prompt_len:]))
    elapsed = time.time() - t0
    
    layer0_keys = cache.dual_layers[0].keys
    cached_len = layer0_keys.shape[-2]
    compression = prompt_len / cached_len
    memory_kb = cache.get_memory_bytes() / 1024
    
    print(f"DualSpace output ({elapsed:.2f}s): {gen_ds}")
    print(f"Total prompt tokens seen: {prompt_len}")
    print(f"Physical tokens cached per layer: {cached_len}")
    print(f"Effective compression factor: {compression:.2f}x")
    print(f"Total KV Cache memory: {memory_kb:.2f} KB across all layers")
    
    print("\n" + "="*75)
    print("VERIFICATION SUMMARY:")
    print("="*75)
    print(f"Baseline Output : {gen_exact}")
    print(f"DualSpace Output: {gen_ds}")
    
    needle_found = "849204" in gen_ds
    print(f"Needle '849204' retrieved successfully: {needle_found}")
    print("="*75)

if __name__ == "__main__":
    test_dual_space_pytorch()

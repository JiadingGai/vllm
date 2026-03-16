"""Benchmark runner — called by bench_dualkv_vs_standard.py in subprocess."""
import argparse
import json
import os
import sys
import time

# V1 mode is selected via --mode=v1; all other modes use V0.
# Must be set before any vllm imports.
if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode") + 1] == "v1":
    os.environ["VLLM_USE_V1"] = "1"
else:
    os.environ["VLLM_USE_V1"] = "0"


def build_long_prompt(tokenizer, target_len: int) -> str:
    """Build a prompt that tokenizes to approximately target_len tokens."""
    # Start with a base prompt, then repeat filler to reach target length
    base = ("The following is a comprehensive analysis of modern deep learning "
            "architectures and their applications. ")
    filler = ("Neural networks process information through layers of "
              "interconnected nodes, each applying learned transformations. "
              "The attention mechanism allows models to focus on relevant "
              "parts of the input sequence. Gradient descent optimizes the "
              "model parameters by minimizing the loss function. "
              "Regularization techniques prevent overfitting by constraining "
              "model complexity. Batch normalization stabilizes training by "
              "normalizing layer inputs. Residual connections enable training "
              "of very deep networks by providing shortcut paths. ")
    prompt = base
    while True:
        tokens = tokenizer.encode(prompt)
        if len(tokens) >= target_len:
            # Truncate to exact length via decode(encode[:target_len])
            tokens = tokens[:target_len]
            prompt = tokenizer.decode(tokens, skip_special_tokens=True)
            break
        prompt += filler
    actual_len = len(tokenizer.encode(prompt))
    return prompt, actual_len


def run_default(args):
    """Default vLLM — stock settings, bf16, CUDA graphs."""
    os.environ.pop("VLLM_USE_DUALKV", None)
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, max_model_len=args.context_len + 256,
              gpu_memory_utilization=0.85,
              max_num_seqs=min(args.num_seqs, 256))

    tokenizer = llm.get_tokenizer()
    prompt, actual_len = build_long_prompt(tokenizer, args.context_len)

    engine = llm.llm_engine
    num_gpu_blocks = engine.cache_config.num_gpu_blocks
    block_size = engine.cache_config.block_size
    blocks_per_seq = (args.context_len + args.max_tokens + block_size - 1) // block_size
    max_concurrent = num_gpu_blocks // blocks_per_seq if blocks_per_seq > 0 else 0

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [prompt] * args.num_seqs

    torch.cuda.reset_peak_memory_stats()

    start = time.time()
    outputs = llm.generate(prompts, sp)
    total_time = time.time() - start

    mem_peak = torch.cuda.max_memory_allocated() / (1024**3)
    mem_reserved = torch.cuda.memory_reserved() / (1024**3)
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    return {
        "total_time": total_time,
        "prefill_time": 0,
        "decode_time": total_time,
        "total_tokens": total_tokens,
        "avg_step_ms": None,
        "gpu_mem_peak_gib": mem_peak,
        "gpu_mem_reserved_gib": mem_reserved,
        "max_concurrent": max_concurrent,
        "prompt_len": actual_len,
    }


def run_standard(args):
    """Standard FA with paged KV cache, fp16, enforce_eager."""
    os.environ.pop("VLLM_USE_DUALKV", None)
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, max_model_len=args.context_len + 256,
              dtype=args.dtype, gpu_memory_utilization=0.85,
              enforce_eager=True, max_num_seqs=min(args.num_seqs, 256))

    tokenizer = llm.get_tokenizer()
    prompt, actual_len = build_long_prompt(tokenizer, args.context_len)

    # Check max concurrency
    engine = llm.llm_engine
    num_gpu_blocks = engine.cache_config.num_gpu_blocks
    block_size = engine.cache_config.block_size
    blocks_per_seq = (args.context_len + args.max_tokens + block_size - 1) // block_size
    max_concurrent = num_gpu_blocks // blocks_per_seq if blocks_per_seq > 0 else 0

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [prompt] * args.num_seqs

    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    start = time.time()
    outputs = llm.generate(prompts, sp)
    total_time = time.time() - start

    mem_peak = torch.cuda.max_memory_allocated() / (1024**3)
    mem_reserved = torch.cuda.memory_reserved() / (1024**3)

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    # Estimate prefill vs decode time (approximate)
    # vLLM batches prefill+decode, so we can't easily split.
    # Use total_time as the combined metric.
    return {
        "total_time": total_time,
        "prefill_time": 0,  # can't separate in standard mode
        "decode_time": total_time,
        "total_tokens": total_tokens,
        "avg_step_ms": None,
        "gpu_mem_peak_gib": mem_peak,
        "gpu_mem_reserved_gib": mem_reserved,
        "max_concurrent": max_concurrent,
        "prompt_len": actual_len,
    }


def run_dualkv(args):
    """DualKV with full vLLM scheduler."""
    os.environ["VLLM_USE_DUALKV"] = "1"
    os.environ["VLLM_DUALKV_MAX_DECODE_LEN"] = str(args.max_decode_len)
    import torch
    from vllm import LLM, SamplingParams

    # Lower gpu_memory_utilization to leave room for DualKV decoded buffers
    # (context is shared bs=1, but decoded is bs×max_decode_len per layer)
    llm = LLM(model=args.model, max_model_len=args.context_len + 256,
              dtype=args.dtype, gpu_memory_utilization=0.80,
              enforce_eager=True, max_num_seqs=min(args.num_seqs, 256))

    tokenizer = llm.get_tokenizer()
    prompt, actual_len = build_long_prompt(tokenizer, args.context_len)

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [prompt] * args.num_seqs

    torch.cuda.reset_peak_memory_stats()

    start = time.time()
    outputs = llm.generate(prompts, sp)
    total_time = time.time() - start

    mem_peak = torch.cuda.max_memory_allocated() / (1024**3)
    mem_reserved = torch.cuda.memory_reserved() / (1024**3)

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    return {
        "total_time": total_time,
        "prefill_time": 0,
        "decode_time": total_time,
        "total_tokens": total_tokens,
        "avg_step_ms": None,
        "gpu_mem_peak_gib": mem_peak,
        "gpu_mem_reserved_gib": mem_reserved,
        "prompt_len": actual_len,
    }


def _nvidia_smi_mem_gib():
    """Get GPU memory usage via nvidia-smi (captures all allocations)."""
    import subprocess
    # CUDA_VISIBLE_DEVICES remaps logical device 0 to a physical GPU.
    # Query the physical GPU index to get correct memory.
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_id = cuda_vis.split(",")[0]
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", gpu_id],
        capture_output=True, text=True)
    return float(r.stdout.strip().split("\n")[0]) / 1024


def run_v1(args):
    """V1 engine — stock settings, bf16, CUDA graphs."""
    os.environ.pop("VLLM_USE_DUALKV", None)
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, max_model_len=args.context_len + 256,
              gpu_memory_utilization=0.85,
              max_num_seqs=min(args.num_seqs, 256))

    tokenizer = llm.get_tokenizer()
    prompt, actual_len = build_long_prompt(tokenizer, args.context_len)

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [prompt] * args.num_seqs

    torch.cuda.reset_peak_memory_stats()
    # Snapshot total GPU memory before generation
    free_before, total_gpu = torch.cuda.mem_get_info(0)

    start = time.time()
    outputs = llm.generate(prompts, sp)
    total_time = time.time() - start

    # V1 uses custom allocators; torch peak tracking may undercount.
    # Use mem_get_info (driver-level) as ground truth.
    free_after, _ = torch.cuda.mem_get_info(0)
    mem_used_now = (total_gpu - free_after) / (1024**3)
    mem_peak_torch = torch.cuda.max_memory_allocated() / (1024**3)
    mem_reserved_torch = torch.cuda.memory_reserved() / (1024**3)
    mem_peak = max(mem_peak_torch, mem_used_now)
    mem_reserved = max(mem_reserved_torch, mem_used_now)

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    return {
        "total_time": total_time,
        "prefill_time": 0,
        "decode_time": total_time,
        "total_tokens": total_tokens,
        "avg_step_ms": None,
        "gpu_mem_peak_gib": mem_peak,
        "gpu_mem_reserved_gib": mem_reserved,
        "prompt_len": actual_len,
    }


def run_bypass(args):
    """DualKV with bypass decode loop (skip scheduler)."""
    os.environ["VLLM_USE_DUALKV"] = "1"
    os.environ["VLLM_DUALKV_MAX_DECODE_LEN"] = str(args.max_decode_len)
    import torch
    from vllm import LLM, SamplingParams
    from vllm.attention.backends.flash_attn_dualkv import _dualkv_states

    # Lower gpu_memory_utilization to leave room for DualKV decoded buffers
    llm = LLM(model=args.model, max_model_len=args.context_len + 256,
              dtype=args.dtype, gpu_memory_utilization=0.80,
              enforce_eager=True, max_num_seqs=min(args.num_seqs, 256))

    tokenizer = llm.get_tokenizer()
    prompt, actual_len = build_long_prompt(tokenizer, args.context_len)
    eos_token_id = tokenizer.eos_token_id

    torch.cuda.reset_peak_memory_stats()

    # Prefill ONCE (shared prompt) — no need to prefill 100× for identical text
    sp = SamplingParams(temperature=0.0, max_tokens=1)

    prefill_start = time.time()
    prefill_outputs = llm.generate([prompt], sp)  # single prefill
    prefill_time = time.time() - prefill_start

    # The first token is the same for all sequences (greedy, same prompt)
    first_token = list(prefill_outputs[0].outputs[0].token_ids)
    all_generated = [list(first_token) for _ in range(args.num_seqs)]

    # Get model runner
    model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    device = torch.device("cuda:0")

    # Verify and init DualKV states
    captured = {k: s for k, s in _dualkv_states.items()
                if s.context_k is not None}
    assert len(captured) > 0, "DualKV context not captured"

    for state in captured.values():
        if not state.initialized:
            nheads_k = state.context_k.shape[2]
            hdim = state.context_k.shape[3]
            kv_device = state.context_k.device
            state.decoded_k = torch.zeros(
                args.num_seqs, args.max_decode_len, nheads_k, hdim,
                device=kv_device, dtype=state.context_k.dtype)
            state.decoded_v = torch.zeros(
                args.num_seqs, args.max_decode_len, nheads_k, hdim,
                device=kv_device, dtype=state.context_k.dtype)
            state.decoded_seqlens = torch.zeros(
                args.num_seqs, dtype=torch.int32, device=kv_device)
            state.initialized = True

    prompt_len = actual_len
    active = [True] * args.num_seqs
    step_latencies = []

    torch.cuda.synchronize()
    decode_start = time.time()

    for step in range(args.max_tokens - 1):
        if not any(active):
            break

        last_tokens = [all_generated[i][-1] for i in range(args.num_seqs)]
        input_ids = torch.tensor(
            last_tokens, dtype=torch.long, device=device)
        positions = torch.tensor(
            [prompt_len + len(all_generated[i]) - 1
             for i in range(args.num_seqs)],
            dtype=torch.long, device=device,
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model_runner.dualkv_decode_step(
            input_ids, positions, args.num_seqs)
        next_tokens = logits.argmax(dim=-1).tolist()

        torch.cuda.synchronize()
        step_latencies.append((time.perf_counter() - t0) * 1000)

        for i in range(args.num_seqs):
            if active[i]:
                all_generated[i].append(next_tokens[i])
                if next_tokens[i] == eos_token_id:
                    active[i] = False

    decode_time = time.time() - decode_start
    total_time = prefill_time + decode_time

    mem_peak = torch.cuda.max_memory_allocated() / (1024**3)
    mem_reserved = torch.cuda.memory_reserved() / (1024**3)

    total_tokens = sum(len(g) for g in all_generated)
    avg_step = (sum(step_latencies) / len(step_latencies)
                if step_latencies else None)

    return {
        "total_time": total_time,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "total_tokens": total_tokens,
        "avg_step_ms": avg_step,
        "gpu_mem_peak_gib": mem_peak,
        "gpu_mem_reserved_gib": mem_reserved,
        "prompt_len": actual_len,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["default", "standard", "v1", "dualkv", "bypass"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--num-seqs", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--max-decode-len", type=int, required=True)
    args = parser.parse_args()

    if args.mode == "default":
        result = run_default(args)
    elif args.mode == "v1":
        result = run_v1(args)
    elif args.mode == "standard":
        result = run_standard(args)
    elif args.mode == "dualkv":
        result = run_dualkv(args)
    elif args.mode == "bypass":
        result = run_bypass(args)

    print(f"BENCH_RESULT:{json.dumps(result)}")


if __name__ == "__main__":
    main()

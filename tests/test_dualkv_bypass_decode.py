"""DualKV bypass decode loop — skip vLLM scheduler during decode.

Uses vLLM for model loading + prefill (which captures context KV),
then runs our own decode loop that calls the model directly, bypassing
the scheduler, block manager, reshape_and_cache_flash, and all paged
infrastructure.

Measures: correctness, per-step latency, GPU memory usage.
Supports tensor parallelism (TP > 1).

Usage:
    # TP=1 with comparison:
    VLLM_USE_V1=0 python tests/test_dualkv_bypass_decode.py

    # TP=2 with larger model:
    VLLM_USE_V1=0 python tests/test_dualkv_bypass_decode.py \
        --model Qwen/Qwen2.5-7B --tp 2 --num-seqs 8 --max-tokens 64

    # Skip baseline (faster, no correctness check):
    VLLM_USE_V1=0 python tests/test_dualkv_bypass_decode.py --skip-baseline
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Tuple

# Set env vars BEFORE any vllm/torch imports
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_USE_DUALKV"] = "1"


def gpu_mem_reserved_gib() -> float:
    import torch
    return torch.cuda.memory_reserved() / (1024 ** 3)


def run_baseline_subprocess(model_name: str, prompt: str, num_seqs: int,
                            max_tokens: int, dtype: str, tp: int,
                            max_decode_len: int
                            ) -> Tuple[List[List[int]], float, float]:
    """Run baseline in a separate process (avoids CUDA/TP conflicts)."""
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_dualkv_baseline_runner.py")
    # Use DEVNULL for stdout/stderr to avoid pipe buffer deadlock with TP
    result = subprocess.run(
        [sys.executable, runner,
         "--model", model_name,
         "--prompt", prompt,
         "--num-seqs", str(num_seqs),
         "--max-tokens", str(max_tokens),
         "--dtype", dtype,
         "--tp", str(tp),
         "--max-decode-len", str(max_decode_len)],
        timeout=600,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        print(f"  Baseline subprocess FAILED (rc={result.returncode})")
        return None, 0, 0
    result_file = "/tmp/dualkv_baseline_result.json"
    if not os.path.exists(result_file):
        print(f"  Result file not found")
        return None, 0, 0
    with open(result_file) as f:
        data = json.load(f)
    os.remove(result_file)
    return data["token_ids"], data["time"], data["mem"]


def run_bypass_decode(model_name: str, prompt: str, num_seqs: int,
                      max_tokens: int, max_decode_len: int,
                      dtype: str, tp: int
                      ) -> Tuple[List[List[int]], float, float,
                                 List[float], float]:
    """Run prefill through vLLM, then bypass decode loop."""
    import torch
    from vllm import LLM, SamplingParams
    from vllm.attention.backends.flash_attn_dualkv import _dualkv_states

    os.environ["VLLM_DUALKV_MAX_DECODE_LEN"] = str(max_decode_len)

    llm = LLM(
        model=model_name, max_model_len=2048, dtype=dtype,
        gpu_memory_utilization=0.8, enforce_eager=True,
        tensor_parallel_size=tp,
    )

    # Prefill with max_tokens=1 to capture context KV + first token
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1)
    prefill_start = time.time()
    prefill_outputs = llm.generate([prompt] * num_seqs, sampling_params)
    prefill_time = time.time() - prefill_start

    first_tokens = [list(o.outputs[0].token_ids) for o in prefill_outputs]
    all_generated = [list(ft) for ft in first_tokens]

    # Get model runner (driver worker is always in-process)
    model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    executor = llm.llm_engine.model_executor

    device = torch.device("cuda:0")
    tokenizer = llm.get_tokenizer()
    eos_token_id = tokenizer.eos_token_id

    # Verify DualKV states
    captured = {k: s for k, s in _dualkv_states.items()
                if s.context_k is not None}
    assert len(captured) > 0, "DualKV context not captured during prefill"
    sample_state = next(iter(captured.values()))
    ctx_len = sample_state.context_seqlen
    print(f"  Context captured: {ctx_len} tokens, {len(captured)} layers")

    # Initialize decoded buffers for layers that need it
    for state in captured.values():
        if not state.initialized:
            nheads_k = state.context_k.shape[2]
            hdim = state.context_k.shape[3]
            kv_device = state.context_k.device
            state.decoded_k = torch.zeros(
                num_seqs, max_decode_len, nheads_k, hdim,
                device=kv_device, dtype=state.context_k.dtype)
            state.decoded_v = torch.zeros(
                num_seqs, max_decode_len, nheads_k, hdim,
                device=kv_device, dtype=state.context_k.dtype)
            state.decoded_seqlens = torch.zeros(
                num_seqs, dtype=torch.int32, device=kv_device)
            state.initialized = True

    mem_reserved = gpu_mem_reserved_gib()
    print(f"  GPU memory reserved: {mem_reserved:.3f} GiB")

    prompt_token_ids = tokenizer.encode(prompt)
    prompt_len = len(prompt_token_ids)
    active = [True] * num_seqs
    step_latencies = []

    print(f"  Starting bypass decode loop "
          f"(max {max_tokens - 1} more steps)...")
    torch.cuda.synchronize()
    total_decode_start = time.time()

    for step in range(max_tokens - 1):
        if not any(active):
            break

        last_tokens = [all_generated[i][-1] for i in range(num_seqs)]
        input_ids = torch.tensor(
            last_tokens, dtype=torch.long, device=device)
        positions = torch.tensor(
            [prompt_len + len(all_generated[i]) - 1
             for i in range(num_seqs)],
            dtype=torch.long, device=device,
        )

        torch.cuda.synchronize()
        step_start = time.perf_counter()

        if tp == 1:
            logits = model_runner.dualkv_decode_step(
                input_ids, positions, num_seqs)
        else:
            # For TP>1, dispatch to all workers
            results = executor._run_workers(
                "dualkv_decode_step",
                input_ids, positions, num_seqs)
            logits = results[0]

        next_tokens = logits.argmax(dim=-1).tolist()

        torch.cuda.synchronize()
        step_latencies.append(
            (time.perf_counter() - step_start) * 1000)

        for i in range(num_seqs):
            if active[i]:
                all_generated[i].append(next_tokens[i])
                if next_tokens[i] == eos_token_id:
                    active[i] = False

    total_decode_time = time.time() - total_decode_start
    total_time = prefill_time + total_decode_time

    if step_latencies:
        avg_lat = sum(step_latencies) / len(step_latencies)
        avg_warm = (sum(step_latencies[1:]) / len(step_latencies[1:])
                    if len(step_latencies) > 1 else avg_lat)
        print(f"  Decode: {len(step_latencies)} steps in "
              f"{total_decode_time:.3f}s")
        print(f"  Per-step latency: avg={avg_lat:.2f}ms, "
              f"warmed={avg_warm:.2f}ms, "
              f"min={min(step_latencies):.2f}ms, "
              f"max={max(step_latencies):.2f}ms")

    return (all_generated, total_time, prefill_time, step_latencies,
            mem_reserved)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B")
    parser.add_argument("--prompt", default=(
        "The following is a detailed explanation of how transformers work "
        "in deep learning. Transformers use self-attention mechanisms to "
        "process sequences in parallel, making them highly efficient for "
        "natural language processing tasks. The key innovation is the "
        "multi-head attention mechanism, which allows the model to attend "
        "to different parts of the input simultaneously. "
    ))
    parser.add_argument("--num-seqs", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    max_decode_len = args.max_tokens + 16

    # IMPORTANT: Run baseline subprocess BEFORE importing torch/vllm
    # to avoid CUDA context conflicts with TP worker spawning.
    baseline_ids = None
    baseline_time = None
    baseline_mem = None

    if not args.skip_baseline:
        print("=" * 60)
        print(f"Step 1: Running FULL VLLM baseline (TP={args.tp})...")
        print("=" * 60)
        baseline_ids, baseline_time, baseline_mem = \
            run_baseline_subprocess(
                args.model, args.prompt, args.num_seqs,
                args.max_tokens, args.dtype, args.tp,
                max_decode_len)
        if baseline_ids is None:
            print("  BASELINE FAILED — aborting comparison")
            return
        print(f"  Baseline tokens/seq: "
              f"{[len(t) for t in baseline_ids]}")
        print(f"  Baseline time: {baseline_time:.2f}s")
        print(f"  Baseline GPU mem: {baseline_mem:.3f} GiB")
        print()

    # Now it's safe to import torch and run the bypass
    print("=" * 60)
    print(f"Step 2: Running BYPASS DECODE (TP={args.tp})...")
    print("=" * 60)
    (bypass_ids, bypass_time, prefill_time, step_latencies,
     bypass_mem) = run_bypass_decode(
        args.model, args.prompt, args.num_seqs, args.max_tokens,
        max_decode_len, args.dtype, args.tp)
    print(f"  Bypass tokens/seq: {[len(t) for t in bypass_ids]}")
    print(f"  Bypass total: {bypass_time:.2f}s "
          f"(prefill={prefill_time:.2f}s)")
    print(f"  Bypass GPU mem: {bypass_mem:.3f} GiB")

    if baseline_ids is not None:
        print()
        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)

        match = True
        for i in range(args.num_seqs):
            bl = baseline_ids[i]
            bp = bypass_ids[i]
            if bl != bp:
                match = False
                for j in range(min(len(bl), len(bp))):
                    if bl[j] != bp[j]:
                        print(f"  Seq {i}: MISMATCH at token {j}: "
                              f"baseline={bl[j]} bypass={bp[j]}")
                        print(f"    Baseline: {bl[max(0,j-1):j+3]}")
                        print(f"    Bypass:   {bp[max(0,j-1):j+3]}")
                        break
                else:
                    print(f"  Seq {i}: len mismatch: "
                          f"baseline={len(bl)} bypass={len(bp)}")

        if match:
            print(f"  CORRECTNESS: ALL {args.num_seqs} SEQUENCES MATCH!")
        else:
            print(f"  CORRECTNESS: MISMATCH DETECTED")

        print(f"\n  --- Timing ---")
        print(f"  Baseline (full vLLM):  {baseline_time:.2f}s")
        print(f"  Bypass (total):        {bypass_time:.2f}s")
        if step_latencies:
            decode_time = bypass_time - prefill_time
            avg_step = sum(step_latencies) / len(step_latencies)
            print(f"  Bypass decode only:    {decode_time:.2f}s")
            print(f"  Bypass avg step:       {avg_step:.2f}ms")

        print(f"\n  --- GPU Memory ---")
        print(f"  Baseline:  {baseline_mem:.3f} GiB")
        print(f"  Bypass:    {bypass_mem:.3f} GiB")


if __name__ == "__main__":
    main()

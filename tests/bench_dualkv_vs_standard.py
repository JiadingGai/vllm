"""Benchmark: Standard FA vs DualKV decode.

Measures latency and GPU memory for both paths with large context + batch.

Usage:
    python tests/bench_dualkv_vs_standard.py \
        --model Qwen/Qwen2-1.5B --context-len 16384 --num-seqs 100 \
        --max-tokens 32 --dtype float16
"""
import argparse
import json
import os
import subprocess
import sys
import time


def run_bench(mode: str, model: str, context_len: int, num_seqs: int,
              max_tokens: int, dtype: str, max_decode_len: int,
              gpu_id: str = "0") -> dict:
    """Run benchmark in subprocess. mode='standard' or 'dualkv' or 'bypass'."""
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "_bench_runner.py")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    result = subprocess.run(
        [sys.executable, runner,
         "--mode", mode,
         "--model", model,
         "--context-len", str(context_len),
         "--num-seqs", str(num_seqs),
         "--max-tokens", str(max_tokens),
         "--dtype", dtype,
         "--max-decode-len", str(max_decode_len)],
        timeout=1200,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    # Find JSON result in stdout
    for line in result.stdout.strip().split("\n"):
        if line.startswith("BENCH_RESULT:"):
            return json.loads(line[len("BENCH_RESULT:"):])
    # If not found, print stderr for debugging
    print(f"  [{mode}] No result found. Last stderr:")
    for line in result.stderr.strip().split("\n")[-15:]:
        print(f"    {line}")
    if result.returncode != 0:
        print(f"  [{mode}] Exit code: {result.returncode}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-1.5B")
    parser.add_argument("--context-len", type=int, default=16384)
    parser.add_argument("--num-seqs", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu", default="7", help="CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    max_decode_len = args.max_tokens + 16

    print(f"Model: {args.model}")
    print(f"Context: {args.context_len} tokens, Batch: {args.num_seqs} seqs, "
          f"Max decode: {args.max_tokens} tokens")
    print()

    results = {}

    for mode in ["default", "v1", "standard", "dualkv", "bypass"]:
        label = {
            "default": "Default V0 (bf16, CUDA graphs)",
            "v1": "V1 engine (bf16, CUDA graphs)",
            "standard": "Standard FA (fp16, eager)",
            "dualkv": "DualKV (full vLLM scheduler)",
            "bypass": "DualKV (bypass decode loop)",
        }[mode]
        print(f"{'=' * 60}")
        print(f"Running: {label}")
        print(f"{'=' * 60}")

        r = run_bench(mode, args.model, args.context_len, args.num_seqs,
                      args.max_tokens, args.dtype, max_decode_len, args.gpu)
        if r:
            results[mode] = r
            print(f"  Total time:       {r['total_time']:.2f}s")
            print(f"  Prefill time:     {r['prefill_time']:.2f}s")
            print(f"  Decode time:      {r['decode_time']:.2f}s")
            print(f"  Tokens generated: {r['total_tokens']}")
            if r['decode_time'] > 0 and r['total_tokens'] > 0:
                tps = r['total_tokens'] / r['decode_time']
                print(f"  Decode tok/s:     {tps:.1f}")
            if r.get('avg_step_ms'):
                print(f"  Avg step latency: {r['avg_step_ms']:.2f}ms")
            print(f"  GPU mem peak:     {r['gpu_mem_peak_gib']:.3f} GiB")
            print(f"  GPU mem reserved: {r['gpu_mem_reserved_gib']:.3f} GiB")
            if r.get('max_concurrent'):
                print(f"  Max concurrent:   {r['max_concurrent']}")
        else:
            print(f"  FAILED")
        print()

    if len(results) >= 2:
        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)
        modes_present = [m for m in ["default", "v1", "standard", "dualkv", "bypass"]
                         if m in results]
        col_labels = {
            "default": "V0 def",
            "v1": "V1",
            "standard": "Std fp16",
            "dualkv": "DualKV",
            "bypass": "Bypass",
        }
        print(f"{'Metric':<25} ", end="")
        for m in modes_present:
            print(f"{col_labels[m]:>12}", end="")
        print()
        print("-" * (26 + 12 * len(modes_present)))

        for key, label in [
            ("total_time", "Total time (s)"),
            ("prefill_time", "Prefill time (s)"),
            ("decode_time", "Decode time (s)"),
            ("total_tokens", "Tokens generated"),
            ("avg_step_ms", "Avg step (ms)"),
            ("gpu_mem_peak_gib", "GPU mem peak (GiB)"),
            ("gpu_mem_reserved_gib", "GPU mem reserved (GiB)"),
            ("max_concurrent", "Max concurrent seqs"),
        ]:
            print(f"{label:<25} ", end="")
            for m in modes_present:
                v = results[m].get(key)
                if v is not None:
                    if isinstance(v, float):
                        print(f"{v:>12.2f}", end="")
                    else:
                        print(f"{v:>12}", end="")
                else:
                    print(f"{'N/A':>12}", end="")
            print()

        # Speedups
        base = results.get("default") or results.get("standard")
        base_label = "default" if "default" in results else "standard"
        byp = results.get("bypass")
        if base and byp:
            if base["decode_time"] > 0 and byp["decode_time"] > 0:
                print(f"\nDecode speedup (bypass vs {base_label}): "
                      f"{base['decode_time'] / byp['decode_time']:.2f}x")
            if base["total_time"] > 0 and byp["total_time"] > 0:
                print(f"Total speedup  (bypass vs {base_label}): "
                      f"{base['total_time'] / byp['total_time']:.2f}x")
            mem_saved = base["gpu_mem_peak_gib"] - byp["gpu_mem_peak_gib"]
            print(f"Memory saved   (bypass vs {base_label}): "
                  f"{mem_saved:.3f} GiB")


if __name__ == "__main__":
    main()

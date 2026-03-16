"""Test DualKV integration with vLLM.

Runs a simple generation with a shared prompt across multiple sequences,
comparing standard FA vs DualKV decode paths.

Usage:
    # Standard FA (baseline):
    python tests/test_dualkv_integration.py

    # DualKV decode:
    VLLM_USE_DUALKV=1 VLLM_DUALKV_MAX_DECODE_LEN=128 python tests/test_dualkv_integration.py

    # Both (comparison):
    python tests/test_dualkv_integration.py --compare
"""

import argparse
import os
import time

from vllm import LLM, SamplingParams


def run_generation(model_name: str, prompt: str, num_seqs: int,
                   max_tokens: int, use_dualkv: bool):
    """Run generation and return outputs + timing."""
    if use_dualkv:
        os.environ["VLLM_USE_DUALKV"] = "1"
        os.environ["VLLM_DUALKV_MAX_DECODE_LEN"] = str(max_tokens + 16)
    else:
        os.environ.pop("VLLM_USE_DUALKV", None)

    # Reset dualkv states if switching modes
    from vllm.attention.backends.flash_attn_dualkv import reset_all_dualkv_states
    reset_all_dualkv_states()

    llm = LLM(model=model_name, max_model_len=2048,
              gpu_memory_utilization=0.8, enforce_eager=True)

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy for deterministic comparison
        max_tokens=max_tokens,
    )

    # All sequences share the same prompt
    prompts = [prompt] * num_seqs

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    texts = [o.outputs[0].text for o in outputs]
    token_counts = [len(o.outputs[0].token_ids) for o in outputs]

    return texts, token_counts, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model to test with (must have hdim=128)")
    parser.add_argument("--prompt", default=(
        "The following is a detailed explanation of how transformers work "
        "in deep learning. Transformers use self-attention mechanisms to "
        "process sequences in parallel, making them highly efficient for "
        "natural language processing tasks. The key innovation is the "
        "multi-head attention mechanism, which allows the model to attend "
        "to different parts of the input simultaneously. "
    ), help="Shared prompt for all sequences")
    parser.add_argument("--num-seqs", type=int, default=4,
                        help="Number of sequences to generate")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Max new tokens per sequence")
    parser.add_argument("--compare", action="store_true",
                        help="Run both standard FA and DualKV and compare")
    args = parser.parse_args()

    if args.compare:
        print("=" * 60)
        print("Running STANDARD FA...")
        print("=" * 60)
        std_texts, std_counts, std_time = run_generation(
            args.model, args.prompt, args.num_seqs, args.max_tokens,
            use_dualkv=False)
        print(f"  Time: {std_time:.2f}s")
        print(f"  Tokens per seq: {std_counts}")
        print(f"  Sample output: {std_texts[0][:100]}...")

        # Need to delete the LLM to free GPU memory before creating a new one
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()

        print()
        print("=" * 60)
        print("Running DUALKV...")
        print("=" * 60)
        dk_texts, dk_counts, dk_time = run_generation(
            args.model, args.prompt, args.num_seqs, args.max_tokens,
            use_dualkv=True)
        print(f"  Time: {dk_time:.2f}s")
        print(f"  Tokens per seq: {dk_counts}")
        print(f"  Sample output: {dk_texts[0][:100]}...")

        print()
        print("=" * 60)
        print("COMPARISON")
        print("=" * 60)
        match = all(s == d for s, d in zip(std_texts, dk_texts))
        print(f"  Outputs match: {match}")
        if not match:
            for i, (s, d) in enumerate(zip(std_texts, dk_texts)):
                if s != d:
                    print(f"  Seq {i} MISMATCH:")
                    print(f"    Standard: {s[:80]}...")
                    print(f"    DualKV:   {d[:80]}...")
        print(f"  Speedup: {std_time / dk_time:.2f}x")
    else:
        use_dualkv = os.environ.get("VLLM_USE_DUALKV", "0") == "1"
        mode = "DUALKV" if use_dualkv else "STANDARD FA"
        print(f"Running {mode} generation...")
        texts, counts, elapsed = run_generation(
            args.model, args.prompt, args.num_seqs, args.max_tokens,
            use_dualkv=use_dualkv)
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Tokens per seq: {counts}")
        for i, t in enumerate(texts):
            print(f"  Seq {i}: {t[:100]}...")


if __name__ == "__main__":
    main()

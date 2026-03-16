"""Comprehensive DualKV vs standard FA comparison test.

Runs standard FA and DualKV as separate processes to ensure clean state,
then compares outputs for exact match (greedy sampling).
"""
import json
import os
import subprocess
import sys
import tempfile


def run_single_config(model, prompt, num_seqs, max_tokens, use_dualkv, output_file):
    """Run a single generation config in a subprocess."""
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"
    if use_dualkv:
        env["VLLM_USE_DUALKV"] = "1"
        env["VLLM_DUALKV_MAX_DECODE_LEN"] = str(max_tokens + 32)
    else:
        env.pop("VLLM_USE_DUALKV", None)

    script = f'''
import json, sys, os
os.environ["VLLM_USE_V1"] = "0"
from vllm import LLM, SamplingParams

model = {repr(model)}
prompt = {repr(prompt)}
num_seqs = {num_seqs}
max_tokens = {max_tokens}

llm = LLM(model=model, max_model_len=2048,
          gpu_memory_utilization=0.8, enforce_eager=True,
          dtype="float16")
params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
outputs = llm.generate([prompt] * num_seqs, params)
results = []
for o in outputs:
    results.append({{
        "text": o.outputs[0].text,
        "token_ids": list(o.outputs[0].token_ids),
        "num_tokens": len(o.outputs[0].token_ids),
    }})
with open({repr(output_file)}, "w") as f:
    json.dump(results, f)
print(f"Wrote {{len(results)}} results to {repr(output_file)}")
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env, capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-500:]}")
        return None
    with open(output_file) as f:
        return json.load(f)


def main():
    model = "Qwen/Qwen2-1.5B"

    prompts = {
        "short": "Hello world",
        "medium": (
            "The following is a detailed explanation of how transformers work "
            "in deep learning. Transformers use self-attention mechanisms to "
            "process sequences in parallel, making them highly efficient for "
            "natural language processing tasks."
        ),
        "long": (
            "The following is a comprehensive overview of modern deep learning "
            "architectures and their applications in natural language processing. "
            "Transformer models have revolutionized the field by introducing "
            "self-attention mechanisms that allow the model to process all positions "
            "in a sequence simultaneously, rather than sequentially as in recurrent "
            "neural networks. The key components of a transformer include multi-head "
            "attention layers, feed-forward networks, layer normalization, and "
            "residual connections. These components work together to create powerful "
            "representations of input sequences that can be used for various downstream "
            "tasks such as machine translation, text summarization, and question answering."
        ),
    }

    configs = [
        # (prompt_name, num_seqs, max_tokens)
        ("short", 1, 20),
        ("short", 4, 20),
        ("short", 8, 20),
        ("medium", 1, 20),
        ("medium", 2, 20),
        ("medium", 4, 20),
        ("medium", 4, 30),
        ("medium", 4, 64),
        ("medium", 8, 30),
        ("long", 1, 20),
        ("long", 4, 20),
        ("long", 8, 20),
        ("long", 16, 20),
    ]

    results = []
    for prompt_name, num_seqs, max_tokens in configs:
        prompt = prompts[prompt_name]
        label = f"prompt={prompt_name:6s}, seqs={num_seqs:2d}, max_tok={max_tokens:2d}"
        print(f"\n{'='*60}")
        print(f"Config: {label}")
        print(f"{'='*60}")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            std_file = f.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            dk_file = f.name

        print("  Running standard FA...")
        std_out = run_single_config(model, prompt, num_seqs, max_tokens, False, std_file)
        if std_out is None:
            print("  [ERROR] Standard FA failed")
            results.append((label, "ERROR"))
            continue

        print("  Running DualKV...")
        dk_out = run_single_config(model, prompt, num_seqs, max_tokens, True, dk_file)
        if dk_out is None:
            print("  [ERROR] DualKV failed")
            results.append((label, "ERROR"))
            continue

        # Compare
        all_match = True
        for i in range(num_seqs):
            std_toks = std_out[i]["token_ids"]
            dk_toks = dk_out[i]["token_ids"]
            if std_toks != dk_toks:
                all_match = False
                # Find first divergence
                for j in range(min(len(std_toks), len(dk_toks))):
                    if std_toks[j] != dk_toks[j]:
                        print(f"  Seq {i}: DIVERGE at token {j}: "
                              f"std={std_toks[j]} vs dk={dk_toks[j]}")
                        print(f"    Std text: {std_out[i]['text'][:80]}")
                        print(f"    DK  text: {dk_out[i]['text'][:80]}")
                        break

        status = "PASS" if all_match else "FAIL"
        gen_count = dk_out[0]["num_tokens"]
        print(f"  [{status}] {label} (generated {gen_count} tokens)")
        results.append((label, status))

        # Cleanup
        os.unlink(std_file)
        os.unlink(dk_file)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    pass_count = sum(1 for _, s in results if s == "PASS")
    fail_count = sum(1 for _, s in results if s == "FAIL")
    err_count = sum(1 for _, s in results if s == "ERROR")
    for label, status in results:
        print(f"  [{status}] {label}")
    print(f"\n  {pass_count} PASS, {fail_count} FAIL, {err_count} ERROR out of {len(results)}")


if __name__ == "__main__":
    main()

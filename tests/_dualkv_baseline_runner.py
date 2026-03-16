"""Helper: run standard DualKV vLLM generation and output JSON result."""
import argparse
import json
import os
import sys
import time

os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_USE_DUALKV"] = "1"

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--num-seqs", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--max-decode-len", type=int, required=True)
    args = parser.parse_args()

    os.environ["VLLM_DUALKV_MAX_DECODE_LEN"] = str(args.max_decode_len)

    llm = LLM(
        model=args.model, max_model_len=2048, dtype=args.dtype,
        gpu_memory_utilization=0.8, enforce_eager=True,
        tensor_parallel_size=args.tp,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    start = time.time()
    outputs = llm.generate([args.prompt] * args.num_seqs, sp)
    elapsed = time.time() - start

    token_ids = [list(o.outputs[0].token_ids) for o in outputs]

    import torch
    mem = torch.cuda.memory_reserved() / (1024 ** 3)

    # Write result to a temp file (avoids stdout pollution)
    result = {"token_ids": token_ids, "time": elapsed, "mem": mem}
    result_file = "/tmp/dualkv_baseline_result.json"
    with open(result_file, "w") as f:
        json.dump(result, f)
    print(f"DUALKV_BASELINE_DONE:{result_file}")


if __name__ == "__main__":
    main()

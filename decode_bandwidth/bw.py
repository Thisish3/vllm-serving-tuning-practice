"""Decode memory-bandwidth utilization on one GPU.

  python bw.py memcpy            -> achievable device memory bandwidth (GB/s)
  python bw.py decode [--eager]  -> decode step time and effective bandwidth per batch size
Results are written as JSON under /work/results.
"""
import json
import os
import statistics
import sys
import time

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
OUT_DIR = "/work/results"
PROMPT_LEN = 32
OUT_TOKENS = 256
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
REPS = 3


def memcpy():
    import torch

    gib = 1024**3
    a = torch.empty(gib // 4, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    runs = []
    for _ in range(REPS):
        t = time.perf_counter()
        for _ in range(20):
            b.copy_(a)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 20
        runs.append(2 * gib / dt / 1e9)  # read + write
    return {
        "device": torch.cuda.get_device_name(),
        "copy_gbps_runs": [round(x, 1) for x in runs],
        "copy_gbps": round(statistics.median(runs), 1),
    }


def model_bytes():
    from transformers import AutoConfig

    c = AutoConfig.from_pretrained(MODEL)
    h, inter, L, v = c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.vocab_size
    hd = h // c.num_attention_heads
    kv = c.num_key_value_heads
    per_layer = h * h * 2 + h * kv * hd * 2 + 3 * h * inter + 2 * h
    elt = 2  # fp16/bf16
    weights_read = (L * per_layer + v * h + h) * elt  # layers + lm_head + final norm
    kv_per_token = 2 * L * kv * hd * elt
    return weights_read, kv_per_token


def decode(eager: bool):
    import random

    from vllm import LLM, SamplingParams

    weights, kv_tok = model_bytes()
    llm = LLM(
        model=MODEL,
        max_model_len=2048,
        gpu_memory_utilization=0.80,
        enforce_eager=eager,
        enable_prefix_caching=False,
    )
    rng = random.Random(0)

    def prompts(n):
        return [
            {"prompt_token_ids": [rng.randrange(100, 30000) for _ in range(PROMPT_LEN)]}
            for _ in range(n)
        ]

    gen = SamplingParams(max_tokens=OUT_TOKENS, ignore_eos=True, temperature=0)
    one = SamplingParams(max_tokens=1, ignore_eos=True, temperature=0)
    llm.generate(prompts(8), gen, use_tqdm=False)  # warm-up

    rows = []
    for bsz in BATCHES:
        steps = []
        for _ in range(REPS):
            p = prompts(bsz)
            t = time.perf_counter()
            llm.generate(p, one, use_tqdm=False)
            t_prefill = time.perf_counter() - t
            p = prompts(bsz)
            t = time.perf_counter()
            llm.generate(p, gen, use_tqdm=False)
            t_full = time.perf_counter() - t
            steps.append((t_full - t_prefill) / (OUT_TOKENS - 1))
        step = statistics.median(steps)
        kv_read = bsz * (PROMPT_LEN + OUT_TOKENS / 2) * kv_tok
        rows.append({
            "batch": bsz,
            "step_ms": round(step * 1e3, 3),
            "step_ms_runs": [round(s * 1e3, 3) for s in steps],
            "tok_per_s": round(bsz / step, 1),
            "bytes_per_step_gb": round((weights + kv_read) / 1e9, 4),
            "eff_gbps": round((weights + kv_read) / step / 1e9, 1),
        })
        print(rows[-1], flush=True)
    return {
        "eager": eager,
        "weights_read_gb": round(weights / 1e9, 4),
        "kv_bytes_per_token": kv_tok,
        "prompt_len": PROMPT_LEN,
        "out_tokens": OUT_TOKENS,
        "rows": rows,
    }


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    mode = sys.argv[1]
    if mode == "memcpy":
        res, name = memcpy(), "memcpy.json"
    else:
        eager = "--eager" in sys.argv
        res, name = decode(eager), f"decode-{'eager' if eager else 'graph'}.json"
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

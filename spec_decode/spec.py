"""n-gram (prompt lookup) speculative decoding on/off, offline LLM(), batch 1, greedy.

    python spec.py off|ngram            -> results/spec_<mode>.json   (5 reps per prompt)
    VLLM_TORCH_PROFILER_DIR=DIR python spec.py off|ngram prof -> torch.profiler trace of one copy-workload generate

Config keys checked in the installed vLLM 0.11.2:
  vllm/config/speculative.py  : method="ngram", num_speculative_tokens, prompt_lookup_max, prompt_lookup_min
  vllm/v1/spec_decode/metrics.py : counters vllm:spec_decode_num_drafts / _num_draft_tokens / _num_accepted_tokens
  vllm/entrypoints/llm.py     : LLM() sets disable_log_stats=True unless given, get_metrics() reads the counters
"""
import json
import os
import statistics
import sys
import time

# Leave VLLM_ENABLE_V1_MULTIPROCESSING at its default (1). With it set to 0 in vLLM 0.11.2 the
# NgramProposer returned 4 draft tokens on every step (counted by wrapping propose() in
# debug_metrics.py) but vllm:spec_decode_num_drafts stayed 0 and the step count equalled the
# output token count. Cause not identified.

import torch
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
OUT_TOKENS = 256
REPS = 5
SPEC = {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_max": 4, "prompt_lookup_min": 2}

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = open(os.path.join(HERE, "copy_source.py.txt")).read()
WORKLOADS = {
    "copy": [f"Repeat the following Python file exactly, character for character, with no explanation.\n\n```python\n{CODE}```"],
    "chat": [
        "What are some good habits for staying focused while studying for long hours?",
        "Explain how a refrigerator keeps food cold, in simple terms.",
        "I'm planning a weekend trip to the mountains. What should I pack?",
    ],
}


def counters(llm):
    want = ("vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens")
    out = {k: 0 for k in want}
    for m in llm.get_metrics():
        if m.name in want:
            out[m.name] += m.value
    return out


def main():
    mode = sys.argv[1]
    prof = len(sys.argv) > 2 and sys.argv[2] == "prof"
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    kw = dict(model=MODEL, revision=REVISION, dtype="float16", max_model_len=4096,
              gpu_memory_utilization=0.8, disable_log_stats=False, seed=0)
    if mode == "ngram":
        kw["speculative_config"] = dict(SPEC)
    llm = LLM(**kw)
    sp = SamplingParams(temperature=0.0, max_tokens=OUT_TOKENS, min_tokens=OUT_TOKENS)
    prompts = {w: [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                   for p in ps] for w, ps in WORKLOADS.items()}

    llm.generate(prompts["chat"][0], sp)  # warm-up
    if prof:
        p = prompts["copy"][0]
        llm.generate(p, sp)
        torch.cuda.synchronize()
        # vLLM's built-in torch.profiler (needs VLLM_TORCH_PROFILER_DIR); it runs inside the engine process.
        # Nsight Systems recorded no GPU kernels on this WSL2 machine, see README.
        llm.start_profile()
        t = time.perf_counter(); o = llm.generate(p, sp, use_tqdm=False); dt = time.perf_counter() - t
        llm.stop_profile()
        print(f"prof {mode} copy: {dt:.3f}s {len(o[0].outputs[0].token_ids)} tokens")
        return

    res = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "vllm": vllm.__version__,
           "model": MODEL, "revision": REVISION, "dtype": "float16", "mode": mode,
           "speculative_config": SPEC if mode == "ngram" else None,
           "sampling": {"temperature": 0.0, "max_tokens": OUT_TOKENS, "min_tokens": OUT_TOKENS}, "batch": 1,
           "reps": REPS, "workloads": {}}
    for w, ps in prompts.items():
        rows = []
        for i, p in enumerate(ps):
            runs, ids = [], None
            c0 = counters(llm)
            for _ in range(REPS):
                t = time.perf_counter(); o = llm.generate(p, sp, use_tqdm=False); dt = time.perf_counter() - t
                n = len(o[0].outputs[0].token_ids)
                assert n == OUT_TOKENS, n
                runs.append(dt)
                cur = list(o[0].outputs[0].token_ids)
                assert ids is None or ids == cur, "output changed between reps"
                ids = cur
            c1 = counters(llm)
            d = {k.split(":")[1]: c1[k] - c0[k] for k in c0}
            tps = [OUT_TOKENS / r for r in runs]
            rows.append({
                "prompt_idx": i, "prompt_tokens": len(o[0].prompt_token_ids),
                "time_s_runs": [round(r, 4) for r in runs], "time_s_median": round(statistics.median(runs), 4),
                "tok_s_runs": [round(v, 1) for v in tps], "tok_s_median": round(statistics.median(tps), 1),
                "ms_per_token_median": round(statistics.median(runs) / OUT_TOKENS * 1000, 3),
                "spec_counters_over_reps": d,
                "acceptance_rate": round(d["spec_decode_num_accepted_tokens"] / d["spec_decode_num_draft_tokens"], 4)
                if d["spec_decode_num_draft_tokens"] else None,
                "output_token_ids": ids,
                "output_text_head": o[0].outputs[0].text[:200],
            })
            print(w, i, rows[-1]["tok_s_median"], rows[-1]["tok_s_runs"], rows[-1]["acceptance_rate"], d)
        res["workloads"][w] = rows
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", f"spec_{mode}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""Debug: count drafts returned by NgramProposer.propose and the spec-decode counters, with and without min_tokens."""
import os
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # this debug run sets 0; see spec.py
from vllm import LLM, SamplingParams
from vllm.v1.spec_decode import ngram_proposer
from transformers import AutoTokenizer
from spec import MODEL, REVISION, SPEC, WORKLOADS

stats = {"calls": 0, "nonempty": 0, "draft_tokens": 0}
orig = ngram_proposer.NgramProposer.propose


def wrapped(self, *a, **k):
    out = orig(self, *a, **k)
    stats["calls"] += 1
    for d in out:
        if len(d):
            stats["nonempty"] += 1
            stats["draft_tokens"] += len(d)
    return out


ngram_proposer.NgramProposer.propose = wrapped

tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
p = tok.apply_chat_template([{"role": "user", "content": WORKLOADS["copy"][0]}], tokenize=False, add_generation_prompt=True)
llm = LLM(model=MODEL, revision=REVISION, dtype="float16", max_model_len=4096, gpu_memory_utilization=0.8,
          disable_log_stats=False, speculative_config=dict(SPEC))
for name, sp in (("min_tokens=256", SamplingParams(temperature=0, max_tokens=256, min_tokens=256)),
                 ("no min_tokens", SamplingParams(temperature=0, max_tokens=256))):
    for k in stats:
        stats[k] = 0
    o = llm.generate(p, sp, use_tqdm=False)
    c = {m.name: m.value for m in llm.get_metrics() if m.name.startswith("vllm:spec_decode_num_") and hasattr(m, "value")}
    print("DBG", name, "out_tokens", len(o[0].outputs[0].token_ids), "proposer", stats, "counters(cumulative)", c)

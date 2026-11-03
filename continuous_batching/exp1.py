"""Experiment 1 of vllm_serving_tuning_benchmark.ipynb (cells 2-5), re-run on a local GPU.

    python exp1.py [ROUNDS]            -> results/exp1_<gpu>.json
    VLLM_TORCH_PROFILER_DIR=DIR python exp1.py prof N
                                       -> torch.profiler trace of one llm.generate of N prompts

LLM(...), SamplingParams, prompts and run_batch are copied from the notebook unchanged.
The only change is that the notebook's batch-size loop is repeated ROUNDS times (default 5)
in the same process, and every round is saved.
"""
# The notebook's libcudart preload (cell 2) was a Colab workaround and is left out.
import os, time, sys, json, statistics
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# notebook sets DEBUG; kept at the notebook value only when asked, it does not change the computation
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

from vllm import LLM, SamplingParams
import torch, vllm

llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", gpu_memory_utilization=0.85, max_model_len=2048, enable_prefix_caching=True)
print("model loaded")

sp = SamplingParams(temperature=0.0, max_tokens=128, min_tokens=128)
animals = ["fox","rabbit","owl","bear","deer","wolf","otter","hawk","lynx","moose","raven","hare","badger","heron","elk","mink","crow","seal","lark","stoat","vole","toad","newt","gull","dove","finch","wren","swan","stork","crane"]
prompts_pool = [f"Write a short story about a {a} exploring a forest." for a in animals]
print(len(prompts_pool), "prompts ready")

def run_batch(n): p = prompts_pool[:n]; t0 = time.time(); outs = llm.generate(p, sp); dt = time.time() - t0; toks = sum(len(o.outputs[0].token_ids) for o in outs); return dt, toks

batch_sizes = [1, 2, 4, 8, 16, 30]

if len(sys.argv) > 1 and sys.argv[1] == "prof":
    # vLLM's built-in torch.profiler (needs VLLM_TORCH_PROFILER_DIR). Nsight Systems recorded no GPU kernels
    # on this WSL2 machine, see README.
    n = int(sys.argv[2])
    run_batch(n)  # warm-up outside the profiled call
    llm.start_profile()
    dt, toks = run_batch(n)
    llm.stop_profile()
    print(f"prof n={n} time={dt:.3f}s toks={toks}")
    sys.exit(0)

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
rounds = []
for r in range(ROUNDS):
    results = []
    for n in batch_sizes: dt, toks = run_batch(n); thpt = toks/dt; lat = dt/n; results.append((n, dt, toks, thpt, lat)); print(f"round{r} n={n:2d}  time={dt:6.2f}s  out_tokens={toks:5d}  throughput={thpt:6.1f} tok/s  avg_latency/req={lat:5.3f}s")
    rounds.append(results)

summary = []
for i, n in enumerate(batch_sizes):
    dts = [rounds[r][i][1] for r in range(ROUNDS)]
    thp = [rounds[r][i][3] for r in range(ROUNDS)]
    summary.append({"n": n, "out_tokens": rounds[0][i][2],
                    "time_s_runs": [round(v, 3) for v in dts], "time_s_median": round(statistics.median(dts), 3),
                    "tok_s_runs": [round(v, 1) for v in thp], "tok_s_median": round(statistics.median(thp), 1),
                    "latency_per_req_s_median": round(statistics.median(dts) / n, 3)})
    print(summary[-1])

dev = torch.cuda.get_device_name()
os.makedirs("results", exist_ok=True)
fn = "results/exp1_" + dev.replace("NVIDIA GeForce ", "").replace(" ", "_").lower() + ".json"
json.dump({"gpu": dev, "torch": torch.__version__, "torch_cuda": torch.version.cuda, "vllm": vllm.__version__,
           "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "model_revision": os.environ.get("MODEL_REVISION"),
           "llm_args": {"gpu_memory_utilization": 0.85, "max_model_len": 2048, "enable_prefix_caching": True},
           "sampling": {"temperature": 0.0, "max_tokens": 128, "min_tokens": 128},
           "rounds": ROUNDS, "note": "rounds run back to back in one process; round 0 is the first generate after model load, as in the notebook",
           "results": summary}, open(fn, "w"), indent=1)
print("wrote", fn)

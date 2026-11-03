"""Per-step GPU numbers from a vLLM torch.profiler trace (VLLM_TORCH_PROFILER_DIR, llm.start_profile/stop_profile).

    python prof_steps.py TRACE.json.gz OUT.json

Nsight Systems recorded no GPU kernels on this WSL2 machine (see README), so this reads the CUPTI
kernel/memcpy/memset events that torch.profiler records instead.

Step boundaries: every engine step ends with a device-to-host copy of the sampled tokens
("Memcpy DtoH"). Step k = from the first GPU event after the previous DtoH copy to the end of its own DtoH copy.
  kernels           kernel events in the step
  span_us           first GPU event start -> DtoH end
  busy_us           union of kernel/memcpy/memset intervals inside the step
  idle_in_step_us   span_us - busy_us
  gap_before_us     previous step's DtoH end -> this step's first GPU event (GPU has nothing queued)
Step 0 contains the prefill; "decode" statistics use steps 1..N-1.
"""
import collections
import gzip
import json
import statistics
import sys


def load(path):
    ev = json.load(gzip.open(path))["traceEvents"]
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    return sorted(gpu, key=lambda e: e["ts"])


def union(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def steps(gpu):
    out, cur = [], []
    for e in gpu:
        cur.append(e)
        if e["cat"] == "gpu_memcpy" and "DtoH" in e["name"]:
            out.append(cur)
            cur = []
    return out


def summarize(vals):
    return {"median": round(statistics.median(vals), 2), "min": round(min(vals), 2), "max": round(max(vals), 2)}


def main():
    gpu = load(sys.argv[1])
    st = steps(gpu)
    rows = []
    prev_end = None
    for s in st:
        start = s[0]["ts"]
        end = s[-1]["ts"] + s[-1]["dur"]
        busy = union([(e["ts"], e["ts"] + e["dur"]) for e in s])
        rows.append({
            "kernels": sum(e["cat"] == "kernel" for e in s),
            "memcpy": sum(e["cat"] == "gpu_memcpy" for e in s),
            "span_us": end - start, "busy_us": busy, "idle_in_step_us": end - start - busy,
            "gap_before_us": (start - prev_end) if prev_end is not None else None,
        })
        prev_end = end
    dec = rows[1:]
    kern = [e for e in gpu if e["cat"] == "kernel"]
    ktot = sum(e["dur"] for e in kern)
    by_name = collections.defaultdict(lambda: [0, 0.0])
    for e in kern:
        by_name[e["name"]][0] += 1
        by_name[e["name"]][1] += e["dur"]
    top = sorted(by_name.items(), key=lambda kv: -kv[1][1])[:5]
    first, last = gpu[0]["ts"], gpu[-1]["ts"] + gpu[-1]["dur"]
    res = {
        "trace": sys.argv[1].split("/")[-1],
        "steps": len(rows),
        "gpu_window_us": round(last - first, 1),
        "gpu_busy_us": round(union([(e["ts"], e["ts"] + e["dur"]) for e in gpu]), 1),
        "kernel_time_us": round(ktot, 1),
        "decode_steps": {
            "kernels_per_step": summarize([r["kernels"] for r in dec]),
            "kernels_per_step_distinct_values": sorted(collections.Counter(r["kernels"] for r in dec).items()),
            "span_us": summarize([r["span_us"] for r in dec]),
            "busy_us": summarize([r["busy_us"] for r in dec]),
            "idle_in_step_us": summarize([r["idle_in_step_us"] for r in dec]),
            "gap_before_us": summarize([r["gap_before_us"] for r in dec]),
        },
        "prefill_step": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in rows[0].items()},
        "top5_kernels": [{"name": n, "calls": c, "total_us": round(t, 1), "pct_of_kernel_time": round(100 * t / ktot, 1)}
                         for n, (c, t) in top],
    }
    print(json.dumps({k: v for k, v in res.items() if k != "top5_kernels"}, indent=1))
    for t in res["top5_kernels"]:
        print(f"{t['pct_of_kernel_time']:5.1f}%  {t['total_us']:9.1f}us  {t['calls']:6d}  {t['name'][:110]}")
    json.dump(res, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()

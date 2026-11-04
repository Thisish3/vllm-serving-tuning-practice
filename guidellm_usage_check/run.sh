#!/bin/bash
# Runs guidellm main and the PR branch against the same vLLM server through proxy.py.
# Usage: run.sh [runs]   (WSL, as root; expects the three venvs below)
set -u
RUNS=${1:-5}
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=${WORK:-/root/guidellm_check}
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
export HF_HOME=/data/hf-cache HF_HUB_OFFLINE=1
mkdir -p "$WORK"
PROXY=

/root/venv-vllm0112/bin/vllm serve $MODEL --port 8000 --gpu-memory-utilization 0.8 > "$WORK/vllm.log" 2>&1 &
VLLM=$!
trap 'kill $VLLM 2>/dev/null; kill $PROXY 2>/dev/null; pkill -f "vllm serve" 2>/dev/null' EXIT
for _ in $(seq 180); do curl -sf localhost:8000/health >/dev/null && break; sleep 1; done
curl -sf localhost:8000/health >/dev/null || { echo "vLLM did not start"; tail -20 "$WORK/vllm.log"; exit 1; }

for i in $(seq "$RUNS"); do
  for v in main pr; do
    rm -f "$WORK/${v}_$i.proxy.jsonl"
    PROXY_LOG="$WORK/${v}_$i.proxy.jsonl" /root/venv-vllm0112/bin/python "$HERE/proxy.py" 8001 &
    PROXY=$!
    for _ in $(seq 30); do curl -sf localhost:8001/health >/dev/null && break; sleep 0.5; done
    /root/venv-guidellm-$v/bin/guidellm run \
      --backend kind=openai_http,target=http://127.0.0.1:8001,model=$MODEL \
      --profile kind=synchronous --constraint kind=max_requests,count=20 \
      --data kind=synthetic_text,prompt_tokens=64,output_tokens=256 \
      --tokenizer kind=hf_auto,model=$MODEL \
      --output kind=json,path="$WORK/${v}_$i.json" --disable-console-interactive \
      > "$WORK/${v}_$i.log" 2>&1
    echo "$v run $i exit=$? proxy_lines=$(wc -l < "$WORK/${v}_$i.proxy.jsonl")"
    kill $PROXY; wait $PROXY 2>/dev/null
  done
done

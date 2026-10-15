#!/bin/bash
# Run bw.py inside the vllm/vllm-openai:v0.11.2 image (already in k3s containerd) on the GPU.
W=$(cd "$(dirname "$0")" && pwd)
IMG=docker.io/vllm/vllm-openai:v0.11.2
K="k3s kubectl"
systemctl start k3s
for i in $(seq 1 60); do $K get nodes 2>/dev/null | grep -q ' Ready' && break; sleep 2; done
# keep the GPU free: the deployment would otherwise come back with replicas=1
$K scale deployment vllm-3080 --replicas=0 >/dev/null 2>&1
for i in $(seq 1 60); do [ -z "$($K get pods -l app=vllm-3080 --no-headers 2>/dev/null)" ] && break; sleep 2; done
echo "[$(date +%H:%M:%S)] GPU free, C free $(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9)G"

run() {  # $1 = name, rest = args to bw.py
  local n=$1; shift
  k3s ctr -n k8s.io run --rm --net-host \
    --runc-binary /usr/bin/nvidia-container-runtime \
    --env NVIDIA_VISIBLE_DEVICES=all --env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    --env HF_HUB_OFFLINE=1 --env VLLM_LOGGING_LEVEL=WARNING \
    --mount type=bind,src=/data/hf-cache,dst=/root/.cache/huggingface,options=rbind:rw \
    --mount type=bind,src=/dev/shm,dst=/dev/shm,options=rbind:rw \
    --mount type=bind,src=$W,dst=/work,options=rbind:rw \
    $IMG bw-$n python3 /work/bw.py "$@" > $W/log-$n.txt 2>&1
  echo "[$(date +%H:%M:%S)] $n exit=$?"
}
run memcpy memcpy
run graph decode
run eager decode --eager
systemctl stop k3s
echo "[$(date +%H:%M:%S)] DONE"

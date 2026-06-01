#!/usr/bin/env zsh
set -euo pipefail

repo_dir="/home/eika/Documents/Code/Github/hopfield_sequential"
log_dir="$repo_dir/results/logs"
log_file="$log_dir/exp07_tomita_full.log"
unit_name="exp07-tomita-full"

mkdir -p "$log_dir"
systemctl --user reset-failed "$unit_name.service" >/dev/null 2>&1 || true
: > "$log_file"

systemd-run \
  --user \
  --unit="$unit_name" \
  --collect \
  --working-directory="$repo_dir" \
  --property=MemoryMax=22G \
  --property=MemorySwapMax=2G \
  --property="StandardOutput=append:$log_file" \
  --property="StandardError=append:$log_file" \
  /usr/bin/env PYTHONUNBUFFERED=1 /usr/bin/python \
  -m experiments.exp07_structured_grammar_learning \
  --full \
  --task tomita \
  --with-hopfield \
  --with-topk \
  --jobs 0 \
  --max-workers 0 \
  --total-memory-gb 22

echo "Started $unit_name.service"
echo "Log: $log_file"
echo "Status: systemctl --user status $unit_name.service"

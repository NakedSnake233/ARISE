#!/usr/bin/env bash
set -euo pipefail

dataset="${1:?dataset required}"
source_commit="${2:?source commit required}"
device="${3:-cuda}"
case "$dataset" in
  tt|aiops) ;;
  *) echo "unsupported dataset: $dataset" >&2; exit 2 ;;
esac
case "$device" in
  cuda|cpu) ;;
  *) echo "unsupported device: $device" >&2; exit 2 ;;
esac

workdir=/workspace/arise-exp-revision-20260929
cd "$workdir"
mkdir -p logs
log="logs/${dataset}_fold0_seed42_context_smoke_${device}.log"
exit_file="logs/${dataset}_fold0_seed42_context_smoke_${device}.exit"
if [[ -e "$log" || -e "$exit_file" ]]; then
  echo "refusing to overwrite an existing remote smoke log" >&2
  exit 2
fi

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export ARISE_SOURCE_COMMIT="$source_commit" ARISE_SOURCE_DIRTY=true
set +e
.venv/bin/python -m exp_revision.train_context --dataset "$dataset" --fold 0 --seed 42 --smoke \
  --device "$device" --output "results_v2/T1_context_smoke_remote_${device}/${dataset}/fold0_seed42" >"$log" 2>&1
status=$?
set -e
printf '%s\n' "$status" >"$exit_file"
exit "$status"

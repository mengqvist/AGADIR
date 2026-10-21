#!/bin/bash
# Progress of every MD run: last progress line per run, and GPU load.
cd "$(dirname "$0")/../runs"
for f in */*/progress.log; do printf "%-28s %s\n" "${f%/progress.log}" "$(tail -1 "$f")"; done 2>/dev/null
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

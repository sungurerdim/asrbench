#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
python scripts/benchmarks/eval_optimizer.py > scripts/benchmarks/run.log 2>&1

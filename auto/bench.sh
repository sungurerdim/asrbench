#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python auto/eval_optimizer.py > auto/run.log 2>&1

#!/bin/bash
set -e
cd "$(dirname "$0")/.."
python auto/eval_wer.py > auto/run.log 2>&1

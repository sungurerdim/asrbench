# AutoTune: asrbench — optimizer throughput

## Objective
Improve IAMSOptimizer.run() orchestration throughput on a 5-param convex landscape.

## Metric
- Primary: latency_ms (lower is better)
- Secondary: pass_rate (monitoring only, must stay 1.0)

## Files
| File | Permission | Purpose |
|------|-----------|---------|
| asrbench/engine/optimizer.py | EDITABLE | Optimization target |
| auto/bench.sh | read-only | Evaluation harness |
| auto/eval_optimizer.py | read-only | Metric extraction |
| auto/.autotune.json | read-only | Configuration |
| auto/results.tsv | append-only | Experiment log |
| All other files | read-only | Keep unchanged |

## Experiment Loop

Repeat forever:

1. Read and analyze asrbench/engine/optimizer.py. What could improve latency_ms?
2. Form a hypothesis. One change per experiment.
3. Edit asrbench/engine/optimizer.py with your experimental idea.
4. Commit: git add asrbench/engine/optimizer.py && git commit -m "description"
5. Run: bash auto/bench.sh
6. Read results: grep "^latency_ms:" auto/run.log
7. Append to auto/results.tsv
8. Decision:
   - latency_ms improved (lower) -> KEEP
   - latency_ms same or worse -> DISCARD (git reset HEAD~1 --hard)
9. Go to step 1.

## Rules
1. ONLY modify asrbench/engine/optimizer.py.
2. Only use packages already in the project.
3. Each experiment must complete within 60 seconds.
4. Simplicity criterion applies.
5. If pass_rate drops below 1.0, DISCARD.

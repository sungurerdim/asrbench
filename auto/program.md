# AutoTune: asrbench

## Objective
Improve WER engine compute() latency on a 100-segment English corpus.

## Metric
- Primary: latency_ms (lower is better)
- Secondary: pass_rate (monitoring only, must stay 1.0 — correctness guard)

## Files
| File | Permission | Purpose |
|------|-----------|---------|
| asrbench/engine/wer.py | EDITABLE | Optimization target |
| auto/bench.sh | read-only | Evaluation harness |
| auto/eval_wer.py | read-only | Metric extraction |
| auto/.autotune.json | read-only | Configuration |
| auto/results.tsv | append-only | Experiment log |
| All other files | read-only | Keep unchanged |

## Baseline
- latency_ms: TBD (measured in Phase 5)
- commit: TBD

## Experiment Loop

Repeat forever:

1. Read and analyze asrbench/engine/wer.py. What could improve latency_ms?
2. Form a hypothesis. Think about what change might help.
3. Edit asrbench/engine/wer.py with your experimental idea.
4. Commit: git add asrbench/engine/wer.py && git commit -m "description of change"
5. Run: bash auto/bench.sh
6. Read results: grep "^latency_ms:" auto/run.log
7. Append to auto/results.tsv (tab-separated):
   <ISO8601_timestamp>\t<commit_7char>\t<status>\t<latency_ms_value>\t<pass_rate_value>\t<HH:MM:SS>\t<description>
8. Decision:
   - latency_ms improved (lower) -> KEEP. Branch advances.
   - latency_ms same or worse -> DISCARD. Run: git reset HEAD~1 --hard
9. Go to step 1. Continue without interruption.

## Rules

1. ONLY modify asrbench/engine/wer.py. Everything else is read-only.
2. Only use packages and dependencies already in the project.
3. Each experiment must complete within 60 seconds.
   If exceeded, kill and treat as crash.
4. Simplicity criterion: a small improvement that adds ugly complexity is NOT worth it.
5. Crash handling:
   - Simple bug (typo, import) -> fix and retry. Log only as retry, keep crash for fundamental failures.
   - Fundamental problem -> skip, log as crash, move on.
   - For crashes: latency_ms=0.000000, status=crash in results.tsv
6. Continue without interruption. Keep experimenting autonomously.
   If stuck, re-read the target file for new angles, combine previous ideas,
   or try more radical approaches.
7. Only attempt experiments with new hypotheses — skip previously discarded approaches.
   Read results.tsv descriptions to avoid duplicates.
8. If pass_rate drops below 1.0, the experiment MUST be discarded — correctness is non-negotiable.

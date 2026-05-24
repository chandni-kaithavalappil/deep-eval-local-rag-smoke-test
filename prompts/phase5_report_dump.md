# Phase 5 Report Dump Prompt

PHASE 5: Print everything needed to update the article after both handcrafted and synthesized evals complete.

Run:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" uv run --no-sync python -c "
import json
from collections import defaultdict
hand = json.load(open('results/handcrafted_scores.json'))
synth = json.load(open('results/synthesized_scores.json'))
def agg(data):
    a = defaultdict(list)
    for d in data:
        for m,s in d['scores'].items():
            if s is not None:
                a[m].append(s)
    return a
print('=== HANDCRAFTED ===')
for m,v in agg(hand).items():
    print(f'{m:30s} avg={sum(v)/len(v):.3f}  pass={sum(1 for x in v if x>=0.7)}/{len(v)}')
print('\\n=== SYNTHESIZED ===')
for m,v in agg(synth).items():
    print(f'{m:30s} avg={sum(v)/len(v):.3f}  pass={sum(1 for x in v if x>=0.7)}/{len(v)}')
print('\\n=== ADVERSARIAL CASES (handcrafted) ===')
for d in hand:
    q = d['input'].lower()
    if 'opus 5' in q or 'employees' in q or '2027' in q:
        print(f'\\nQ: {d[\"input\"]}\\nA: {d[\"actual\"][:400]}\\nScores: {d[\"scores\"]}')
"
```

Paste the full stdout into the chat and update `learner.md` with the final model path and result file locations.

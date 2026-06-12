"""FAST (no-CSP) headroom check for the purely-stateless concat/rconcat subset.
For each cryptarithm problem: for every operator glyph, test if its example results are
a fixed symbol-arrangement of the operands (concat/rconcat, with per-operand or whole-
result reversal = the swap preprocessings). If the QUERY's glyph is such a passthrough
op, we can answer by symbol-copy ALONE (no digits, no search). Cross with baseline
correctness -> how many concat/rconcat queries the model currently fails (recoverable
by the cleanest possible stateless probe)."""
from __future__ import annotations
import json, glob, csv, re
from collections import Counter
from pathlib import Path
PROJECT = Path("/scratch/jwj/NemotronFT")

csv.field_size_limit(10**7)
prompts = {}
for r in csv.DictReader(open(PROJECT / "resource/winning-solution/train.csv")):
    prompts[r["id"]] = r["prompt"]

LINE = re.compile(r"^(.)(.)(.)(.)(.)\s*=\s*(\S+)\s*$")

def arrangements(a0, a1, b0, b1):
    # concat/rconcat × {operand-swap} × {result-reverse}; return dict name->4-tuple
    base = {
        "concat":  (a0, a1, b0, b1),
        "rconcat": (b0, b1, a0, a1),
        "concat_swapops":  (a1, a0, b1, b0),
        "rconcat_swapops": (b1, b0, a1, a0),
    }
    out = {}
    for name, t in base.items():
        out[name] = t
        out[name + "_revres"] = tuple(reversed(t))
    return out

def parse(prompt):
    facts = []
    for line in prompt.splitlines():
        line = line.strip()
        m = LINE.match(line)
        if m and " = " in line:
            a0, a1, op, b0, b1, res = m.groups()
            facts.append((a0, a1, op, b0, b1, res))
    qm = re.search(r"determine the result for:\s*(.)(.)(.)(.)(.)\s*$", prompt)
    query = qm.groups() if qm else None  # (a0,a1,op,b0,b1)
    return facts, query

stats = Counter()
recoverable = 0
concat_total = 0
for shard in sorted(glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*of4.jsonl"))):
    for line in open(shard):
        if not line.strip():
            continue
        r = json.loads(line)
        if not r["category"].startswith("cryptarithm") or r["id"] not in prompts:
            continue
        facts, query = parse(prompts[r["id"]])
        if not query:
            continue
        qa0, qa1, qop, qb0, qb1 = query
        # for the query's glyph, find a passthrough pattern consistent across its examples
        ex = [f for f in facts if f[2] == qop]
        if not ex:
            continue
        good_pattern = None
        for pname in arrangements("a", "b", "c", "d"):  # iterate names
            ok = True
            for (a0, a1, op, b0, b1, res) in ex:
                want = arrangements(a0, a1, b0, b1)[pname]
                if tuple(res) != want:
                    ok = False
                    break
            if ok and ex:
                good_pattern = pname
                break
        if good_pattern is None:
            continue
        # query IS a passthrough op -> predict by the same pattern
        concat_total += 1
        pred = "".join(arrangements(qa0, qa1, qb0, qb1)[good_pattern])
        true = str(r["answer"])
        base_ok = bool(r["correct"])
        match = (pred == true)
        stats[(good_pattern, base_ok, match)] += 1
        if match and not base_ok:
            recoverable += 1

print(f"=== concat/rconcat (pure symbol-passthrough) query subset ===")
print(f"queries detected as passthrough-op: {concat_total}")
print(f"  RECOVERABLE (passthrough-predict matches truth AND baseline wrong): {recoverable}"
      f"  = {recoverable/9500*100:.2f}pp of benchmark")
print("\nbreakdown (pattern, baseline_correct, passthrough_matches_truth) -> count:")
for k in sorted(stats, key=lambda x: -stats[x]):
    print(f"  {k}: {stats[k]}")

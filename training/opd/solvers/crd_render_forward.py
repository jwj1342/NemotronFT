"""Strict-forward 'feature-elimination' CoT renderer (expert's design) — v1: the clean,
validated-recoverable concat/rconcat subset. Each step is a LOCAL structural assertion
on SYMBOLS (no digit guessing, no enumeration, no held state): scan the operator's
signature -> if results are the operands copied in a fixed arrangement, the operator is
(r)concat -> apply by copying symbols. This is exactly the stateless symbol-matching the
model already does at 99% on text-cipher.

render_concat(prompt, answer) -> (cot, pred) or None if the query op isn't passthrough.
"""
from __future__ import annotations
import re
from typing import Optional, Tuple

LINE = re.compile(r"^(.)(.)(.)(.)(.)\s*=\s*(\S+)\s*$")


def _arrangements(a0, a1, b0, b1):
    base = {"concat": (a0, a1, b0, b1), "rconcat": (b0, b1, a0, a1),
            "concat_swapops": (a1, a0, b1, b0), "rconcat_swapops": (b1, b0, a1, a0)}
    out = {}
    for name, t in base.items():
        out[name] = t
        out[name + "_revres"] = tuple(reversed(t))
    return out


_HUMAN = {"concat": "the two operands written left-to-right",
          "rconcat": "the two operands written right-then-left",
          "concat_swapops": "each operand's two symbols swapped, then joined left-to-right",
          "rconcat_swapops": "each operand's two symbols swapped, then joined right-then-left"}


def _parse(prompt: str):
    facts = []
    for line in prompt.splitlines():
        line = line.strip()
        m = LINE.match(line)
        if m and " = " in line:
            facts.append(m.groups())  # (a0,a1,op,b0,b1,res)
    qm = re.search(r"determine the result for:\s*(.)(.)(.)(.)(.)\s*$", prompt)
    return facts, (qm.groups() if qm else None)


def render_concat(prompt: str, answer: Optional[str] = None) -> Optional[Tuple[str, str]]:
    facts, query = _parse(prompt)
    if not query:
        return None
    qa0, qa1, qop, qb0, qb1 = query
    ex = [f for f in facts if f[2] == qop]
    if not ex:
        return None
    pattern = None
    for pname in _arrangements("a", "b", "c", "d"):
        if all(tuple(res) == _arrangements(a0, a1, b0, b1)[pname]
               for (a0, a1, _op, b0, b1, res) in ex):
            pattern = pname
            break
    if pattern is None:
        return None
    label = pattern.replace("_revres", "").replace("concat_swapops", "rconcat").replace("rconcat_swapops", "rconcat")
    label = "concat" if label.startswith("concat") else "rconcat"
    pred = "".join(_arrangements(qa0, qa1, qb0, qb1)[pattern])

    lines = []
    lines.append(f"I check operator '{qop}' by its structural signature, copying symbols only "
                 f"(no digit guessing, no search).")
    for (a0, a1, _op, b0, b1, res) in ex[:4]:
        want = "".join(_arrangements(a0, a1, b0, b1)[pattern])
        lines.append(f"  {a0}{a1}{qop}{b0}{b1} = {res}  → result = {_HUMAN[pattern]} ({want}). matches.")
    lines.append(f"So '{qop}' is a {label} (symbol passthrough): the result is {_HUMAN[pattern]}. "
                 f"No arithmetic or symbol→digit mapping is needed.")
    lines.append(f"Apply to {qa0}{qa1}{qop}{qb0}{qb1}: copy symbols → {pred}.")
    cot = "\n".join(lines)
    return cot, pred


if __name__ == "__main__":
    import json, glob, csv
    from pathlib import Path
    PROJECT = Path("/scratch/jwj/NemotronFT")
    csv.field_size_limit(10**7)
    prompts = {r["id"]: r["prompt"] for r in csv.DictReader(open(PROJECT / "resource/winning-solution/train.csv"))}
    shown = 0
    for shard in sorted(glob.glob(str(PROJECT / "output/eval_full_huikang-baseline-converted_shard*of4.jsonl"))):
        for line in open(shard):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r["category"].startswith("cryptarithm") or r["id"] not in prompts:
                continue
            out = render_concat(prompts[r["id"]], str(r["answer"]))
            if out and not r["correct"]:   # show a RECOVERABLE one (baseline wrong)
                cot, pred = out
                print("=" * 80)
                print(f"id={r['id']}  true={r['answer']}  pred={pred}  match={pred==str(r['answer'])}  baseline=WRONG")
                print("PROMPT:\n" + prompts[r["id"]][:400])
                print("--- STRICT-FORWARD CoT ---\n<think>\n" + cot + "\n</think>\n\\boxed{" + pred + "}")
                shown += 1
                if shown >= 3:
                    break
        if shown >= 3:
            break

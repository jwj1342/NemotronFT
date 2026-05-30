"""Convert Flash-COT-Filtered/*/level_1.jsonl → single pool.jsonl for rollout.py.

Output schema matches training/rl/pool.py:
    {id, prompt, gt, category}
where category is the canonical 9-class label from eval/category.py
(not the Flash-COT folder name, which uses a different taxonomy).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "eval"))
from category import detect_category  # type: ignore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flashcot_dir", type=Path,
                    default=PROJECT_ROOT / "data" / "Flash-COT-Filtered")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    out_rows: list[dict] = []
    per_cat: dict[str, int] = {}
    n_unknown = 0
    for jsonl in sorted(args.flashcot_dir.rglob("*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cat = detect_category(rec["prompt"])
                if cat == "unknown":
                    n_unknown += 1
                    continue
                out_rows.append({
                    "id": str(rec["id"]),
                    "prompt": rec["prompt"],
                    "gt": str(rec["answer"]),
                    "category": cat,
                })
                per_cat[cat] = per_cat.get(cat, 0) + 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    print(f"[flashcot-pool] wrote {len(out_rows)} prompts to {args.out}")
    if n_unknown:
        print(f"[flashcot-pool] dropped {n_unknown} unknown-category prompts")
    print("[flashcot-pool] per-category:")
    for c, n in sorted(per_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {c:30s} {n:5d}")


if __name__ == "__main__":
    main()

"""LongBench table straight from the per-sample predictions (results/<tag>/longbench/<method>/<task>.jsonl).

Scores every complete (method, task) file with the official metrics; a task enters a
method's average only when all its samples are done, and the "common" average uses only
tasks every listed method has finished, so rows are comparable.

  python scripts/longbench_table.py --tag llama31_8b_inst [--methods bf16,diag-wo-alpha:starq.sink4.rw128@0.85,...]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from run_longbench import TASKS, GROUP, longbench_files, score

ROOT = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--methods", default="", help="rows, in order (default: every method directory)")
    p.add_argument("--dir", default="longbench")
    p.add_argument("--limit", type=int, default=0, help="runs made with --limit N: a task is complete at N samples")
    args = p.parse_args()
    cache = os.path.join(ROOT, "longbench_data")
    _, _, metrics = longbench_files(cache)
    base = os.path.join(ROOT, args.tag, args.dir)
    methods = [m.replace(":", "__") for m in args.methods.split(",") if m] or sorted(os.listdir(base))
    methods = [m for m in methods if os.path.isdir(os.path.join(base, m))]
    n_exp = {t: sum(1 for _ in open(os.path.join(cache, "data", f"{t}.jsonl"))) for t in TASKS}
    if args.limit:
        n_exp = {t: min(n, args.limit) for t, n in n_exp.items()}
    table = {}
    for m in methods:
        row = {}
        for t in TASKS:
            f = os.path.join(base, m, f"{t}.jsonl")
            if not os.path.exists(f):
                continue
            recs = [json.loads(l) for l in open(f)]
            if len({r.get("idx", j) for j, r in enumerate(recs)}) >= n_exp[t]:
                row[t] = score(t, recs, metrics)
        table[m] = row
    common = [t for t in TASKS if all(t in table[m] for m in methods)]
    head = "| method | " + " | ".join(common) + " | avg (common) |"
    print(head)
    print("|" + "---|" * (len(common) + 2))
    for m in methods:
        r = table[m]
        avg = sum(r[t] for t in common) / len(common) if common else float("nan")
        print(f"| {m.replace('__', ':')} | " + " | ".join(f"{r[t]:.2f}" for t in common) + f" | {avg:.2f} |")
    groups = {g: [t for t in ts if t in common] for g, ts in GROUP.items()}
    print("\nby category (common tasks only):")
    for m in methods:
        print(f"  {m.replace('__', ':'):42s} " + "  ".join(
            f"{g}={sum(table[m][t] for t in ts) / len(ts):.2f}" for g, ts in groups.items() if ts))
    json.dump(dict(table=table, common=common), open(os.path.join(base, "table.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

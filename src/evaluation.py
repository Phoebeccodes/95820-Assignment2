from __future__ import annotations
import os, json, argparse, statistics as stats
from typing import List, Dict, Any

from .utils import read_jsonl, f1_em, get_logger

logger = get_logger("eval")

def compute(pred_path: str, gold_path: str):
    preds = read_jsonl(pred_path)
    gold_rows = {int(r["id"]): r for r in read_jsonl(gold_path)}
    f1s, ems = [], []
    for p in preds:
        gid = int(p["id"])
        if gid not in gold_rows:
            continue
        gold = gold_rows[gid]["answer"]
        f1, em = f1_em(p["pred_answer"], gold)
        f1s.append(f1)
        ems.append(em)
    out = {
        "n": len(f1s),
        "f1_mean": sum(f1s)/len(f1s) if f1s else 0.0,
        "em_mean": sum(ems)/len(ems) if ems else 0.0,
        "f1_ci95": _ci95(f1s),
        "em_ci95": _ci95(ems),
    }
    return out

def _ci95(values: List[float]) -> float:
    if not values:
        return 0.0
    m = sum(values)/len(values)
    if len(values) < 2:
        return 0.0
    stdev = stats.pstdev(values)  # population stdev for simplicity
    return 1.96 * (stdev / (len(values) ** 0.5))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", required=True)
    ap.add_argument("--gold_path", required=True)
    ap.add_argument("--out_path", default=None)
    args = ap.parse_args()

    metrics = compute(args.pred_path, args.gold_path)
    out_path = args.out_path or os.path.splitext(args.pred_path)[0] + "_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Wrote metrics to {out_path}\n{json.dumps(metrics, indent=2)}")

if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm

from .utils import (
    get_logger,
    load_corpus,
    load_gold,
    Embedder,
    RagIndex,
    GeneratorLLM,
    build_prompt,
    write_jsonl,
)

logger = get_logger("phase4")

def build_context_from_hits(hits: List[Dict[str, Any]], strategy: str, gold_answer: str = "") -> str:
    """
    Turn a list of retrieved hits into the context string shown to the generator.

    Args:
        hits (List[Dict]): Retrieved passages with metadata.
        strategy (str): Strategy for building context.
            - 'first':       use the highest-ranked passage only
            - 'concat':      concatenate top-K passages (stop if too long)
            - 'best_by_len': choose the passage with length closest to gold answer
        gold_answer (str, optional): Gold reference answer (used by 'best_by_len').

    Returns:
        str: Context text to insert into the prompt.
    """
    if not hits:
        return ""

    s = (strategy or "first").lower()

    if s == "first":
        return hits[0]["text"]

    if s == "concat":
        # Concatenate until soft character cap
        parts, total = [], 0
        for h in hits:
            t = (h.get("text") or "").strip()
            if not t:
                continue
            total += len(t)
            if total > 4000:  # guardrail for prompt size
                break
            parts.append(t)
        return "\n\n".join(parts) if parts else hits[0]["text"]

    if s == "best_by_len":
        # Select passage closest in length to gold answer
        gl = len((gold_answer or "").split())
        if gl <= 0:
            return hits[0]["text"]
        idx = min(
            range(len(hits)),
            key=lambda i: abs(len((hits[i].get("text") or "").split()) - gl),
        )
        return hits[idx]["text"]

    # Fallback
    return hits[0]["text"]


def _compute_squad_metrics_from_jsonl(pred_jsonl: str, gold_jsonl: str) -> Dict[str, Any]:
    """
    Compute Exact Match (EM) and F1 metrics using Hugging Face's 'evaluate' SQuAD metric.

    Args:
        pred_jsonl (str): Path to predictions JSONL.
        gold_jsonl (str): Path to gold standard JSONL.

    Returns:
        Dict[str, Any]: Metric dictionary with keys:
            - n (int): number of evaluated samples
            - exact_match (float)
            - f1 (float)
    """
    try:
        import evaluate  # lazy import to keep module load fast
    except Exception as e:
        raise RuntimeError("Please 'pip install evaluate' to run Phase 4 metrics.") from e

    squad = evaluate.load("squad")
    preds, refs, gold_map = [], [], {}

    # Load gold answers
    with open(gold_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold_map[str(r["id"])] = r["answer"]

    # Load predictions
    with open(pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            _id = str(r["id"])
            if _id in gold_map:
                preds.append({"id": _id, "prediction_text": r.get("pred_answer", "")})
                refs.append({"id": _id, "answers": {"text": [gold_map[_id]], "answer_start": [0]}})

    if not preds:
        return {"n": 0, "exact_match": 0.0, "f1": 0.0}

    m = squad.compute(predictions=preds, references=refs)
    m["n"] = len(preds)
    return m


def run_phase4_grid(
    corpus_path: str,
    gold_path: str,
    embedding_models: List[str],
    top_ks: List[int],
    selections: List[str],
    prompt_styles: List[str],
    out_dir: str = "results",
    max_examples: int = 0,  # 0 = use all
):
    """
    Run the full Step-4 parameter sweep and persist predictions, metrics, and summary CSVs.

    Args:
        corpus_path (str): Path to corpus JSONL.
        gold_path (str): Path to gold standard JSONL.
        embedding_models (List[str]): List of SentenceTransformer models to test.
        top_ks (List[int]): List of K values for retrieval.
        selections (List[str]): Context selection strategies ('first', 'concat', 'best_by_len').
        prompt_styles (List[str]): Prompting styles ('instruction', 'persona', 'cot').
        out_dir (str, optional): Output directory for results. Defaults to "results".
        max_examples (int, optional): Limit number of evaluation examples (0 = all).

    Output files:
        - predictions JSONL per run
        - metrics JSON per run
        - phase4_parameter_sweep.csv (summary table)
        - comparison_analysis.csv (merged results across systems/phases)
    """
    # Prepare data
    corpus = load_corpus(corpus_path)
    gold = load_gold(gold_path)
    if max_examples and max_examples > 0:
        gold = gold.iloc[:max_examples].copy()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    # Iterate over embedding models
    for model_name in embedding_models:
        embedder = Embedder(model_name=model_name)
        index = RagIndex(embedder=embedder, corpus_df=corpus)
        index.build()

        # Try to fetch embedding dimension
        try:
            emb_dim = getattr(embedder.model, "get_sentence_embedding_dimension", lambda: None)()
        except Exception:
            emb_dim = None

        # Sweep parameters
        for K in top_ks:
            for sel in selections:
                for style in prompt_styles:
                    safe_model = re.sub(r"[^A-Za-z0-9_-]+", "-", model_name)
                    tag_base = f"phase4_{safe_model}_k{K}_{sel}_{style}"
                    pred_path = f"{out_dir}/{tag_base}.jsonl"
                    metrics_path = f"{out_dir}/{tag_base}_metrics.json"

                    logger.info(f"[Phase4] model={model_name} dim={emb_dim} top_k={K} sel={sel} style={style}")

                    # Generator per configuration
                    gen = GeneratorLLM()

                    # Predict
                    preds: List[Dict[str, Any]] = []
                    for _, row in tqdm(gold.iterrows(), total=len(gold)):
                        q = row["question"]
                        hits = index.retrieve(q, top_k=K)
                        context = build_context_from_hits(hits, strategy=sel, gold_answer=row["answer"])
                        prompt = build_prompt(q, context, style=style)
                        answer = gen.generate(prompt)
                        preds.append({
                            "id": int(row["id"]),
                            "question": q,
                            "gold_answer": row["answer"],
                            "pred_answer": answer,
                            "context": context,
                            "top_k": K,
                            "selection": sel,
                            "embedding_model": model_name,
                            "embedding_dim": emb_dim,
                            "prompt_style": style,
                            "top_doc_id": hits[0]["doc_id"] if hits else None
                        })

                    # Save predictions
                    write_jsonl(preds, pred_path)

                    # Compute metrics
                    m = _compute_squad_metrics_from_jsonl(pred_path, gold_path)
                    record = {
                        "system": "naive",
                        "phase": "phase4",
                        "embedding_model": model_name,
                        "embedding_dim": emb_dim,
                        "top_k": K,
                        "selection": sel,
                        "prompt_style": style,
                        "n": m["n"],
                        "hf_em": m["exact_match"],
                        "hf_f1": m["f1"],
                    }
                    with open(metrics_path, "w") as f:
                        json.dump(record, f, indent=2)

                    rows.append(record)

    # Save summary table
    df = pd.DataFrame(rows).sort_values(["hf_f1", "hf_em"], ascending=False)
    df.to_csv(f"{out_dir}/phase4_parameter_sweep.csv", index=False)

    # Merge with comparison file
    comp = Path(f"{out_dir}/comparison_analysis.csv")
    if comp.exists():
        old = pd.read_csv(comp)
        merged = (pd.concat([old, df], ignore_index=True)
                    .drop_duplicates(subset=["system","phase","embedding_model","top_k","selection","prompt_style"], keep="last"))
        merged.to_csv(comp, index=False)
    else:
        df.to_csv(comp, index=False)

    logger.info(f"[Phase4] Wrote {out_dir}/phase4_parameter_sweep.csv and updated {out_dir}/comparison_analysis.csv")


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for Phase 4 experiments.

    Example usage:
        python phase4_experiments.py --run \
            --embedding_models paraphrase-MiniLM-L3-v2,all-MiniLM-L6-v2 \
            --top_ks 3,5 \
            --selections first,concat \
            --prompt_styles instruction,cot
    """
    ap = argparse.ArgumentParser(description="Step 4 parameter sweep for Naive RAG.")
    ap.add_argument("--run", action="store_true", help="Run the Phase 4 parameter grid.")
    ap.add_argument("--corpus_path", type=str, default="data/processed/corpus.jsonl")
    ap.add_argument("--gold_path", type=str, default="data/evaluation/gold.jsonl")

    ap.add_argument("--embedding_models", type=str,
                    default="paraphrase-MiniLM-L3-v2,all-MiniLM-L6-v2,multi-qa-MiniLM-L6-cos-v1",
                    help="Comma-separated SentenceTransformer model names.")
    ap.add_argument("--top_ks", type=str, default="3,5,10",
                    help="Comma-separated K values, e.g. 3,5,10")
    ap.add_argument("--selections", type=str, default="first,concat,best_by_len",
                    help="Comma-separated strategies: first,concat,best_by_len")
    ap.add_argument("--prompt_styles", type=str, default="instruction",
                    help="Comma-separated prompt styles (e.g., instruction,persona,cot)")
    ap.add_argument("--max_examples", type=int, default=0,
                    help="For quick tests: limit #eval examples (0 = all)")

    return ap.parse_args()


def main() -> None:
    """
    Main entry point for Phase 4 experiments.
    Runs a full parameter sweep if --run is specified.
    """
    args = _parse_args()
    if not args.run:
        print("Nothing to do. Use --run.")
        return

    emb_models = [s.strip() for s in args.embedding_models.split(",") if s.strip()]
    top_ks = [int(x) for x in args.top_ks.split(",") if x.strip()]
    selections = [s.strip() for s in args.selections.split(",") if s.strip()]
    prompt_styles = [s.strip() for s in args.prompt_styles.split(",") if s.strip()]

    run_phase4_grid(
        corpus_path=args.corpus_path,
        gold_path=args.gold_path,
        embedding_models=emb_models,
        top_ks=top_ks,
        selections=selections,
        prompt_styles=prompt_styles,
        out_dir="results",
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
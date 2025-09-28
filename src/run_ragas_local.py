import os
os.environ["RAGAS_MAX_WORKERS"] = "1"           # stability over speed for LLM judging
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # quieter tokenizers / avoid warning spam

import json, argparse, csv
from pathlib import Path
from typing import List, Dict, Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from pydantic.v1 import PrivateAttr

# Embeddings (version tolerant across ragas releases)
try:
    from ragas.embeddings import HuggingfaceEmbeddings as RagasEmbeddings
except Exception:
    from ragas.embeddings import HuggingFaceEmbeddings as RagasEmbeddings


def load_jsonl_fix_schema(fp: Path) -> List[dict]:
    """
    Normalize JSONL rows from your RAG predictions into the RAGAS schema.

    Expected input fields per line (flexible):
        - question (str)
        - pred_answer (str)      -> mapped to 'answer'
        - gold_answer (str)      -> mapped to 'reference' and 'ground_truth'
        - context or contexts    -> 'contexts' (list[str])

    Returns:
        List[dict]: rows with keys:
            question, answer, contexts (list[str]), reference, ground_truth
    """
    rows: List[dict] = []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)

            q = o.get("question", "")
            pred = o.get("pred_answer", "")
            gold = o.get("gold_answer", "")

            # contexts list: accept 'contexts' (list) or 'context' (str|list)
            if isinstance(o.get("contexts"), list):
                ctxs = [str(c) for c in o["contexts"]]
            else:
                c = o.get("context", "")
                if isinstance(c, list):
                    ctxs = [str(x) for x in c]
                else:
                    ctxs = [str(c)] if c else []

            rows.append({
                "question": q,
                "answer": pred,
                "reference": gold,
                "ground_truth": gold,
                "contexts": ctxs
            })
    return rows


def augment_yes_no(rows: List[dict]) -> List[dict]:
    """
    Expand bare Yes/No answers so the faithfulness metric can extract statements.

    Rationale:
        Faithfulness often scores better when the answer includes a concrete
        declarative statement, not just "yes/no". We wrap yes/no into a simple
        claim based on the question to enable extraction.
    """
    out = []
    for r in rows:
        ans = (r.get("answer") or "").strip()
        q = (r.get("question") or "").strip().rstrip("?").strip()
        if q and ans:
            low = ans.lower()
            if low in ("yes", "yes.", "y", "true"):
                r = dict(r); r["answer"] = f"It is true that {q}."
            elif low in ("no", "no.", "n", "false"):
                r = dict(r); r["answer"] = f"It is false that {q}."
        out.append(r)
    return out


class ChatWrappedHF(HuggingFacePipeline):
    """
    Wrap a HuggingFace generation pipeline to:
      1) use the model's chat template, and
      2) instruct it to return ONLY valid JSON (no prose/markdown),
         which helps RAGAS parsers remain stable across prompts.

    Note:
        We keep this light-touch and local—no OpenAI calls.
    """
    _tok = PrivateAttr()

    def __init__(self, pipe, tokenizer, **kwargs):
        super().__init__(pipeline=pipe, **kwargs)
        self._tok = tokenizer

    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        # Add a simple system+user chat wrapper and apply the tokenizer's template
        msgs = [
            {"role": "system", "content": "You are a strict JSON generator. Return ONLY valid JSON. No prose, no markdown."},
            {"role": "user", "content": prompt},
        ]
        formatted = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return super()._call(formatted, stop=stop, run_manager=run_manager, **kwargs)


def build_judge_llm(model_name: str, judge_tokens: int = 128) -> ChatWrappedHF:
    """
    Build a local HF judge LLM pipeline for RAGAS prompts.

    Args:
        model_name (str): HF model id for judging (e.g., 'microsoft/Phi-3.5-mini-instruct').
        judge_tokens (int): max_new_tokens for each judge response.

    Returns:
        ChatWrappedHF: a LangChain-compatible LLM that forces JSON outputs.
    """
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto", trust_remote_code=True, low_cpu_mem_usage=True
    )
    # Ensure a valid pad token to avoid warnings/errors in generation
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=judge_tokens,
        do_sample=False,
        return_full_text=False,
        truncation=True,
        batch_size=1,
    )
    return ChatWrappedHF(gen, tok)


def run_ragas(preds_path: Path, llm_name: str, embed_name: str, limit: int, judge_tokens: int):
    """
    Run the full RAGAS evaluation for a given predictions file.

    Args:
        preds_path (Path): Path to predictions JSONL (naive or enhanced).
        llm_name (str): HF judge model name.
        embed_name (str): HF embeddings model name for RAGAS.
        limit (int): cap number of examples (0 or None = all).
        judge_tokens (int): token budget for judge responses.

    Returns:
        result object (implementation-specific): RAGAS summary (later normalized).
    """
    rows = load_jsonl_fix_schema(preds_path)
    rows = augment_yes_no(rows)
    if limit and limit > 0:
        rows = rows[:limit]
    ds = Dataset.from_list(rows)

    llm = build_judge_llm(llm_name, judge_tokens)
    emb = RagasEmbeddings(model_name=embed_name)

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    return evaluate(ds, metrics=metrics, llm=llm, embeddings=emb, raise_exceptions=False)


def to_scalar_dict(result_obj) -> Dict[str, float]:
    """
    Normalize RAGAS result objects across versions into a flat dict.

    Handles several possible shapes:
        - dict
        - objects with .scores/.results/.summary/.dict (callable or attr)

    Returns:
        Dict[str, float|Any]: scalar dictionary (floats where possible).
    """
    # RAGAS returns slightly different structures across versions; normalize.
    if isinstance(result_obj, dict):
        return {k: float(v) if isinstance(v, (int, float)) else v for k, v in result_obj.items()}
    for attr in ("scores", "results", "summary", "dict"):
        if hasattr(result_obj, attr):
            v = getattr(result_obj, attr)
            if callable(v):
                try:
                    v = v()
                except Exception:
                    continue
            if isinstance(v, dict):
                return {k: float(x) if isinstance(x, (int, float)) else x for k, x in v.items()}
    return {"raw": str(result_obj)}


def write_json(p: Path, data: Dict[str, Any]):
    """
    Write a JSON file with safe directory creation.

    Args:
        p (Path): output path.
        data (dict): JSON-serializable object.
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(p: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    """
    Write a CSV file with headers.

    Args:
        p (Path): output path.
        rows (List[dict]): row dicts.
        fieldnames (List[str]): CSV header fields (in order).
    """
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    """
    Command-line entry point.

    Flags & arguments:
        --mode {naive, enhanced, both}  Which predictions to evaluate (default: both)
        --naive PATH                     Path to naive predictions JSONL
        --enhanced PATH                  Path to enhanced predictions JSONL
        --llm MODEL                      HF judge model id (default: Phi-3.5-mini-instruct)
        --embed MODEL                    HF embeddings model id (default: BGE small)
        --limit N                        Evaluate first N rows (default: 200 for speed)
        --judge_tokens N                 Max new tokens for judge model (default: 128)
        --out_dir PATH                   Output directory (default: results)

    Example:
        python eval_ragas_min.py --mode both --limit 200 \
            --llm microsoft/Phi-3.5-mini-instruct \
            --embed BAAI/bge-small-en-v1.5
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--naive", default="results/predictions_naive.jsonl", type=Path)
    ap.add_argument("--enhanced", default="results/predictions_enhanced.jsonl", type=Path)
    ap.add_argument("--llm", default="microsoft/Phi-3.5-mini-instruct")
    ap.add_argument("--embed", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--judge_tokens", type=int, default=128)
    ap.add_argument("--out_dir", default="results", type=Path)
    ap.add_argument("--mode", choices=["naive", "enhanced", "both"], default="both")
    args = ap.parse_args()

    n_scores = None
    e_scores = None
    rows = None  # will hold comparison rows only if both are computed

    # Run RAGAS on naive predictions
    if args.mode in ("naive", "both"):
        print("== RAGAS: NAIVE ==")
        res_n = run_ragas(args.naive, args.llm, args.embed, args.limit, args.judge_tokens)
        n_scores = to_scalar_dict(res_n)
        write_json(args.out_dir / "ragas_naive_scores.json", n_scores)

    # Run RAGAS on enhanced predictions
    if args.mode in ("enhanced", "both"):
        print("\n== RAGAS: ENHANCED ==")
        res_e = run_ragas(args.enhanced, args.llm, args.embed, args.limit, args.judge_tokens)
        e_scores = to_scalar_dict(res_e)
        write_json(args.out_dir / "ragas_enhanced_scores.json", e_scores)

    # Only build comparison if we have both
    if n_scores is not None and e_scores is not None:
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        rows = []
        for m in metrics:
            # Defensive conversion to floats (handle missing keys gracefully)
            n = float(n_scores.get(m, "nan")) if isinstance(n_scores.get(m, None), (int, float)) else float("nan")
            e = float(e_scores.get(m, "nan")) if isinstance(e_scores.get(m, None), (int, float)) else float("nan")
            d = (e - n) if (isinstance(n, float) and isinstance(e, float)) else float("nan")
            rows.append({
                "metric": m,
                "naive": f"{n:.4f}" if n == n else "nan",
                "enhanced": f"{e:.4f}" if e == e else "nan",
                "delta (enhanced - naive)": f"{d:+.4f}" if d == d else "nan"
            })
        write_csv(args.out_dir / "ragas_comparison.csv", rows, list(rows[0].keys()))
        print("\nWrote comparison CSV:", args.out_dir / "ragas_comparison.csv")

    # Pretty print (only if comparison exists)
    if rows is not None:
        print("\n=== RAGAS COMPARISON (limit =", args.limit, ") ===")
        for r in rows:
            print(f"{r['metric']:<18}  naive={r['naive']:<8}  enhanced={r['enhanced']:<8}  Δ={r['delta (enhanced - naive)']}")
        print("\nSaved:")
        print(" -", args.out_dir / "ragas_naive_scores.json")
        print(" -", args.out_dir / "ragas_enhanced_scores.json")
        print(" -", args.out_dir / "ragas_comparison.csv")
    else:
        # If only one side ran, still tell the user what was saved
        print("\nSaved:")
        if n_scores is not None:
            print(" -", args.out_dir / "ragas_naive_scores.json")
        if e_scores is not None:
            print(" -", args.out_dir / "ragas_enhanced_scores.json")

    print("Done")


if __name__ == "__main__":
    main()
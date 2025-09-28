from __future__ import annotations
import os, json, argparse
from typing import List, Dict, Any
from tqdm import tqdm

from .utils import (get_logger, load_corpus, load_gold, Embedder, RagIndex, 
                    GeneratorLLM, build_prompt, write_jsonl, raga_stub_metrics)

# Initialize logger for the naive pipeline
logger = get_logger("naive")


def run_naive(corpus_path: str, gold_path: str, embedding_model: str, top_k: int, prompt_style: str, out_prefix: str):
    """
    Run the naive Retrieval-Augmented Generation (RAG) pipeline.

    Args:
        corpus_path (str): Path to corpus JSONL file containing passages.
        gold_path (str): Path to gold standard JSONL file containing QA pairs.
        embedding_model (str): HuggingFace embedding model name (e.g., 'all-MiniLM-L6-v2').
        top_k (int): Number of candidate documents to retrieve from the index.
        prompt_style (str): Prompt style to use when generating answers. 
                            Options: "instruction", "persona", "cot".
        out_prefix (str): Prefix for output filenames (e.g., 'naive').

    Pipeline Steps:
        1. Load corpus and gold QA dataset.
        2. Build embedding-based index for retrieval.
        3. For each question:
            - Retrieve top-k candidate documents.
            - Select the first (naive choice).
            - Construct a prompt with the retrieved context.
            - Generate an answer with the LLM.
        4. Save predictions and simple stub metrics.
    """
    # Load corpus and gold dataset
    corpus = load_corpus(corpus_path)
    gold = load_gold(gold_path)

    # Build retrieval index with embeddings
    embedder = Embedder(model_name=embedding_model)
    index = RagIndex(embedder=embedder, corpus_df=corpus)
    index.build()

    # Initialize generator
    gen = GeneratorLLM()

    preds = []
    for _, row in tqdm(gold.iterrows(), total=len(gold)):
        q = row["question"]

        # Retrieve top-k passages (no reranking here, naive approach)
        hits = index.retrieve(q, top_k=top_k)

        # Naively take the first hit (if available) as context
        context = hits[0]["text"] if hits else ""

        # Build prompt using context
        prompt = build_prompt(q, context, style=prompt_style)

        # Generate answer
        answer = gen.generate(prompt)

        # Store prediction details
        preds.append({
            "id": int(row["id"]),
            "question": q,
            "gold_answer": row["answer"],
            "pred_answer": answer,
            "context": context,
            "top_doc_id": hits[0]["doc_id"] if hits else None
        })

    # Save predictions and evaluation results
    os.makedirs("results", exist_ok=True)
    write_jsonl(preds, f"results/predictions_{out_prefix}.jsonl")

    # Compute simple stub RAG metrics (placeholder for RAGAs/ARES)
    rag_metrics = raga_stub_metrics(preds)
    with open(f"results/{out_prefix}_results.json", "w") as f:
        json.dump(rag_metrics, f, indent=2)

    logger.info("Saved naive predictions and results.")


def main():
    """
    Command-line entry point for running the naive RAG pipeline.

    Example usage:
        python naive.py --run \
            --corpus_path data/processed/corpus.jsonl \
            --gold_path data/evaluation/gold.jsonl \
            --embedding_model all-MiniLM-L6-v2 \
            --top_k 3 \
            --prompt_style instruction \
            --out_prefix naive
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_index", action="store_true", help="(Kept for compatibility; index builds on run.)")
    ap.add_argument("--run", action="store_true", help="Run the naive RAG pipeline.")
    ap.add_argument("--corpus_path", default="data/processed/corpus.jsonl", help="Path to corpus file.")
    ap.add_argument("--gold_path", default="data/evaluation/gold.jsonl", help="Path to gold evaluation file.")
    ap.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Embedding model to use.")
    ap.add_argument("--top_k", type=int, default=3, help="Number of documents to retrieve.")
    ap.add_argument("--prompt_style", default="instruction", choices=["instruction","persona","cot"],
                    help="Prompting style for the generator.")
    ap.add_argument("--out_prefix", default="naive", help="Prefix for output file names.")
    args = ap.parse_args()

    if args.run:
        run_naive(args.corpus_path, args.gold_path, args.embedding_model, args.top_k, args.prompt_style, args.out_prefix)
    else:
        print("Nothing to do. Use --run.")


if __name__ == "__main__":
    main()
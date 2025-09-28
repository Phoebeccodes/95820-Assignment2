from __future__ import annotations
import os, json, argparse
from typing import List, Dict, Any
from tqdm import tqdm

from .utils import (get_logger, load_corpus, load_gold, Embedder, RagIndex, 
                    GeneratorLLM, build_prompt, write_jsonl, heuristic_query_rewrite,
                    Reranker, raga_stub_metrics)

# Initialize logger for this module
logger = get_logger("enhanced")


def run_enhanced(corpus_path: str, gold_path: str, embedding_model: str, top_k: int, prompt_style: str):
    """
    Run the enhanced Retrieval-Augmented Generation (RAG) pipeline.

    Args:
        corpus_path (str): Path to corpus JSONL file containing passages.
        gold_path (str): Path to gold standard JSONL file containing QA pairs.
        embedding_model (str): HuggingFace embedding model name (e.g., 'all-MiniLM-L6-v2').
        top_k (int): Number of candidate documents to retrieve from the index.
        prompt_style (str): Prompt style to use for generation. 
                            Options: "instruction", "persona", "cot".

    Pipeline Steps:
        1. Load corpus and gold standard QA dataset.
        2. Build an embedding-based index for retrieval.
        3. For each query:
            - Rewrite query for clarity.
            - Retrieve top-k candidate documents.
            - Rerank candidates to select the most relevant passage.
            - Construct a prompt using the chosen passage.
            - Generate an answer with the LLM.
        4. Save predictions and evaluation metrics.
    """
    # Load corpus and gold QA dataset
    corpus = load_corpus(corpus_path)
    gold = load_gold(gold_path)

    # Build retrieval index using embeddings
    embedder = Embedder(model_name=embedding_model)
    index = RagIndex(embedder=embedder, corpus_df=corpus)
    index.build()

    # Initialize reranker and generator
    reranker = Reranker()
    gen = GeneratorLLM()

    preds = []
    for _, row in tqdm(gold.iterrows(), total=len(gold)):
        q = row["question"]

        # Step 1: Rewrite query (heuristic improvements)
        q_re = heuristic_query_rewrite(q)

        # Step 2: Retrieve candidate passages
        hits = index.retrieve(q_re, top_k=top_k)

        # Step 3: Rerank retrieved passages
        passages = [h["text"] for h in hits]
        if passages:
            order = reranker.rerank(q, passages, top_k=min(5, len(passages)))
            best = hits[order[0]]
            context = best["text"]
            top_doc_id = best["doc_id"]
        else:
            context, top_doc_id = "", None

        # Step 4: Build prompt and generate answer
        prompt = build_prompt(q, context, style=prompt_style)
        answer = gen.generate(prompt)

        # Store prediction details
        preds.append({
            "id": int(row["id"]),
            "question": q,
            "rewritten": q_re,
            "gold_answer": row["answer"],
            "pred_answer": answer,
            "context": context,
            "top_doc_id": top_doc_id
        })

    # Save predictions and evaluation results
    os.makedirs("results", exist_ok=True)
    write_jsonl(preds, f"results/predictions_enhanced.jsonl")
    rag_metrics = raga_stub_metrics(preds)
    with open("results/enhanced_results.json", "w") as f:
        json.dump(rag_metrics, f, indent=2)

    logger.info("Saved enhanced predictions and results.")


def main():
    """
    Command-line entry point for running the enhanced RAG pipeline.

    Example usage:
        python enhanced.py --run \
            --corpus_path data/processed/corpus.jsonl \
            --gold_path data/evaluation/gold.jsonl \
            --embedding_model all-MiniLM-L6-v2 \
            --top_k 10 \
            --prompt_style instruction
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="Run the enhanced RAG pipeline.")
    ap.add_argument("--corpus_path", default="data/processed/corpus.jsonl", help="Path to corpus file.")
    ap.add_argument("--gold_path", default="data/evaluation/gold.jsonl", help="Path to gold evaluation file.")
    ap.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Embedding model to use.")
    ap.add_argument("--top_k", type=int, default=10, help="Number of documents to retrieve.")
    ap.add_argument("--prompt_style", default="instruction", choices=["instruction","persona","cot"],
                    help="Prompting style for the generator.")
    args = ap.parse_args()

    if args.run:
        run_enhanced(args.corpus_path, args.gold_path, args.embedding_model, args.top_k, args.prompt_style)
    else:
        print("Nothing to do. Use --run.")


if __name__ == "__main__":
    main()
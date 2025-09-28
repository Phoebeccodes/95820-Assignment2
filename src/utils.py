"""
utils.py
========
Utility functions and lightweight components shared across the RAG pipeline.

Design goals
------------
- **Portable**: FAISS is optional; if unavailable, we fall back to a NumPy cosine search.
- **Colab-friendly**: Auto-pick GPU when available for SentenceTransformer/CrossEncoder/Generator.
- **Modular**: Small classes for Embedding, Indexing, Reranking, Generation, Prompting, and Metrics.
- **Traceable**: Minimal logging to understand what the system is doing.

Author: Phoebe Zhou
"""

from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import faiss as _faiss  # type: ignore
except Exception:
    _faiss = None

USE_FAISS = os.getenv("USE_FAISS", "0") == "1"  # default OFF to avoid surprise crashes


try:
    import torch  # type: ignore
    from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
    from transformers import pipeline  # type: ignore
except Exception:
    torch = None  # torch or transformers might not be present at import time in some envs
    SentenceTransformer = None  # type: ignore
    CrossEncoder = None  # type: ignore
    pipeline = None  # type: ignore


def get_logger(name: str = "rag") -> logging.Logger:
    """Return a module-level logger with a sensible default format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


logger = get_logger()


def _infer_device() -> str:
    """
    Best-effort device selection:
      - "cuda" if a GPU is visible and PyTorch is available,
      - else "cpu".
    """
    try:
        if torch is not None and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSON Lines file into a list of dicts."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    """Write a list of dicts to JSON Lines format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_corpus(corpus_path: str) -> pd.DataFrame:
    """
    Load the indexed corpus from JSONL.

    Expects fields:
      - id: str
      - text: str
      - title: (optional) str

    Returns a DataFrame with columns ["id", "title", "text"].
    """
    rows = read_jsonl(corpus_path)
    df = pd.DataFrame(rows)
    if "title" not in df.columns:
        df["title"] = ""
    return df[["id", "title", "text"]]


def load_gold(gold_path: str) -> pd.DataFrame:
    """
    Load the evaluation set (questions + gold answers) from JSONL.

    Expects fields:
      - id: int/str
      - question: str
      - answer: str
    """
    rows = read_jsonl(gold_path)
    df = pd.DataFrame(rows)
    return df[["id", "question", "answer"]]



@dataclass
class Embedder:
    """
    Thin wrapper around Sentence-Transformers to produce float32 embeddings.

    Parameters
    ----------
    model_name: str
        Sentence-Transformers model id. Default: "all-MiniLM-L6-v2".
    device: Optional[str]
        "cuda" or "cpu". If None, we auto-detect.
    normalize: bool
        If True, embeddings are L2-normalized, which makes dot product == cosine.
    """
    model_name: str = "all-MiniLM-L6-v2"
    device: Optional[str] = None
    normalize: bool = True

    def __post_init__(self) -> None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not available.")
        dev = self.device or _infer_device()
        logger.info(f"Loading encoder '{self.model_name}' on {dev} …")
        self.model = SentenceTransformer(self.model_name, device=dev)

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(emb, dtype="float32")



def build_index(embeddings: np.ndarray):
    """
    Build a similarity index over corpus embeddings.

    Returns
    -------
    A tuple (mode, obj) where:
      - mode == "faiss": obj is a FAISS IndexFlatIP
      - mode == "numpy": obj is the raw embedding matrix
    """
    if USE_FAISS and _faiss is not None:
        dim = embeddings.shape[1]
        index = _faiss.IndexFlatIP(dim)  # inner product; with normalized vectors -> cosine
        index.add(embeddings)
        return ("faiss", index)
    return ("numpy", embeddings)


def search(index_obj, query_vecs: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search the index for nearest neighbors.

    Parameters
    ----------
    index_obj: tuple
        Output of build_index().
    query_vecs: np.ndarray
        Query embeddings of shape (B, d).
    top_k: int
        Number of neighbors to return.

    Returns
    -------
    (D, I):
        D: (B, top_k) similarities
        I: (B, top_k) integer indices into the corpus matrix
    """
    mode, obj = index_obj
    if mode == "faiss":
        D, I = obj.search(query_vecs, top_k)
        return D, I

    # NumPy fallback: cosine similarity (dot product as embeddings are normalized)
    emb = obj  # (N, d)
    sims = query_vecs @ emb.T                     # (B, N)
    I = np.argsort(-sims, axis=1)[:, :top_k]      # top-k argsort
    D = np.take_along_axis(sims, I, axis=1)       # gather scores
    return D, I


def heuristic_query_rewrite(q: str) -> str:
    """
    A tiny, cheap query-rewriter:
    - lowercase
    - remove punctuation
    - drop trivial stopwords
    Useful as a baseline; replace with a stronger rewriter later if desired.
    """
    q = q.strip()
    q = re.sub(r"[^\w\s\-']", " ", q)
    tokens = q.lower().split()
    stop = set(
        "a an and are as at be by for from has have how i in is it of on or that "
        "the to was were what when where which who why will with".split()
    )
    keep = [t for t in tokens if t not in stop and len(t) > 2]
    return " ".join(keep) if keep else q



@dataclass
class Reranker:
    """
    Cross-Encoder reranker for improved passage ordering.

    Default: "cross-encoder/ms-marco-MiniLM-L-6-v2" (fast and small).
    """
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: Optional[str] = None

    def __post_init__(self) -> None:
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers (CrossEncoder) is not available.")
        dev = self.device or _infer_device()
        logger.info(f"Loading reranker '{self.model_name}' on {dev} …")
        self.model = CrossEncoder(self.model_name, device=dev)

    def rerank(self, query: str, passages: List[str], top_k: int = 5) -> List[int]:
        """
        Score query–passage pairs, return indices of the best `top_k` passages.
        """
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs).tolist()
        order = np.argsort(scores)[::-1][:top_k]
        return order.tolist()



@dataclass
class GeneratorLLM:
    """
    Text generator interface. By default we use an open model (Flan-T5)
    so the pipeline runs with zero paid dependencies. You can swap this out
    for OpenAI/Anthropic/Cohere calls in a single place later.

    Parameters
    ----------
    model_name: str
        HF model id for text2text generation.
    max_new_tokens: int
        Length cap for generated answers.
    """
    model_name: str = "google/flan-t5-base"
    max_new_tokens: int = 128

    def __post_init__(self) -> None:
        if pipeline is None:
            raise RuntimeError("transformers is not available.")
        dev = _infer_device()
        device_arg = 0 if dev == "cuda" else -1
        logger.info(f"Loading generator '{self.model_name}' on {dev} …")
        self.pipe = pipeline("text2text-generation", model=self.model_name, device=device_arg)

    def generate(self, prompt: str) -> str:
        out = self.pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )[0]["generated_text"]
        return out.strip()



def build_prompt(question: str, context: str, style: str = "instruction") -> str:
    """
    Simple prompt templates used in Step 3 experiments.
    """
    if style == "instruction":
        return (
            "Answer the question using ONLY the context.\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )
    if style == "persona":
        return (
            "You are a helpful history tutor. Using the provided context, give a concise, "
            "student-friendly answer.\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nTutor:"
        )
    if style == "cot":
        return (
            "Use step-by-step reasoning with the given context to answer.\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nLet's reason step by step:"
        )
    # fallback
    return (
        "Answer using the context.\n"
        f"Context:\n{context}\n\n"
        f"Q: {question}\nA:"
    )


def normalize_text(s: str) -> str:
    """Lowercase + remove articles/punctuation/extra whitespace (SQuAD recipe)."""
    import string

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_em(pred: str, gold: str) -> Tuple[float, float]:
    """
    Token-level F1 and exact match, after SQuAD normalization.
    """
    pred_toks = normalize_text(pred).split()
    gold_toks = normalize_text(gold).split()
    common = set(pred_toks) & set(gold_toks)
    num_same = sum(min(pred_toks.count(w), gold_toks.count(w)) for w in common)

    if len(pred_toks) == 0 or len(gold_toks) == 0:
        match = float(pred_toks == gold_toks)
        return match, match

    precision = num_same / len(pred_toks) if pred_toks else 0.0
    recall = num_same / len(gold_toks) if gold_toks else 0.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    em = 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0
    return f1, em


@dataclass
class RagIndex:
    """
    Small convenience wrapper that:
      - encodes corpus passages,
      - builds a FAISS (or NumPy) index,
      - retrieves top-k matches for a query.
    """
    embedder: Embedder
    corpus_df: pd.DataFrame
    index: Optional[object] = None
    embeddings: Optional[np.ndarray] = None

    def build(self) -> None:
        texts = self.corpus_df["text"].tolist()
        self.embeddings = self.embedder.encode(texts)
        self.index = build_index(self.embeddings)
        mode = self.index[0]
        logger.info(
            f"Built {mode.upper()} index over {len(texts)} passages (dim={self.embeddings.shape[1]})."
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        qv = self.embedder.encode([query])
        D, I = search(self.index, qv, top_k=top_k)
        hits: List[Dict[str, Any]] = []
        for rank, idx in enumerate(I[0]):
            row = self.corpus_df.iloc[idx]
            hits.append(
                {
                    "rank": rank + 1,
                    "doc_id": row["id"],
                    "title": row.get("title", ""),
                    "text": row["text"],
                    "score": float(D[0][rank]),
                }
            )
        return hits


def raga_stub_metrics(preds: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Extremely lightweight "context precision/recall" proxy:

    - Normalize gold answer and context.
    - Count word overlap as a rough proxy for whether the gold answer
      appears in the retrieved context.
    - This is only a placeholder until you wire up RAGAs or ARES.

    Returns a dict with two floats averaged over examples.
    """
    ctx_prec, ctx_rec = [], []
    for p in preds:
        gold = p.get("gold_answer", "")
        ctx = p.get("context", "")
        if not gold or not ctx:
            continue

        gold_norm = set(normalize_text(gold).split())
        ctx_norm = set(normalize_text(ctx).split())
        if not gold_norm:
            continue

        overlap = len(gold_norm & ctx_norm)
        ctx_prec.append(overlap / max(1, len(ctx_norm)))
        ctx_rec.append(overlap / len(gold_norm))

    return {
        "context_precision_stub": float(np.mean(ctx_prec)) if ctx_prec else 0.0,
        "context_recall_stub": float(np.mean(ctx_rec)) if ctx_rec else 0.0,
    }
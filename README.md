[]

# Rag Evaluation

This project implements and evaluates baseline and enhanced RAG pipelines for open-domain question answering. It includes parameter comparison, RAGAS evaluation, and reproducibility notebooks.

## Project Structure

```
assignment2-rag/
├── data/                      # Corpus and gold evaluation JSONL files
├── docs/                      # Documentation (reports, writeups, etc.)
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── system_evaluation.ipynb
│   └── final_analysis.ipynb
├── results/                   # Predictions, metrics, CSV outputs
├── src/
│   ├── enhanced_rag.py        # Enhanced pipeline (query rewrite + rerank)
│   ├── naive_rag.py           # Baseline naive pipeline
│   ├── phase4_experiments.py  # Parameter sweep grid
│   ├── run_ragas_local.py     # RAGAS evaluation (local HF)
│   ├── utils.py               # Shared components
│   └── evaluation.py          # Additional evaluation helpers
├── README.md
└── requirements.txt
```

## Setup

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/Phoebeccodes/95820-Assignment2.git
cd 95820-Assignment2
pip install -r requirements.txt

```

Core dependencies:
• Python 3.10+
• torch
• transformers
• sentence-transformers
• faiss-cpu (optional)
• pandas, numpy, tqdm
• ragas, evaluate, datasets, jupyter

### 2. Data format

    •	Corpus (data/processed/corpus.jsonl):
    {"id": "doc1", "title": "Hamlet", "text": "Hamlet is a play written by William Shakespeare in 1600."}

    •	Gold evaluation (data/evaluation/gold.jsonl):
    {"id": 1, "question": "Who wrote Hamlet?", "answer": "William Shakespeare"}

## Usage

All commands should be run from the project root (assignment2-rag/):

```bash
# Naive pipeline
prompt_styles = ["instruction", "persona", "cot"]
!python -m src.naive_rag --run --top_k 1 --prompt_style {style}

# Parameter sweep
!python -m src.phase4_experiments --run

#Enhanced rag (query rewrite + rerank)
!python -m src.enhanced_rag --run --top_k 10 --prompt_style instruction

#RAGAS evaluation
!python src/run_ragas_local.py \
  --naive results/predictions_naive.jsonl \
  --llm microsoft/Phi-3.5-mini-instruct \
  --embed BAAI/bge-small-en-v1.5 \
  --judge_tokens 128 \
  --mode naive

!python src/run_ragas_local.py \
  --enhanced results/predictions_enhanced.jsonl \
  --llm microsoft/Phi-3.5-mini-instruct \
  --embed BAAI/bge-small-en-v1.5 \
  --judge_tokens 128 \
  --mode enhanced


```

## Results

• Predictions → results/predictions\_\*.jsonl

• Metrics → results/\*\_results.json, results/phase4_parameter_sweep.csv, results/comparison_analysis.csv

• RAGAS comparison → results/ragas_comparison.csv

## Notes

• AI assistance (ChatGPT) was used for coding, debugging, and documentation support.

• All experiments used HuggingFace models only: Flan-T5, MiniLM embeddings, CrossEncoder reranker, and Phi-3.5-mini-instruct for RAGAS judging.

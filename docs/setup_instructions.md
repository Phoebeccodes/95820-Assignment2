# Setup Instructions

1. Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```

2. Install requirements:
```
pip install -r requirements.txt
```

3. Prepare data:
- Place your **corpus** at `data/processed/corpus.jsonl` with lines like:
  `{"id": "doc1", "title": "Lincoln", "text": "Abraham Lincoln was the 16th President..."}`
- Place your **gold** Q/A at `data/evaluation/gold.jsonl` with lines like:
  `{"id": 1, "question": "Was Abraham Lincoln the sixteenth President of the United States?", "answer": "yes"}`

4. Run:
```
python -m src.naive_rag --run --top_k 3
python -m src.enhanced_rag --run --top_k 10
python -m src.evaluation --pred_path results/predictions_naive.jsonl --gold_path data/evaluation/gold.jsonl
```

# LLM Evaluation Framework

Benchmarks GPT-4o-mini, LLaMA-3, and Mixtral on 25 medical QA questions.
Measures faithfulness, answer relevancy, context precision, and hallucination rate.

## Files
```
pipeline.py    ← runs all models, scores answers, saves eval_results.csv
dashboard.py   ← streamlit dashboard, reads eval_results.csv
requirements.txt
```

## Setup
```bash
pip install -r requirements.txt

export OPENAI_API_KEY=sk-...
export GROQ_API_KEY=gsk_...     # free at console.groq.com
```

## Run
```bash
# Step 1 — generate results
python pipeline.py

# Step 2 — view dashboard
streamlit run dashboard.py
```

## Metrics
| Metric | Tool | What it measures |
|---|---|---|
| Faithfulness | RAGAS | Is the answer grounded in the context? |
| Answer Relevancy | RAGAS | Does it actually answer the question? |
| Context Precision | RAGAS | Was the right context used? |
| Hallucination | DeepEval | Did the model make things up? |
| Latency | Manual | Response time per query |

## Notes
- Mistral is run via Groq (mixtral-8x7b) — no HuggingFace inference needed
- RAGAS scores are per-model averages over all 25 questions
- DeepEval hallucination is scored per question

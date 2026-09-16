# LLM Evaluation Framework

Automated benchmarking system that runs 25 medical QA questions through 3 LLMs simultaneously, scores every answer using an LLM-as-judge approach, and displays results on an interactive Streamlit dashboard.

---

## What it does

```
25 Questions
     │
     ├──► llama-3.1-8b   ──► Answer + latency + tokens
     ├──► llama-3.3-70b  ──► Answer + latency + tokens
     └──► qwen3-32b      ──► Answer + latency + tokens
                                       │
                              LLM-as-judge scoring
                              (faithfulness · relevancy · correctness)
                                       │
                               eval_results.csv
                                       │
                            Streamlit dashboard
```

---

## Models

| Model | Provider | Size | via |
|---|---|---|---|
| llama-3.1-8b | Meta | 8B | Groq |
| llama-3.3-70b | Meta | 70B | Groq |
| qwen3-32b | Alibaba | 32B | Groq |

All models run via Groq — **completely free**, no credit card needed.

---

## Metrics

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is the answer grounded in the provided context? |
| **Relevancy** | Does the answer actually address the question? |
| **Correctness** | How close is the answer to the ground truth? |
| **Latency** | Response time per query in seconds |
| **Tokens used** | Total tokens consumed per response |

Scoring is done via **LLM-as-judge** — a separate LLaMA-3.1-8b instance evaluates each answer on all three metrics and returns a score between 0.0 and 1.0.

---

## Project structure

```
llm-eval-framework/
├── pipeline.py       ← runs all models, scores answers, saves eval_results.csv
├── dashboard.py      ← Streamlit dashboard, reads eval_results.csv
├── requirements.txt
└── README.md
```

Two files. That is the whole project.

---

## Setup

**1. Get a free Groq API key**

Go to [console.groq.com](https://console.groq.com) → API Keys → Create API Key. No credit card required.

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Add your API key**

Open `pipeline.py` and replace the key on line 13:

```python
groq_client = Groq(api_key="your_key_here")
```

---

## Running

**Step 1 — Run the pipeline**

```bash
python pipeline.py
```

This will send all 25 questions to each of the 3 models (75 API calls total), score every answer, and save results to `eval_results.csv`. Takes about 3–5 minutes.

**Step 2 — Launch the dashboard**

```bash
python -m streamlit run dashboard.py
```

Opens at `http://localhost:8501`

---

## Dashboard

- Model averages table — side-by-side comparison of all metrics
- Bar charts — faithfulness, correctness, relevancy, and latency per model
- Radar chart — overall model profile across all quality metrics
- Latency vs correctness scatter — bubble size = tokens used
- Per-question breakdown — drill into individual answers per model

---

## Tech stack

| Library | Purpose |
|---|---|
| `groq` | Call all 3 LLMs via Groq API |
| `streamlit` | Dashboard UI |
| `plotly` | Interactive charts |
| `pandas` | Read CSV and compute averages |
| `csv`, `time`, `json` | Save results, measure latency, parse scores |

No LangChain. No LangGraph. No heavyweight frameworks. Just Python.

---

## Why LLM-as-judge instead of RAGAS

RAGAS is the standard library for RAG evaluation but has dependency conflicts with newer Python versions due to a broken `langchain_community` integration. LLM-as-judge is a valid alternative — flexible, framework-free, and widely used in research. The known tradeoff is potential self-evaluation bias since the judge model is from the same family as one of the evaluated models.

---

## Sample results

| Model | Faithfulness | Relevancy | Correctness | Avg Latency |
|---|---|---|---|---|
| llama-3.1-8b | 0.984 | 1.000 | 0.904 | 1.09s |
| llama-3.3-70b | 0.980 | 1.000 | 0.896 | 0.31s |
| qwen3-32b | 0.968 | 1.000 | 0.880 | 0.85s |

---

## Resume bullets

```
• Built an automated LLM evaluation framework benchmarking LLaMA-3.1-8B,
  LLaMA-3.3-70B, and Qwen3-32B across 25 domain-specific medical QA questions

• Implemented LLM-as-judge scoring pipeline measuring faithfulness, answer
  relevancy, and correctness using Groq API (zero cost)

• Tracked latency and token usage per model across 75 API calls;
  identified LLaMA-3.3-70B as 3x faster than LLaMA-3.1-8B at comparable quality

• Deployed interactive Streamlit dashboard with bar charts, radar plot,
  latency-vs-correctness scatter, and per-question drill-down
```

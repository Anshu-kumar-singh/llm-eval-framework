# 📊 LLM Evaluation Framework

An automated benchmarking system that evaluates multiple Large Language Models (LLMs) on domain-specific question answering tasks using an **LLM-as-Judge** evaluation pipeline. The framework measures answer quality, latency, and token usage, then visualizes the results through an interactive Streamlit dashboard.

🚀 **Live Demo:** https://anhu321-llm-eval-framework.hf.space

The framework automatically:

* Runs multiple LLMs on the same benchmark dataset
* Measures inference latency and token usage
* Scores answers using an independent LLM-as-Judge
* Stores evaluation results in CSV format
* Visualizes model performance through interactive dashboards

---

# 🚀 Live Demo

🌐 **Try the dashboard here:**

**https://anhu321-llm-eval-framework.hf.space**

No installation required—open the dashboard to explore model benchmarking results and performance visualizations.

---

# ✨ Features

## 🤖 Multi-Model Benchmarking

The framework benchmarks multiple LLMs simultaneously.

Current models include:

* Meta Llama-3.1-8B
* Meta Llama-3.3-70B
* Alibaba Qwen3-32B

All models are served through the **Groq API**.

---

## ⚖️ Automated LLM-as-Judge Evaluation

Each generated answer is evaluated by an independent judge model instead of relying on manual scoring.

Evaluation metrics include:

* Faithfulness
* Relevancy
* Correctness

Scores range from **0.0 → 1.0**.

---

## ⏱️ Performance Monitoring

For every model response, the framework records:

* Response latency
* Total token usage
* Generated answer
* Evaluation scores

This enables quality-versus-speed comparisons.

---

## 📊 Interactive Dashboard

The Streamlit dashboard includes:

* Model comparison table
* Quality metric bar charts
* Radar chart
* Latency vs Correctness scatter plot
* Per-question drill-down
* Interactive filtering

---

# 🏗️ Evaluation Pipeline

```text
25 Benchmark Questions
        │
        ├───────────────┐
        │               │
        ▼               ▼
 Llama-3.1-8B      Llama-3.3-70B
        │               │
        └───────┬───────┘
                │
                ▼
          Qwen3-32B
                │
                ▼
     Collect Answers + Latency + Tokens
                │
                ▼
      LLM-as-Judge Evaluation
                │
                ▼
      Faithfulness
      Relevancy
      Correctness
                │
                ▼
        eval_results.csv
                │
                ▼
      Streamlit Dashboard
```

---

# 🛠️ Models

| Model         | Provider | Size | API  |
| ------------- | -------- | ---- | ---- |
| Llama-3.1-8B  | Meta     | 8B   | Groq |
| Llama-3.3-70B | Meta     | 70B  | Groq |
| Qwen3-32B     | Alibaba  | 32B  | Groq |

All models are accessed through the **Groq API**, allowing fast inference without infrastructure management.

---

# 📈 Evaluation Metrics

| Metric           | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| **Faithfulness** | Measures whether the answer is grounded in the provided context |
| **Relevancy**    | Measures how well the answer addresses the user's question      |
| **Correctness**  | Measures similarity to the reference answer                     |
| **Latency**      | Time required to generate the response                          |
| **Tokens Used**  | Total tokens consumed during inference                          |

---

# 📂 Project Structure

```text
llm-eval-framework/
│
├── pipeline.py
├── dashboard.py
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/llm-eval-framework.git

cd llm-eval-framework
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure API Key

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

# ▶️ Running the Framework

## Option 1: Use the Live Dashboard

Open your browser and visit:

**https://anhu321-llm-eval-framework.hf.space**

---

## Option 2: Run the Evaluation Pipeline

```bash
python pipeline.py
```

The pipeline:

* Sends benchmark questions to every model
* Collects responses
* Measures latency
* Records token usage
* Performs LLM-as-Judge evaluation
* Saves results to **eval_results.csv**

---

## Launch the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard opens on your local Streamlit server.

---

# 📊 Dashboard Features

The dashboard provides:

* 📈 Model average comparison table
* 📊 Faithfulness comparison
* 📊 Correctness comparison
* 📊 Relevancy comparison
* 📉 Latency comparison
* 📡 Radar chart
* 🎯 Latency vs Correctness scatter plot
* 🔍 Per-question analysis

---

# 🛠️ Tech Stack

| Technology | Purpose                    |
| ---------- | -------------------------- |
| Groq API   | High-speed LLM inference   |
| Streamlit  | Interactive dashboard      |
| Plotly     | Interactive visualizations |
| Pandas     | Data processing            |
| Python     | Evaluation pipeline        |
| CSV        | Result storage             |

---

# 📖 How It Works

### Step 1

The benchmark dataset is loaded.

↓

### Step 2

Each question is sent to every LLM.

↓

### Step 3

Responses are collected along with:

* Latency
* Token usage

↓

### Step 4

A separate LLM evaluates every generated answer.

↓

### Step 5

Scores are written to **eval_results.csv**.

↓

### Step 6

The Streamlit dashboard visualizes all benchmark results.

---

# 📋 Sample Results

| Model         | Faithfulness | Relevancy | Correctness | Avg Latency |
| ------------- | -----------: | --------: | ----------: | ----------: |
| Llama-3.1-8B  |        0.984 |     1.000 |       0.904 |      1.09 s |
| Llama-3.3-70B |        0.980 |     1.000 |       0.896 |      0.31 s |
| Qwen3-32B     |        0.968 |     1.000 |       0.880 |      0.85 s |

---

# 🚀 Future Improvements

* Support additional benchmark datasets
* Multi-domain evaluation
* Human evaluation integration
* Pairwise model comparison
* Automatic benchmark report generation
* RAGAS integration
* LLM leaderboard
* Exportable PDF reports

---

# 💡 Why This Project?

Evaluating LLMs solely through manual inspection is slow and subjective.

This framework automates the benchmarking process by combining:

* ✅ Multi-model evaluation
* ✅ LLM-as-Judge scoring
* ✅ Latency measurement
* ✅ Token usage tracking
* ✅ Interactive visual analytics

The result is a lightweight, production-style evaluation framework for comparing LLM quality and performance.

---

# 📄 License

MIT License

---

# 👨‍💻 Author

Built as a production-inspired LLM benchmarking framework demonstrating automated evaluation, LLM-as-Judge scoring, performance analysis, and interactive visualization using **Groq**, **Python**, and **Streamlit**.

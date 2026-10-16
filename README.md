# 🧠 Deal Intelligence Agent

An AI sales agent with **persistent memory**. It remembers customer interactions, learns which tactics won previous deals, and briefs a sales rep before the next call. Every step is **traced and evaluated** with built-in AI observability.

Built with Python, Streamlit, [Hindsight](https://hindsight.vectorize.io) (memory), Groq (`openai/gpt-oss-120b`).

## Features

| Tab | What it does |
|---|---|
| 🏠 Overview | KPIs, pipeline diagram, **one-click demo** that loads 3 deals into memory, deal timeline |
| 📝 Capture call | Save notes + outcome; animated status shows the retain step |
| 🎯 Pre-call brief | Recalls memories, generates a structured brief (concerns, competitors, stakeholders, tactic, evidence from past deals, talking points, risks), downloadable as Markdown |
| ⚖️ Memory vs. no memory | Runs the same prompt without memories side by side to prove the value of memory |
| 💬 Ask memory | Chat with your deal history; shows exactly which memories were used |
| 📡 Observability | Latency (p50/p95), tokens, error rate, trace explorer (input/output per span), automatic evals, JSON export |

## The loop

Capture → Retain → Recall → Recommend → Observe

Example: Globex had a pricing objection that an ROI case study solved. When Acme Corp raises the same objection, Hindsight recalls Globex and the agent recommends the ROI approach, citing Globex as evidence.

## AI observability

`observability.py` records each Hindsight call and LLM call as a **span**, grouped into **runs** (one per user action), with:

- latency, prompt/completion tokens, status and errors
- full input/output per span for debugging hallucinations
- automatic **evals** per brief: `valid_json`, `memories_retrieved`, `cross_deal_learning`, `has_actionable_tactic`

The data model matches Langfuse / Arize Phoenix / LangSmith, so the same instrumentation points can be forwarded to one of them in production.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then fill in your keys
streamlit run streamlit_app.py
```

Then open **Overview → Load demo deals**, go to **Pre-call brief** (deal: `Acme Corp`) and generate the brief.

## Architecture

```text
Sales Rep → Streamlit UI ─┬─ Capture ──► Hindsight.retain
                          ├─ Brief / Chat ─► Hindsight.recall ─► Groq LLM ─► Brief + Evals
                          └─ Every call ──► Observability (spans, runs, evals)
```

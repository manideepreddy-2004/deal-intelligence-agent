"""
Lightweight, dependency-free AI observability for the Deal Intelligence Agent.

Every Hindsight call and every LLM call is recorded as a *span*; spans that belong
to one user action (e.g. "Generate pre-call brief") are grouped into a *run*.
Each run can also carry automatic quality checks (evals).

This mirrors the data model of tools like Langfuse / Arize Phoenix / LangSmith,
so the same instrumentation points can later be forwarded to one of them.
"""
import time
import uuid
from contextlib import contextmanager
from datetime import datetime

import streamlit as st


def _store():
    st.session_state.setdefault("obs_spans", [])
    st.session_state.setdefault("obs_runs", {})
    return st.session_state


def _clip(value, n=600):
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= n else text[:n] + " …"


@contextmanager
def run(name: str, **meta):
    """Group several spans into one user-level action."""
    s = _store()
    run_id = uuid.uuid4().hex[:8]
    s["obs_runs"][run_id] = {
        "id": run_id,
        "name": name,
        "ts": datetime.now().strftime("%H:%M:%S"),
        "meta": meta,
        "evals": {},
        "total_ms": None,
    }
    prev = s.get("obs_current_run")
    s["obs_current_run"] = run_id
    t0 = time.perf_counter()
    try:
        yield s["obs_runs"][run_id]
    finally:
        s["obs_runs"][run_id]["total_ms"] = round((time.perf_counter() - t0) * 1000)
        s["obs_current_run"] = prev


@contextmanager
def span(name: str, kind: str, input=None, **meta):
    """kind: 'memory' (Hindsight) | 'llm' (Groq) | 'app'"""
    s = _store()
    rec = {
        "id": uuid.uuid4().hex[:8],
        "run_id": s.get("obs_current_run"),
        "ts": datetime.now().strftime("%H:%M:%S"),
        "name": name,
        "kind": kind,
        "input": _clip(input),
        "output": "",
        "status": "ok",
        "error": "",
        "latency_ms": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "meta": dict(meta),
    }
    t0 = time.perf_counter()
    try:
        yield rec
    except Exception as exc:  # noqa: BLE001 - we re-raise
        rec["status"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec["latency_ms"] = round((time.perf_counter() - t0) * 1000)
        rec["output"] = _clip(rec["output"])
        s["obs_spans"].append(rec)


def set_eval(run_id: str, results: dict):
    _store()["obs_runs"].get(run_id, {}).setdefault("evals", {}).update(results)


def all_spans():
    return list(_store()["obs_spans"])


def all_runs():
    return list(_store()["obs_runs"].values())


def reset():
    st.session_state["obs_spans"] = []
    st.session_state["obs_runs"] = {}


def percentile(values, p):
    if not values:
        return 0
    v = sorted(values)
    k = min(len(v) - 1, int(round((p / 100) * (len(v) - 1))))
    return v[k]

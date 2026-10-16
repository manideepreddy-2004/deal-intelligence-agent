import json
import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from hindsight_client import Hindsight

import observability as obs

load_dotenv()

BANK_ID = "deal-intelligence"
MODEL = "openai/gpt-oss-120b"

st.set_page_config(page_title="Deal Intelligence Agent", page_icon="🧠", layout="wide")

# ------------------------------------------------------------------ STYLE
st.markdown(
    """
<style>
.hero {padding: 28px 32px; border-radius: 18px; margin-bottom: 18px;
  background: linear-gradient(120deg, rgba(99,102,241,.22), rgba(236,72,153,.16) 60%, rgba(14,165,233,.16));
  border: 1px solid rgba(128,128,128,.25);}
.hero h1 {margin: 0; font-size: 2.2rem;}
.hero p {margin: 6px 0 0 0; opacity: .8; font-size: 1.05rem;}
.kpi {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 14px 16px;
  background: rgba(128,128,128,.07);}
.kpi .v {font-size: 1.7rem; font-weight: 700; line-height: 1.1;}
.kpi .l {font-size: .78rem; opacity: .7; text-transform: uppercase; letter-spacing: .06em;}
.pipe {display:flex; gap:10px; flex-wrap:wrap; margin: 6px 0 14px 0;}
.pipe div {flex:1; min-width:140px; text-align:center; padding:12px 8px; border-radius:12px;
  border:1px solid rgba(128,128,128,.3); background: rgba(99,102,241,.10); font-weight:600;}
.pipe small {display:block; font-weight:400; opacity:.7; margin-top:2px;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 14px 18px;
  margin-bottom: 12px; background: rgba(128,128,128,.06);}
.card h4 {margin: 0 0 8px 0; font-size: .85rem; text-transform: uppercase;
  letter-spacing: .07em; opacity: .75;}
.tactic {border-left: 5px solid #6366f1; background: rgba(99,102,241,.12);}
.chip {display:inline-block; padding:3px 10px; margin:2px 4px 2px 0; border-radius:999px;
  font-size:.82rem; background: rgba(99,102,241,.18); border:1px solid rgba(99,102,241,.35);}
.chip.red {background: rgba(239,68,68,.15); border-color: rgba(239,68,68,.4);}
.chip.green {background: rgba(34,197,94,.15); border-color: rgba(34,197,94,.4);}
.mem {border-left: 4px solid #0ea5e9; padding: 8px 12px; margin-bottom: 8px;
  border-radius: 6px; background: rgba(14,165,233,.08); font-size: .92rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ CLIENTS
missing = [k for k in ("HINDSIGHT_API_URL", "HINDSIGHT_API_KEY", "GROQ_API_KEY") if not os.getenv(k)]
if missing:
    st.error(f"Missing environment variables: {', '.join(missing)}. Copy `.env.example` to `.env` and fill it in.")
    st.stop()


@st.cache_resource
def get_clients():
    return (
        Hindsight(base_url=os.getenv("HINDSIGHT_API_URL"), api_key=os.getenv("HINDSIGHT_API_KEY")),
        Groq(api_key=os.getenv("GROQ_API_KEY")),
    )


hindsight, groq = get_clients()

# ------------------------------------------------------------------ STATE
st.session_state.setdefault("calls", [])  # calls captured this session (timeline)
st.session_state.setdefault("chat", [])
st.session_state.setdefault("last_brief", None)

DEMO_CALLS = [
    {
        "deal": "Globex",
        "notes": "Globex's CFO said our pricing is too high compared to their current vendor. "
        "Procurement and the VP of Operations are involved in the decision.",
        "outcome": "We presented an ROI case study showing 30% lower operational costs. "
        "The CFO re-engaged and the customer moved forward to contract review.",
    },
    {
        "deal": "Initech",
        "notes": "Initech's champion left the company mid-deal and the evaluation stalled. "
        "They were also comparing us with HubSpot.",
        "outcome": "Re-engaged through an executive sponsor (their COO) with a 20-minute business review. "
        "Deal restarted and closed within 6 weeks.",
    },
    {
        "deal": "Acme Corp",
        "notes": "Acme Corp says pricing is too high and they are evaluating Salesforce. "
        "Their Head of Sales Ops is the main contact; Finance must approve any purchase.",
        "outcome": "Outcome not recorded yet. Follow-up call is being scheduled.",
    },
]


# ------------------------------------------------------------------ HELPERS
def mem_texts(result):
    items = getattr(result, "results", result) or []
    return [getattr(m, "text", str(m)) for m in items]


def retain_call(deal, notes, outcome):
    content = (
        f"Deal: {deal}\n\nSales call:\n{notes}\n\n"
        f"Outcome:\n{outcome.strip() or 'Outcome not recorded yet.'}\n"
    )
    with obs.span("hindsight.retain", "memory", input=content) as sp:
        hindsight.retain(bank_id=BANK_ID, content=content)
        sp["output"] = "stored"
    st.session_state["calls"].append(
        {"time": datetime.now().strftime("%H:%M"), "deal": deal, "notes": notes, "outcome": outcome}
    )


def recall(query):
    with obs.span("hindsight.recall", "memory", input=query) as sp:
        res = hindsight.recall(bank_id=BANK_ID, query=query)
        texts = mem_texts(res)
        sp["meta"]["memories_returned"] = len(texts)
        sp["output"] = " | ".join(t.replace("\n", " ")[:120] for t in texts[:3])
    return texts


def llm(name, prompt, temperature=0.2):
    with obs.span(name, "llm", input=prompt, model=MODEL) as sp:
        resp = groq.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=temperature
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        sp["prompt_tokens"] = getattr(usage, "prompt_tokens", 0) or 0
        sp["completion_tokens"] = getattr(usage, "completion_tokens", 0) or 0
        sp["output"] = text
    return text


def parse_json(text):
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                return None
    return None


BRIEF_SCHEMA = """{
  "customer": "string",
  "summary": "1-2 sentence situation summary",
  "key_concerns": ["string"],
  "competitors": ["string"],
  "stakeholders": ["string"],
  "recommended_tactic": "string",
  "why_this_tactic": "string",
  "evidence_from_past_deals": ["Deal name - what happened and what worked"],
  "talking_points": ["string"],
  "risks": ["string"]
}"""


def brief_prompt(deal, memories):
    mem = "\n".join(f"- {m}" for m in memories) if memories else "(no memories available)"
    return f"""You are a sales intelligence agent.
Prepare a pre-call briefing for the next call with {deal}.
Use ONLY the retrieved sales memories below. If information is missing, use an empty list or say "unknown". Never invent facts.
If a successful tactic from ANOTHER deal is relevant, name that deal explicitly in evidence_from_past_deals.

Retrieved memories:
{mem}

Respond with ONLY valid JSON matching this schema:
{BRIEF_SCHEMA}"""


def run_evals(deal, brief, memories):
    """Automatic quality checks, logged against the run (a mini LLM-eval suite)."""
    blob = json.dumps(brief or {}).lower()
    known = {
        m.group(1).strip()
        for t in memories
        for m in re.finditer(r"Deal:\s*([^\n]+)", t)
    }
    others = [d for d in known if d.lower() != deal.lower()]
    return {
        "valid_json": brief is not None,
        "memories_retrieved": len(memories) > 0,
        "cross_deal_learning": any(d.lower() in blob for d in others) if others else False,
        "has_actionable_tactic": bool(brief and str(brief.get("recommended_tactic", "")).strip()
                                      and str(brief.get("recommended_tactic")).lower() != "unknown"),
    }


def chips(items, cls=""):
    if not items:
        return "<span style='opacity:.6'>None found in memory</span>"
    return "".join(f"<span class='chip {cls}'>{i}</span>" for i in items)


def render_brief(b, deal):
    st.markdown(
        f"<div class='card'><h4>Situation · {b.get('customer', deal)}</h4>{b.get('summary', '')}</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='card'><h4>Key concerns</h4>{chips(b.get('key_concerns'), 'red')}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><h4>Competitors</h4>{chips(b.get('competitors'))}</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><h4>Stakeholders</h4>{chips(b.get('stakeholders'))}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='card tactic'><h4>🎯 Recommended tactic</h4><b>{b.get('recommended_tactic', '')}</b>"
        f"<br><span style='opacity:.85'>{b.get('why_this_tactic', '')}</span></div>",
        unsafe_allow_html=True,
    )
    d1, d2 = st.columns(2)
    ev = "".join(f"<div class='mem'>{e}</div>" for e in b.get("evidence_from_past_deals", [])) or "<i>No past-deal evidence</i>"
    d1.markdown(f"<div class='card'><h4>🧠 Evidence from past deals</h4>{ev}</div>", unsafe_allow_html=True)
    tp = "".join(f"<li>{t}</li>" for t in b.get("talking_points", []))
    rk = "".join(f"<li>{t}</li>" for t in b.get("risks", []))
    d2.markdown(
        f"<div class='card'><h4>💬 Talking points</h4><ul>{tp}</ul><h4>⚠️ Risks</h4><ul>{rk}</ul></div>",
        unsafe_allow_html=True,
    )


def brief_markdown(b, deal):
    L = lambda xs: "\n".join(f"- {x}" for x in xs) or "- none"
    return (
        f"# Pre-call brief: {deal}\n\n{b.get('summary','')}\n\n"
        f"## Key concerns\n{L(b.get('key_concerns', []))}\n\n"
        f"## Competitors\n{L(b.get('competitors', []))}\n\n"
        f"## Stakeholders\n{L(b.get('stakeholders', []))}\n\n"
        f"## Recommended tactic\n{b.get('recommended_tactic','')}\n\n{b.get('why_this_tactic','')}\n\n"
        f"## Evidence from past deals\n{L(b.get('evidence_from_past_deals', []))}\n\n"
        f"## Talking points\n{L(b.get('talking_points', []))}\n\n"
        f"## Risks\n{L(b.get('risks', []))}\n"
    )


# ------------------------------------------------------------------ SIDEBAR
st.sidebar.title("💼 Deal Workspace")
deal_name = st.sidebar.text_input("Current deal", value="Acme Corp")
st.sidebar.markdown("---")
st.sidebar.markdown("### Connections")
st.sidebar.markdown("🟢 Hindsight memory  \n🟢 Groq · `gpt-oss-120b`  \n🟢 Built-in observability")
st.sidebar.markdown("---")
st.sidebar.markdown("### Memory loop")
st.sidebar.markdown("📝 Capture → 🧠 Retain → 🔎 Recall → 🎯 Recommend → 📡 Observe")

# ------------------------------------------------------------------ HERO
st.markdown(
    """
<div class="hero"><h1>🧠 Deal Intelligence Agent</h1>
<p>A sales agent with persistent memory: it learns what worked in past deals and briefs you before the next call.</p></div>
""",
    unsafe_allow_html=True,
)

tab_home, tab_capture, tab_brief, tab_chat, tab_obs = st.tabs(
    ["🏠 Overview", "📝 Capture call", "🎯 Pre-call brief", "💬 Ask memory", "📡 Observability"]
)

# ------------------------------------------------------------------ OVERVIEW
with tab_home:
    spans = obs.all_spans()
    llm_spans = [s for s in spans if s["kind"] == "llm"]
    k = st.columns(4)
    kpis = [
        (len(st.session_state["calls"]), "Calls captured (session)"),
        (len({c["deal"] for c in st.session_state["calls"]}), "Deals tracked"),
        (len(spans), "Traced operations"),
        (f"{int(sum(s['latency_ms'] for s in llm_spans) / len(llm_spans))} ms" if llm_spans else "–", "Avg LLM latency"),
    ]
    for col, (v, l) in zip(k, kpis):
        col.markdown(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>", unsafe_allow_html=True)

    st.markdown("#### How it works")
    st.markdown(
        """<div class="pipe">
<div>📝 Capture<small>call notes + outcome</small></div>
<div>🧠 Retain<small>Hindsight stores facts</small></div>
<div>🔎 Recall<small>similar past deals</small></div>
<div>🎯 Recommend<small>LLM builds the brief</small></div>
<div>📡 Observe<small>trace, measure, evaluate</small></div></div>""",
        unsafe_allow_html=True,
    )

    st.markdown("#### 🚀 One-click demo")
    st.write(
        "Loads three deals into Hindsight: **Globex** (pricing objection solved with an ROI case study), "
        "**Initech** (champion left, solved via executive sponsor) and **Acme Corp** (the current deal). "
        "Then open **Pre-call brief** and watch the agent reuse the Globex playbook for Acme."
    )
    if st.button("Load demo deals into memory", type="primary"):
        with obs.run("Seed demo data"):
            bar = st.progress(0.0, text="Storing memories…")
            for i, c in enumerate(DEMO_CALLS, 1):
                retain_call(c["deal"], c["notes"], c["outcome"])
                bar.progress(i / len(DEMO_CALLS), text=f"Retained {c['deal']}")
        st.success("Demo deals stored in Hindsight. Memory extraction can take a few seconds.")
        st.balloons()

    if st.session_state["calls"]:
        st.markdown("#### Deal timeline (this session)")
        for c in reversed(st.session_state["calls"]):
            with st.expander(f"{c['time']} · {c['deal']}"):
                st.markdown(f"**Notes:** {c['notes']}")
                st.markdown(f"**Outcome:** {c['outcome'] or '_not recorded_'}")

# ------------------------------------------------------------------ CAPTURE
with tab_capture:
    st.subheader("Capture sales interaction")
    st.write("Save what happened and what worked. The agent learns which tactics actually win deals.")
    notes = st.text_area("Call notes", height=150,
                         placeholder="Customer says pricing is too high. They are evaluating Salesforce.")
    outcome = st.text_area("Call outcome", height=110,
                           placeholder="ROI case study reduced the pricing concern. Follow-up booked with Finance.")
    if st.button("🧠 Save call + outcome", type="primary"):
        if notes.strip():
            with obs.run("Capture call", deal=deal_name):
                with st.status("Saving to persistent memory…", expanded=True) as status:
                    st.write("📝 Formatting interaction")
                    st.write("🧠 Sending to Hindsight (retain)")
                    retain_call(deal_name, notes, outcome)
                    status.update(label="Stored in Hindsight ✅", state="complete")
        else:
            st.warning("Please enter call notes first.")

# ------------------------------------------------------------------ BRIEF
with tab_brief:
    st.subheader("Prepare for the next call")
    st.write(f"Generate a briefing for **{deal_name}** from everything the agent remembers.")
    compare = st.toggle("⚖️ Compare with a memory-less agent", value=True,
                        help="Runs the same prompt without Hindsight memories so you can see the difference.")

    if st.button("🎯 Generate pre-call brief", type="primary"):
        with obs.run("Pre-call brief", deal=deal_name) as r:
            with st.status("Agent working…", expanded=True) as status:
                st.write("🔎 Searching Hindsight memory…")
                memories = recall(
                    f"Prepare for the next sales call with {deal_name}. Find pricing concerns, competitors, "
                    "stakeholders, previous discussions, successful tactics and relevant experiences from similar deals."
                )
                st.write(f"🧠 Recalled {len(memories)} memories")
                st.write("🤖 Turning memory into a brief…")
                brief = parse_json(llm("llm.brief_with_memory", brief_prompt(deal_name, memories)))
                baseline = None
                if compare:
                    st.write("⚖️ Running memory-less baseline…")
                    baseline = parse_json(llm("llm.brief_no_memory", brief_prompt(deal_name, [])))
                evals = run_evals(deal_name, brief, memories)
                obs.set_eval(r["id"], evals)
                status.update(label="Brief ready ✅", state="complete")
        st.session_state["last_brief"] = {
            "deal": deal_name, "brief": brief, "memories": memories, "baseline": baseline, "evals": evals,
        }

    lb = st.session_state["last_brief"]
    if lb:
        with st.expander(f"🧠 Hindsight memories recalled ({len(lb['memories'])})", expanded=False):
            for m in lb["memories"][:10]:
                st.markdown(f"<div class='mem'>{m}</div>", unsafe_allow_html=True)
            if not lb["memories"]:
                st.warning("No relevant memories found. Load the demo deals on the Overview tab.")
        if lb["brief"]:
            render_brief(lb["brief"], lb["deal"])
            st.download_button("⬇️ Download brief (.md)", brief_markdown(lb["brief"], lb["deal"]),
                               file_name=f"brief-{lb['deal'].replace(' ', '-').lower()}.md")
        else:
            st.error("The model did not return valid JSON. See the Observability tab for the raw output.")
        if lb["baseline"]:
            st.markdown("### ⚖️ Memory vs. no memory")
            a, b = st.columns(2)
            with a:
                st.markdown("**🧠 With Hindsight memory**")
                st.success((lb["brief"] or {}).get("recommended_tactic", "–"))
                for e in (lb["brief"] or {}).get("evidence_from_past_deals", []):
                    st.caption(f"✔ {e}")
            with b:
                st.markdown("**🚫 Without memory**")
                st.warning(lb["baseline"].get("recommended_tactic", "–"))
                st.caption("No past-deal evidence available.")

# ------------------------------------------------------------------ CHAT
with tab_chat:
    st.subheader("Ask your deal memory")
    st.caption("e.g. “Which tactic worked for pricing objections?” · “Who are the stakeholders at Globex?”")
    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("memories"):
                with st.expander("Memories used"):
                    for t in m["memories"][:8]:
                        st.markdown(f"<div class='mem'>{t}</div>", unsafe_allow_html=True)
    q = st.chat_input("Ask about any deal…")
    if q:
        st.session_state["chat"].append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with obs.run("Ask memory", question=q):
                with st.spinner("Searching memory…"):
                    mems = recall(q)
                    mem_block = "\n".join(f"- {m}" for m in mems) or "(none)"
                    ans = llm(
                        "llm.chat_answer",
                        f"You are a sales intelligence assistant. Answer using ONLY these memories. "
                        f"If the answer isn't there, say you don't have it in memory.\n\nMemories:\n{mem_block}\n\nQuestion: {q}",
                    )
            st.markdown(ans)
            with st.expander("Memories used"):
                for t in mems[:8]:
                    st.markdown(f"<div class='mem'>{t}</div>", unsafe_allow_html=True)
        st.session_state["chat"].append({"role": "assistant", "content": ans, "memories": mems})

# ------------------------------------------------------------------ OBSERVABILITY
with tab_obs:
    st.subheader("📡 AI observability")
    st.caption("Every Hindsight and LLM call is traced: latency, tokens, errors, inputs/outputs, and automatic quality checks.")
    spans = obs.all_spans()
    runs = obs.all_runs()
    if not spans:
        st.info("No traces yet. Capture a call or generate a brief and they will appear here.")
    else:
        lat = [s["latency_ms"] for s in spans]
        errs = sum(1 for s in spans if s["status"] == "error")
        toks = sum(s["prompt_tokens"] + s["completion_tokens"] for s in spans)
        c = st.columns(5)
        for col, (v, l) in zip(c, [
            (len(spans), "Spans"), (f"{obs.percentile(lat, 50)} ms", "p50 latency"),
            (f"{obs.percentile(lat, 95)} ms", "p95 latency"), (f"{toks:,}", "Total tokens"),
            (f"{errs / len(spans):.0%}", "Error rate"),
        ]):
            col.markdown(f"<div class='kpi'><div class='v'>{v}</div><div class='l'>{l}</div></div>", unsafe_allow_html=True)

        df = pd.DataFrame(spans)
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**Latency per call (ms)**")
            st.bar_chart(df.reset_index().set_index("index")["latency_ms"])
        with g2:
            st.markdown("**Avg latency by operation (ms)**")
            st.bar_chart(df.groupby("name")["latency_ms"].mean())

        eval_runs = [r for r in runs if r["evals"]]
        if eval_runs:
            st.markdown("#### ✅ Automatic quality checks (evals)")
            names = sorted({k for r in eval_runs for k in r["evals"]})
            cols = st.columns(len(names))
            for col, n in zip(cols, names):
                vals = [r["evals"][n] for r in eval_runs if n in r["evals"]]
                col.metric(n.replace("_", " "), f"{sum(vals)}/{len(vals)} passed")
            st.caption("`cross_deal_learning` passes when the brief explicitly reuses a tactic from a different deal. "
                       "This is the behaviour memory is supposed to enable.")

        st.markdown("#### 🔍 Trace explorer")
        for r in reversed(runs):
            rs = [s for s in spans if s["run_id"] == r["id"]]
            if not rs:
                continue
            with st.expander(f"{r['ts']} · {r['name']} · {r['total_ms']} ms · {len(rs)} spans"):
                if r["evals"]:
                    st.markdown(" ".join(f"{'✅' if v else '❌'} {k}" for k, v in r["evals"].items()))
                for s in rs:
                    icon = {"memory": "🧠", "llm": "🤖"}.get(s["kind"], "⚙️")
                    st.markdown(
                        f"**{icon} {s['name']}** · {s['latency_ms']} ms · "
                        f"{s['prompt_tokens']}→{s['completion_tokens']} tokens · "
                        f"{'🟢 ok' if s['status'] == 'ok' else '🔴 ' + s['error']}"
                    )
                    t1, t2 = st.tabs(["Input", "Output"])
                    t1.code(s["input"] or "–", language="text")
                    t2.code(s["output"] or "–", language="text")

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Export traces (JSON)", json.dumps({"runs": runs, "spans": spans}, indent=2),
                           file_name="traces.json")
        if d2.button("🗑️ Clear traces"):
            obs.reset()
            st.rerun()

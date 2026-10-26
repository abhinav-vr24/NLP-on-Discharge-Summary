"""
app_discharge.py - a live Streamlit showcase of the Discharge Summary Assistant
=================================================================================
Run:   pip install streamlit pandas          (plus the packages in requirements.txt)
       python make_eval_set.py               (once: makes the synthetic test set for the Performance tab)
       streamlit run app_discharge.py

What you see
  Brief        the clinician brief, recomputed instantly every time you edit the note
  Source       the full note with the summary sentences and medication lines highlighted
  Performance  precision / recall / F1 on the synthetic test set, with baselines (see metrics.py)

Privacy: this app runs on your computer and sends nothing anywhere (see .streamlit/config.toml).
Use synthetic or properly de-identified notes only. It is decision support, not a medical device.
"""

import html
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from discharge_assistant import analyze, extract_medications
from metrics import evaluate, load_eval

st.set_page_config(page_title="Discharge Summary Assistant", layout="wide")
st.title("Discharge Summary Assistant")
st.caption("Turns a free-text discharge summary into a one-page brief. Decision support only: "
           "always verify against the full note. Runs locally; nothing leaves this computer.")

HERE = Path(__file__).parent
STATUS_ICON = {"NEW": "🟢 NEW", "CHANGED": "🟠 CHANGED", "STOPPED": "🔴 STOPPED",
               "CONTINUE": "⚪ CONTINUE", "UNSPECIFIED": "❔ no status"}
STATUS_COLOR = {"NEW": "#d9f2e0", "CHANGED": "#fbeccc", "STOPPED": "#f8dde2", "CONTINUE": "#eef0f3", "UNSPECIFIED": "#e8e8e8"}


# ----------------------------------------------------------------------------------------
# Sidebar: choose a note and a summary length
# ----------------------------------------------------------------------------------------
st.sidebar.header("Note")
source = st.sidebar.radio("Source", ["Sample note", "Test-set note", "Paste text", "Upload .txt"])

if source == "Sample note":
    note = (HERE / "sample_discharge_summary.txt").read_text(encoding="utf-8")
elif source == "Test-set note":
    if not (HERE / "evaluation_set.json").exists():
        st.sidebar.error("Run  python make_eval_set.py  first.")
        st.stop()
    eval_data = load_eval()
    pick = st.sidebar.selectbox("Synthetic note", [d["id"] for d in eval_data])
    note = next(d["text"] for d in eval_data if d["id"] == pick)
elif source == "Paste text":
    note = st.sidebar.text_area("Paste a synthetic or de-identified note", height=280)
else:
    up = st.sidebar.file_uploader("Text file", type=["txt"])
    note = up.read().decode("utf-8", errors="replace") if up else ""

k = st.sidebar.slider("Summary length (sentences)", 1, 8, 4)

if not note.strip():
    st.info("Choose a sample note, or paste or upload one in the sidebar.")
    st.stop()


# ----------------------------------------------------------------------------------------
# Run the assistant and time it (this is the "real time" part: it reruns on every change)
# ----------------------------------------------------------------------------------------
t0 = time.perf_counter()
result = analyze(note, n_sentences=k)
elapsed_ms = (time.perf_counter() - t0) * 1000
st.sidebar.metric("Analysis time", f"{elapsed_ms:.0f} ms")


def flexible(text):
    """Regex that matches `text` even if line breaks or spaces differ from the original note."""
    return r"\s+".join(re.escape(w) for w in text.split())


def highlight_note(note, summary, meds_on):
    """HTML version of the note: summary sentences in yellow, medication lines coloured by status."""
    marks = []                                                  # (start, end, colour)
    for s in summary:
        m = re.search(flexible(s), note) if s.strip() else None
        if m:
            marks.append((m.start(), m.end(), "#fff3a8"))
    if meds_on:
        pos = 0
        for line in note.splitlines(keepends=True):
            found = extract_medications(line.strip())
            if found:
                marks.append((pos, pos + len(line.rstrip("\n")), STATUS_COLOR[found[0]["status"]]))
            pos += len(line)
    out, last = [], 0
    for s, e, color in sorted(marks):
        if s < last:
            continue                                            # skip overlaps
        out.append(html.escape(note[last:s]))
        out.append(f"<span style='background:{color};border-radius:4px;padding:1px 3px'>{html.escape(note[s:e])}</span>")
        last = e
    out.append(html.escape(note[last:]))
    return "<div style='white-space:pre-wrap;font-family:monospace;font-size:0.85rem;line-height:1.7'>" + "".join(out) + "</div>"


tab_brief, tab_source, tab_perf = st.tabs(["Brief", "Source with highlights", "Performance"])

# ----------------------------------------------------------------------------------------
# TAB 1: the brief
# ----------------------------------------------------------------------------------------
with tab_brief:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Medication changes")
        meds = result["medications"]
        if not meds:
            st.warning("No medication lines were recognised. The section header or the line format may be "
                       "unfamiliar. Check the Source tab.")
        else:
            df = pd.DataFrame(meds)
            df["status"] = df["status"].map(STATUS_ICON)
            order = {"🟢 NEW": 0, "🟠 CHANGED": 1, "🔴 STOPPED": 2, "❔ no status": 3, "⚪ CONTINUE": 4}
            df = df.sort_values("status", key=lambda s: s.map(order)).reset_index(drop=True)
            st.dataframe(df[["status", "name", "dose", "route", "freq", "note"]], hide_index=True)
            n_changed = sum(m["status"] in ("NEW", "CHANGED", "STOPPED") for m in meds)
            st.caption(f"{len(meds)} medications found, {n_changed} with a change. Changes are listed first.")

        st.subheader("Key facts")
        facts = {k_: v or "not found" for k_, v in result["key_facts"].items()}
        st.table(pd.Series(facts, name="value"))

    with right:
        st.subheader("Hospital course: key sentences")
        if result["summary"]:
            for s in result["summary"]:
                st.markdown(f"- {s}")
            st.caption("Extractive: sentences are copied from the note, never written by a model. "
                       "Do not rely on this alone for medication changes; use the table.")
        else:
            st.warning("No 'Hospital Course' section was found, so no summary was made.")

        st.subheader("Follow-up")
        for i, s in enumerate(result["follow_up"]):
            st.checkbox(s, key=f"fu{i}")
        if not result["follow_up"]:
            st.caption("No follow-up section found.")

        st.subheader("Pending results")
        if result["pending"].strip():
            st.error(f"Someone must check: {result['pending']}")
        else:
            st.success("None found in the note.")

        st.subheader("Warning signs for the patient")
        for w in result["warnings"] or ["none found"]:
            st.markdown(f"- {w}")

        st.subheader("Medical terms (scispaCy)")
        st.write(", ".join(result["terms"]) if result["terms"] is not None
                 else "scispaCy model not installed, so this optional step was skipped.")

# ----------------------------------------------------------------------------------------
# TAB 2: source text with highlights
# ----------------------------------------------------------------------------------------
with tab_source:
    st.caption("Yellow: summary sentences. Green/orange/red/grey: medication lines by status. "
               "Anything not highlighted was not used by the brief.")
    show_meds = st.checkbox("Colour medication lines", value=True)
    st.markdown(highlight_note(note, result["summary"], show_meds), unsafe_allow_html=True)
    st.write("**Sections found:**", ", ".join(result["sections"]) or "none")

# ----------------------------------------------------------------------------------------
# TAB 3: performance metrics
# ----------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Evaluating on the synthetic test set...")
def cached_eval(k_):
    return evaluate(k=k_)


with tab_perf:
    if not (HERE / "evaluation_set.json").exists():
        st.error("Run  python make_eval_set.py  to create the synthetic test set.")
        st.stop()
    m = cached_eval(k)
    st.warning("These scores come from 40 SYNTHETIC notes whose answers are known by construction. They show how "
               "the code behaves on formats we invented, not how it will perform on real hospital notes. "
               "See metrics.py for what each number means.")

    c = st.columns(4)
    c[0].metric("Sections found (recall)", f"{m['sections']['overall']['recall']:.0%}")
    c[1].metric("Medications found (recall)", f"{m['medications']['name_only']['recall']:.0%}",
                help="Share of real medication lines the assistant found. Precision is "
                     f"{m['medications']['name_only']['precision']:.0%}: what it reports is almost always right.")
    c[2].metric("Changed drugs (recall)", f"{m['medications']['changed_drugs']['recall']:.0%}",
                help="Only NEW / CHANGED / STOPPED drugs. A miss here can harm a patient.")
    c[3].metric("Median speed", f"{m['speed']['median_ms']} ms / note")

    st.subheader("Where does the medication parser fail?")
    fmt = pd.DataFrame({"recall": m["medications"]["recall_by_line_format"],
                        "lines": m["medications"]["line_counts_by_format"]})
    a, b = st.columns(2)
    a.caption("Recall by medication line format")
    a.bar_chart(fmt["recall"])
    hdr = pd.Series(m["medications"]["recall_by_header_spelling"], name="recall")
    b.caption("Recall by how the 'Discharge Medications' header was spelled")
    b.bar_chart(hdr)
    st.caption("Formats at 0% are not handled by the current regex. A header spelling at 0% means that whole "
               "section was never found. These are the first things to fix.")

    st.subheader("Is the summary better than doing nothing clever?")
    base = pd.DataFrame(m["summary"]["ranking_only"]).T
    base.columns = ["precision @ k", "key-sentence recall", "changed drugs mentioned"]
    st.dataframe(base.style.format("{:.2f}"))
    st.bar_chart(base[["key-sentence recall", "changed drugs mentioned"]])
    st.caption(f"Change the summary length in the sidebar (now {k}) and these update. "
               "'lead_k' takes the first k sentences; 'random' picks k at random. If the assistant does not "
               "clearly beat both, its ranking is not adding value. Note how the assistant does not mention "
               "changed drugs more often than random sentences do: use the medication table for that.")

    st.subheader("Everything else")
    other = pd.DataFrame({
        "Follow-up sentences": m["other"]["follow_up"], "Warning signs": m["other"]["warning_signs"],
        "Pending results (yes/no)": m["other"]["pending_results"]}).T
    st.dataframe(other[["precision", "recall", "f1", "tp", "fp", "fn"]])

    with st.expander("Examples of medications the assistant missed"):
        st.dataframe(pd.DataFrame(m["medications"]["missed_examples"]), hide_index=True)
    with st.expander("How to read precision, recall and F1"):
        st.markdown("""
* **Recall**: of everything that was really there, how much did we find? A low value means things are being **missed**.
* **Precision**: of everything we reported, how much was right? A low value means **wrong** items are being added.
* **F1**: one number balancing both.
* For medication changes and allergies, **recall matters most**. For alerts that interrupt clinicians, **precision** does.
""")

"""
Discharge Summary Assistant
===========================
Reads a US-style hospital discharge summary and produces a short,
one-page "clinician brief" containing:

  1. Key facts      (dates, diagnosis, allergies, condition)
  2. Medication changes (NEW / CHANGED / STOPPED / CONTINUE)
  3. A short extractive summary of the hospital course
  4. Follow-up tasks and pending results (things easy to miss!)
  5. Patient warning signs / return precautions
  6. (Optional) medical terms found by scispaCy

Design choice: this tool NEVER writes new medical text. It only
pulls out and re-orders sentences that are already in the note.
That means it cannot "hallucinate" a drug or a diagnosis.

Run:
    python discharge_assistant.py                       # uses the sample file
    python discharge_assistant.py my_note.txt           # your own (de-identified!) note
    python discharge_assistant.py my_note.txt --save brief.txt
"""

import re
import argparse
from pathlib import Path

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


# ----------------------------------------------------------------------
# STEP 0: Settings you can tweak
# ----------------------------------------------------------------------

# US discharge summaries usually use these section titles.
# Add more here if your hospital uses different names.
SECTION_HEADERS = [
    "Admission Date", "Discharge Date", "Chief Complaint",
    "Principal Diagnosis", "Hospital Course", "Discharge Medications",
    "Allergies", "Follow-up", "Pending Results",
    "Discharge Condition", "Discharge Instructions",
]

# Words that tell us a medication line is new/changed/stopped/continued.
MED_STATUS_WORDS = ["NEW", "CHANGED", "STOPPED", "CONTINUE"]

# Sentences containing these words are more "important" in the hospital course.
CLINICAL_KEYWORDS = [
    "diagnosed", "started", "held", "stopped", "restarted", "increased",
    "decreased", "resolved", "improved", "worsened", "elevated", "showed",
    "weaned", "rose", "returned", "discharged", "history", "presented",
]

# How many sentences to keep in the hospital-course summary.
SUMMARY_SENTENCES = 4


# ----------------------------------------------------------------------
# STEP 1: Load the NLP tools
# ----------------------------------------------------------------------

def load_nlp():
    """
    Returns (nlp_for_sentences, nlp_for_medical_terms_or_None).

    - spacy.blank("en") + sentencizer: lightweight sentence splitter.
      Needs NO model download, so the project always runs.
    - scispaCy "en_core_sci_sm": optional biomedical model. If it is not
      installed we simply skip the medical-terms section.
    """
    sentence_nlp = spacy.blank("en")
    sentence_nlp.add_pipe("sentencizer")

    try:
        medical_nlp = spacy.load("en_core_sci_sm")
    except OSError:
        medical_nlp = None  # scispaCy model not installed -> optional feature off

    return sentence_nlp, medical_nlp


# ----------------------------------------------------------------------
# STEP 2: Split the note into sections
# ----------------------------------------------------------------------

def split_sections(text):
    """
    Turn the raw note into a dictionary: {"Hospital Course": "...", ...}

    We look for lines that start with a known header followed by a colon,
    e.g. "Allergies: Penicillin (rash)."  Everything until the next header
    belongs to that section.
    """
    # Build one regex like:  ^(Admission Date|Discharge Date|...)\s*:\s*(.*)$
    header_pattern = re.compile(
        r"^\s*(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r")\s*:\s*(.*)$",
        flags=re.IGNORECASE,
    )

    sections = {}
    current = None  # the section we are currently filling

    for line in text.splitlines():
        match = header_pattern.match(line)
        if match:
            # Normalise the header to our canonical spelling
            current = next(h for h in SECTION_HEADERS if h.lower() == match.group(1).lower())
            sections[current] = match.group(2).strip()
        elif current and line.strip():
            sections[current] += "\n" + line.strip()

    return sections


# ----------------------------------------------------------------------
# STEP 3: Medication table (the part clinicians care about most)
# ----------------------------------------------------------------------

# Matches lines like:  "1. Furosemide 40 mg PO BID (CHANGED - increased ...)"
MED_REGEX = re.compile(
    r"^\s*(?:\d+\.|-)?\s*"                                  # optional "1." or "-"
    r"(?P<name>[A-Za-z][A-Za-z\- ]*?)\s+"                   # drug name
    r"(?P<dose>\d+(?:\.\d+)?)\s*(?P<unit>mg|mcg|g|units|mL)\s*"  # dose + unit
    r"(?P<route>PO|IV|SC|SQ|IM|SL|inhaled)?\s*"             # route (optional)
    r"(?P<freq>daily|BID|TID|QID|QHS|PRN|weekly)?",         # frequency (optional)
    flags=re.IGNORECASE,
)


def extract_medications(med_section):
    """Parse each medication line into a small dictionary."""
    meds = []
    for line in med_section.splitlines():
        m = MED_REGEX.match(line)
        if not m:
            continue  # not a medication line

        # Find the status word in the parentheses, if any
        status = "UNSPECIFIED"
        for word in MED_STATUS_WORDS:
            if re.search(rf"\b{word}\b", line, flags=re.IGNORECASE):
                status = word
                break

        # Keep the explanation in brackets, e.g. "increased from 20 mg daily"
        note = re.search(r"\((.*?)\)", line)
        meds.append({
            "name": m.group("name").strip().title(),
            "dose": f"{m.group('dose')} {m.group('unit')}",
            "route": (m.group("route") or "").upper(),
            "freq": (m.group("freq") or "").upper(),
            "status": status,
            "note": note.group(1) if note else "",
        })
    return meds


# ----------------------------------------------------------------------
# STEP 4: Extractive summary of the hospital course
# ----------------------------------------------------------------------

def summarize(text, sentence_nlp, n_sentences=SUMMARY_SENTENCES):
    """
    Pick the n most informative sentences, keep them in original order.

    How "informative" is scored:
      - TF-IDF: words that are rare in this note but present in the
        sentence give it a higher score (e.g. "diuresed", "creatinine").
      - Bonus for clinical action words (started, held, resolved ...).
    """
    sentences = [s.text.strip() for s in sentence_nlp(text).sents if s.text.strip()]
    if len(sentences) <= n_sentences:
        return sentences  # short already, return everything

    # TF-IDF turns each sentence into a vector of word weights
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(sentences)

    # Base score = total word weight in the sentence, divided by number of
    # words so that very long sentences don't always win.
    scores = np.asarray(matrix.sum(axis=1)).ravel()
    lengths = np.array([len(s.split()) for s in sentences])
    scores = scores / np.sqrt(lengths)

    # Add a small bonus for each clinical keyword
    for i, sentence in enumerate(sentences):
        bonus = sum(kw in sentence.lower() for kw in CLINICAL_KEYWORDS)
        scores[i] += 0.15 * bonus

    # Take the top-n indexes, then sort so the story reads in order
    top = sorted(np.argsort(scores)[-n_sentences:])
    return [sentences[i] for i in top]


# ----------------------------------------------------------------------
# STEP 5: Follow-up, pending results, warning signs
# ----------------------------------------------------------------------

def split_into_sentences(text, sentence_nlp):
    return [s.text.strip() for s in sentence_nlp(text).sents if s.text.strip()]


def extract_warning_signs(instructions, sentence_nlp):
    """Return instruction sentences that tell the patient when to seek help."""
    triggers = ["return to", "call your", "call the", "seek", "emergency", "911"]
    return [
        s for s in split_into_sentences(instructions, sentence_nlp)
        if any(t in s.lower() for t in triggers)
    ]


# ----------------------------------------------------------------------
# STEP 6 (optional): Medical terms with scispaCy
# ----------------------------------------------------------------------

def extract_medical_terms(text, medical_nlp, limit=15):
    """List unique medical terms found by scispaCy (diseases, tests, drugs ...)."""
    if medical_nlp is None:
        return None
    seen, terms = set(), []
    for ent in medical_nlp(text).ents:
        key = ent.text.lower()
        if key not in seen and len(key) > 2:
            seen.add(key)
            terms.append(ent.text)
    return terms[:limit]


# ----------------------------------------------------------------------
# STEP 7: Build the clinician brief
# ----------------------------------------------------------------------

def build_brief(text):
    sentence_nlp, medical_nlp = load_nlp()
    sec = split_sections(text)

    lines = []
    add = lines.append  # shortcut

    add("=" * 60)
    add("CLINICIAN BRIEF  (auto-generated - verify against full note)")
    add("=" * 60)

    # --- Key facts
    add("\n[KEY FACTS]")
    for field in ["Admission Date", "Discharge Date", "Principal Diagnosis",
                  "Allergies", "Discharge Condition"]:
        add(f"  {field}: {sec.get(field, 'not found')}")

    # --- Medication changes: show anything that is not "CONTINUE" first
    meds = extract_medications(sec.get("Discharge Medications", ""))
    add("\n[MEDICATION CHANGES]  <- check these first")
    changed = [m for m in meds if m["status"] in ("NEW", "CHANGED", "STOPPED")]
    if changed:
        for m in changed:
            detail = f" ({m['note']})" if m["note"] else ""
            add(f"  {m['status']:<8} {m['name']} {m['dose']} {m['route']} {m['freq']}{detail}")
    else:
        add("  No changes detected.")

    unchanged = [m for m in meds if m["status"] not in ("NEW", "CHANGED", "STOPPED")]
    if unchanged:
        add("\n[CONTINUED MEDICATIONS]")
        for m in unchanged:
            add(f"  - {m['name']} {m['dose']} {m['route']} {m['freq']}")

    # --- Short hospital course summary
    add("\n[HOSPITAL COURSE - KEY SENTENCES]")
    for s in summarize(sec.get("Hospital Course", ""), sentence_nlp):
        add(f"  * {s}")

    # --- Things that are easy to forget after discharge
    add("\n[FOLLOW-UP TASKS]")
    for s in split_into_sentences(sec.get("Follow-up", "not found"), sentence_nlp):
        add(f"  [ ] {s}")

    add("\n[PENDING RESULTS - someone must check these]")
    add(f"  [ ] {sec.get('Pending Results', 'none listed')}")

    # --- Warning signs for the patient
    add("\n[WARNING SIGNS / RETURN PRECAUTIONS]")
    warnings = extract_warning_signs(sec.get("Discharge Instructions", ""), sentence_nlp)
    for w in warnings or ["none found"]:
        add(f"  ! {w}")

    # --- Optional medical terms
    terms = extract_medical_terms(
        sec.get("Principal Diagnosis", "") + ". " + sec.get("Hospital Course", ""),
        medical_nlp,
    )
    add("\n[MEDICAL TERMS FOUND (scispaCy)]")
    add("  " + (", ".join(terms) if terms is not None
                else "scispaCy model not installed - skipped (see requirements.txt)"))

    add("\n" + "-" * 60)
    add("Decision support only. Not a substitute for clinical judgment.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Command-line entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discharge summary -> clinician brief")
    parser.add_argument("file", nargs="?",
                        default=str(Path(__file__).with_name("sample_discharge_summary.txt")),
                        help="Path to a de-identified discharge summary (.txt)")
    parser.add_argument("--save", help="Optional path to save the brief as a text file")
    args = parser.parse_args()

    note = Path(args.file).read_text(encoding="utf-8")
    brief = build_brief(note)
    print(brief)

    if args.save:
        Path(args.save).write_text(brief, encoding="utf-8")
        print(f"\nSaved to {args.save}")

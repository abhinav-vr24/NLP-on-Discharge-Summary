"""
make_eval_set.py - build a SYNTHETIC test set where the correct answers are known
===================================================================================
To measure how good the assistant is, we need notes AND the right answers ("gold").
Because we write these fake notes ourselves, we know the gold answers by construction:
which medicines changed, which sentences matter, which headers exist, and so on.

The notes deliberately mix in REALISTIC formatting differences (different bullet styles,
dose ranges, header spellings). Some of these the assistant cannot handle yet. That is on
purpose: a test set that the code passes 100 percent tells you nothing.

IMPORTANT: this is a small invented set. Scores on it show how the code behaves on formats
we thought of. They do NOT show how it will perform on real hospital notes. For that, use
real de-identified notes with labels made by a clinician (see Weeks 3 and 4 of the guide).

Run:  python make_eval_set.py          (writes evaluation_set.json)
"""

import json
import random

random.seed(7)                        # fixed seed -> the same 40 notes every time

# ---- Building blocks ---------------------------------------------------------------
DRUGS = {                              # drug -> plausible doses (mg)
    "Furosemide": [20, 40, 80], "Lisinopril": [2.5, 5, 10, 20], "Metoprolol succinate": [25, 50, 100],
    "Atorvastatin": [10, 20, 40, 80], "Metformin": [500, 1000], "Spironolactone": [12.5, 25],
    "Apixaban": [2.5, 5], "Aspirin": [81], "Amlodipine": [2.5, 5, 10], "Warfarin": [2, 5],
    "Carvedilol": [3.125, 6.25, 12.5], "Sertraline": [25, 50, 100],
}
FREQS = ["daily", "BID", "TID", "QHS"]
STATUSES = ["NEW", "CHANGED", "STOPPED", "CONTINUE"]

# Medication line formats. 'supported' = our regex is expected to handle it. We do NOT use this flag
# when scoring, only to explain results afterwards.
LINE_FORMATS = {
    "numbered":      ("{i}. {name} {dose} mg PO {freq} ({status})", 0.38),
    "bracket":       ("- {name} {dose}mg PO {freq} [{status}]", 0.14),
    "dash_status":   ("{name} {dose} mg PO {freq} - {status}", 0.10),
    "lowercase":     ("{i}. {lname} {dose} mg PO {freq} ({status})", 0.10),
    "paren_number":  ("{i}) {name} {dose} mg PO {freq} ({status})", 0.10),   # '1)' is not handled
    "bullet_char":   ("\u2022 {name} {dose} mg PO {freq} ({status})", 0.10),   # a real bullet is not handled
    "dose_range":    ("{i}. {name} {range} mg PO {freq} ({status})", 0.08),   # '25-50 mg' is not handled
}

# Header spellings. The assistant knows only the canonical ones.
HEADER_VARIANTS = {
    "Discharge Medications": [("Discharge Medications", .70), ("DISCHARGE MEDICATIONS", .10),
                              ("discharge medications", .10), ("Discharge Meds", .10)],
    "Hospital Course":       [("Hospital Course", .70), ("HOSPITAL COURSE", .10),
                              ("Brief Hospital Course", .20)],
    "Follow-up":             [("Follow-up", .80), ("Followup Instructions", .20)],
    "Pending Results":       [("Pending Results", .85), ("Results Pending", .15)],
}

EVENTS = ["Chest X-ray showed pulmonary congestion.", "CT of the abdomen showed no acute findings.",
          "Echocardiogram showed an ejection fraction of 35 percent.", "Blood cultures were drawn on admission.",
          "Creatinine rose to 1.6 and returned to baseline before discharge."]
OUTCOMES = ["Symptoms improved and he was weaned to room air.", "Her pain resolved and she tolerated a regular diet.",
            "He was discharged home in stable condition."]
FILLER = ["He was seen by physical therapy.", "Diet was advanced as tolerated.",
          "Social history was reviewed with the family.",                 # contains the keyword 'history'
          "Nursing staff reviewed the medication schedule with the patient.",
          "She presented to the unit calm and cooperative.",             # contains the keyword 'presented'
          "Case management arranged transport home."]
FOLLOW = ["Primary care in {n} days with repeat basic metabolic panel.", "Cardiology clinic in {n} weeks.",
          "Heart failure nurse phone call in {n} days.", "Repeat chest X-ray in {n} weeks."]
PENDING = ["Final blood culture report pending at discharge.", "Pathology from biopsy pending."]
WARN = ["Return to the emergency department if you have severe shortness of breath or chest pain.",
        "Call your doctor if your weight increases by more than 3 pounds in one day.",
        "Seek help right away if you faint or have a fever above 101."]
NEUTRAL = ["Weigh yourself every morning.", "Follow a 2 gram sodium diet.", "Take your medicines as listed.",
           "Bring your emergency contact list to every appointment."]    # 'emergency' is a trap word


def pick(options):
    """Pick one item from [(value, weight), ...]."""
    r, acc = random.random(), 0.0
    for value, weight in options:
        acc += weight
        if r <= acc:
            return value
    return options[-1][0]


def make_note(idx):
    """Return (note_text, gold_answers) for one fake patient."""
    # --- medications -------------------------------------------------------------
    chosen = random.sample(list(DRUGS), random.randint(3, 6))
    meds, lines = [], []
    for i, name in enumerate(chosen, start=1):
        dose = random.choice(DRUGS[name])
        status = random.choice(STATUSES)
        if random.random() < 0.12:                      # some lines carry no status word at all
            status = "UNSPECIFIED"
        fmt_name = pick([(k, w) for k, (_, w) in LINE_FORMATS.items()])
        template = LINE_FORMATS[fmt_name][0]
        status_text = "" if status == "UNSPECIFIED" else status
        line = template.format(i=i, name=name, lname=name.lower(), dose=dose, freq=random.choice(FREQS),
                               status=status_text, range=f"{dose}-{dose * 2:g}")
        if status == "UNSPECIFIED":                    # remove the empty brackets or trailing dash
            line = line.replace(" ()", "").replace(" []", "").rstrip(" -")
        lines.append(line)
        meds.append({"name": name, "status": status, "dose": f"{dose} mg" if fmt_name != "dose_range" else None,
                     "format": fmt_name})

    # --- hospital course: key sentences (about medicine changes and results) and filler ----
    # The first sentence is background. Sentences about medicine changes, a test result and the outcome
    # are the KEY sentences a clinician needs. The rest is filler.
    course = [("A patient presented with worsening symptoms over several days.", False)]
    SENTENCE = {"NEW": "{name} was started during admission.",
                "CHANGED": "The dose of {name} was changed during admission.",
                "STOPPED": "{name} was stopped because of low blood pressure."}
    for m in meds:
        if m["status"] in SENTENCE:
            course.append((SENTENCE[m["status"]].format(name=m["name"]), True))
    course.append((random.choice(EVENTS), True))
    course.append((random.choice(OUTCOMES), True))
    course += [(s, False) for s in random.sample(FILLER, random.randint(2, 3))]
    first, rest = course[0], course[1:]
    random.shuffle(rest)                                    # mix the order of the rest
    course = [first] + rest

    # --- other sections --------------------------------------------------------------
    follow = [f.format(n=random.choice([2, 3, 7, 14])) for f in random.sample(FOLLOW, random.randint(1, 3))]
    has_pending = random.random() < 0.4
    warn = random.sample(WARN, random.randint(1, 2))
    neutral = random.sample(NEUTRAL, random.randint(1, 2))
    instructions = warn + neutral
    random.shuffle(instructions)

    h_meds = pick(HEADER_VARIANTS["Discharge Medications"])
    h_course = pick(HEADER_VARIANTS["Hospital Course"])
    h_follow = pick(HEADER_VARIANTS["Follow-up"])
    h_pend = pick(HEADER_VARIANTS["Pending Results"])

    out = [f"DISCHARGE SUMMARY (SYNTHETIC {idx})", "",
           "Admission Date: 03/02/2025", "Discharge Date: 03/07/2025", "",
           "Chief Complaint: Symptoms for several days.",
           "Principal Diagnosis: See hospital course.", "",
           f"{h_course}: " + " ".join(s for s, _ in course), "",
           f"{h_meds}:"] + lines + ["", "Allergies: Penicillin (rash).", "",
           f"{h_follow}: " + " ".join(follow)]
    if has_pending:
        out += ["", f"{h_pend}: {random.choice(PENDING)}"]
    out += ["", "Discharge Condition: Stable.", "", "Discharge Instructions: " + " ".join(instructions)]

    expected_headers = ["Admission Date", "Discharge Date", "Chief Complaint", "Principal Diagnosis",
                        "Hospital Course", "Discharge Medications", "Allergies", "Follow-up",
                        "Discharge Condition", "Discharge Instructions"] + (["Pending Results"] if has_pending else [])
    gold = {
        "headers": expected_headers,
        "medications": meds,
        "course_sentences": [s for s, _ in course],
        "course_key_idx": [i for i, (_, key) in enumerate(course) if key],
        "follow_up": follow,
        "has_pending": has_pending,
        "warnings": warn,
        "header_variants": {"meds": h_meds, "course": h_course, "follow": h_follow, "pending": h_pend if has_pending else None},
    }
    return "\n".join(out), gold


if __name__ == "__main__":
    data = []
    for i in range(1, 41):
        text, gold = make_note(i)
        data.append({"id": f"note_{i:02d}", "text": text, "gold": gold})
    with open("evaluation_set.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    n_med = sum(len(d["gold"]["medications"]) for d in data)
    print(f"Wrote {len(data)} synthetic notes with {n_med} medication lines to evaluation_set.json")
    print(data[0]["text"][:900])

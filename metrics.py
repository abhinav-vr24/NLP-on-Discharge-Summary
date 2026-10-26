"""
metrics.py - how good is the discharge assistant? (performance metrics, explained)
==================================================================================
Run:  python metrics.py                 # prints a report and saves metrics_report.json
      python metrics.py --k 3           # evaluate a 3-sentence summary instead of 4

HOW TO READ THE NUMBERS (plain language)
----------------------------------------
Each part of the assistant makes two kinds of mistakes:

  * It can MISS something real.        -> measured by RECALL    = found / (everything that was there)
  * It can ADD something that is wrong -> measured by PRECISION = correct / (everything it reported)

  F1 is one number that balances the two (it is low if either is low).

Which matters more depends on the job:
  * Medication changes and allergies: a MISS can harm a patient, so watch RECALL.
  * Alerts that interrupt a busy clinician: too many wrong flags make people ignore the tool,
    so watch PRECISION.

WHAT THESE SCORES ARE (and are not)
-----------------------------------
They are measured on SYNTHETIC notes whose right answers are known by construction (see
make_eval_set.py). They show how the code behaves on formats we invented. They are NOT a clinical
validation. To trust the tool on real notes you need real de-identified notes labelled by a clinician.
"""

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from discharge_assistant import analyze

EVAL_FILE = "evaluation_set.json"
REPORT_FILE = "metrics_report.json"
CHANGED = ("NEW", "CHANGED", "STOPPED")        # the medication changes a clinician must not miss


# ----------------------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------------------
def prf(tp, fp, fn):
    """Precision, recall and F1 from counts. Returns 0.0 (not an error) when a ratio is undefined."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn}


def load_eval():
    if not Path(EVAL_FILE).exists():
        raise SystemExit(f"{EVAL_FILE} not found. Create it first:  python make_eval_set.py")
    return json.loads(Path(EVAL_FILE).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------------------
# METRIC 1: Did we find the sections of the note?
# ----------------------------------------------------------------------------------------
def section_metrics(results):
    """
    Everything downstream depends on finding the right sections, so we measure it first.
    Recall per header = in how many notes that really had the header did we find it?
    """
    per_header = defaultdict(lambda: [0, 0])                  # header -> [found, expected]
    tp = fp = fn = 0
    for item, out in results:
        expected, found = set(item["gold"]["headers"]), set(out["sections"])
        tp += len(expected & found); fp += len(found - expected); fn += len(expected - found)
        for h in expected:
            per_header[h][1] += 1
            per_header[h][0] += h in found
    table = {h: round(f / e, 3) for h, (f, e) in sorted(per_header.items())}
    return {"overall": prf(tp, fp, fn), "recall_by_header": table}


# ----------------------------------------------------------------------------------------
# METRIC 2: Medications (the highest-risk part)
# ----------------------------------------------------------------------------------------
def medication_metrics(results):
    """
    A medication is "found" if we report the same drug name. It is "found correctly" if the
    status (NEW / CHANGED / STOPPED / CONTINUE) also matches.

    Why several numbers:
      * name_only        - did we see the drug at all?
      * name_and_status  - did we also get the change right? (a drug marked CONTINUE when it was
                           STOPPED is a dangerous error even though the name is right)
      * changed_drugs    - recall for ONLY the drugs that changed. This is the safety number.
      * dose_accuracy    - among drugs found, was the dose text right?
    """
    name_c, ns_c, chg_c = [0, 0, 0], [0, 0, 0], [0, 0, 0]       # [tp, fp, fn]
    status_ok = status_n = dose_ok = dose_n = 0
    by_format = defaultdict(lambda: [0, 0])                     # line format -> [found, total]
    by_header = defaultdict(lambda: [0, 0])                     # medication header spelling -> [found, total]
    misses = []                                                 # examples of missed drugs
    for item, out in results:
        gold = {m["name"].lower(): m for m in item["gold"]["medications"]}
        pred = {m["name"].lower(): m for m in out["medications"]}
        for name, g in gold.items():
            hit = name in pred
            by_format[g["format"]][0] += hit; by_format[g["format"]][1] += 1
            hv = item["gold"]["header_variants"]["meds"]
            by_header[hv][0] += hit; by_header[hv][1] += 1
            if not hit:
                misses.append({"note": item["id"], "drug": g["name"], "line_format": g["format"], "header": hv})
        # name only
        name_c[0] += len(gold.keys() & pred.keys()); name_c[1] += len(pred.keys() - gold.keys())
        name_c[2] += len(gold.keys() - pred.keys())
        # name + status
        g_ns = {(n, m["status"]) for n, m in gold.items()}
        p_ns = {(n, m["status"]) for n, m in pred.items()}
        ns_c[0] += len(g_ns & p_ns); ns_c[1] += len(p_ns - g_ns); ns_c[2] += len(g_ns - p_ns)
        # only the changed drugs
        g_c = {(n, m["status"]) for n, m in gold.items() if m["status"] in CHANGED}
        chg_c[0] += len(g_c & p_ns); chg_c[2] += len(g_c - p_ns)
        chg_c[1] += len({(n, m["status"]) for n, m in pred.items() if m["status"] in CHANGED} - g_c)
        # accuracy on matched drugs
        for n in gold.keys() & pred.keys():
            status_n += 1; status_ok += gold[n]["status"] == pred[n]["status"]
            if gold[n]["dose"] is not None:
                dose_n += 1; dose_ok += gold[n]["dose"] == pred[n]["dose"]
    return {
        "name_only": prf(*name_c), "name_and_status": prf(*ns_c), "changed_drugs": prf(*chg_c),
        "status_accuracy_when_found": round(status_ok / status_n, 3) if status_n else 0.0,
        "dose_accuracy_when_found": round(dose_ok / dose_n, 3) if dose_n else 0.0,
        "recall_by_line_format": {k: round(f / t, 3) for k, (f, t) in sorted(by_format.items())},
        "line_counts_by_format": {k: t for k, (f, t) in sorted(by_format.items())},
        "recall_by_header_spelling": {k: round(f / t, 3) for k, (f, t) in sorted(by_header.items())},
        "missed_examples": misses[:12],
    }


# ----------------------------------------------------------------------------------------
# METRIC 3: Is the summary any good?
# ----------------------------------------------------------------------------------------
def summary_metrics(data, k, results_for_k):
    """
    Gold = the "key" sentences (medicine changes, a test result, the outcome) written into each fake note.
    We ask: of the k sentences the summary chose, how many are key?  (precision@k)
            of the key sentences, how many did the summary include?  (recall@k)

    A score on its own means little, so we compare with two BASELINES that need no intelligence:
      * lead-k : just take the first k sentences (a surprisingly strong baseline in real news text)
      * random : take k random sentences (the floor: anything below this is useless)
    If the assistant does not clearly beat both, the TF-IDF ranking is not earning its keep.

    Two views:
      * ranking_only : only notes where the Hospital Course section was found. Measures the ranking itself.
      * end_to_end   : all notes. A note whose section was missed scores zero. Measures the real pipeline.
    """
    rnd = random.Random(0)
    sums = defaultdict(lambda: [0, 0, 0])      # method -> [hits, predicted, key_total]
    covered = defaultdict(lambda: [0, 0])      # method -> [changed drugs mentioned, changed drugs total]
    words_sum = [0, 0]                         # [summary words, course words]
    all_key = 0
    sys_hits_all = 0
    unmatched = 0
    for item, out in results_for_k:
        gold = item["gold"]
        sents, key = gold["course_sentences"], set(gold["course_key_idx"])
        changed_names = [m["name"].lower() for m in gold["medications"] if m["status"] in CHANGED]
        all_key += len(key)
        pred_sents = [s.strip() for s in out["summary"]]
        pred_idx = {sents.index(s) for s in pred_sents if s in sents}
        unmatched += sum(1 for s in pred_sents if s not in sents)
        sys_hits_all += len(pred_idx & key)
        if "Hospital Course" not in out["sections"]:
            continue                                            # ranking_only skips this note
        methods = {"assistant": pred_idx,
                   "lead_k": set(range(min(k, len(sents)))),
                   "random": set(rnd.sample(range(len(sents)), min(k, len(sents))))}
        for name, idx in methods.items():
            sums[name][0] += len(idx & key); sums[name][1] += len(idx); sums[name][2] += len(key)
            text = " ".join(sents[i] for i in idx).lower()
            covered[name][0] += sum(1 for n in changed_names if n in text); covered[name][1] += len(changed_names)
        words_sum[0] += sum(len(s.split()) for s in pred_sents); words_sum[1] += sum(len(s.split()) for s in sents)
    ranking = {}
    for name, (hits, pred, keys) in sums.items():
        p = hits / pred if pred else 0.0; r = hits / keys if keys else 0.0
        ranking[name] = {"precision_at_k": round(p, 3), "key_sentence_recall": round(r, 3),
                         "changed_drugs_mentioned": round(covered[name][0] / covered[name][1], 3) if covered[name][1] else 0.0}
    return {"k": k, "ranking_only": ranking,
            "end_to_end_key_recall": round(sys_hits_all / all_key, 3) if all_key else 0.0,
            "compression": round(words_sum[0] / words_sum[1], 3) if words_sum[1] else 0.0,
            "summary_sentences_not_found_in_gold": unmatched,
            "note": "random baseline uses a fixed seed; compression = summary words / hospital course words"}


# ----------------------------------------------------------------------------------------
# METRIC 4: Follow-up, pending results, warning signs
# ----------------------------------------------------------------------------------------
def other_metrics(results):
    """
    * Follow-up and warning signs are sets of sentences, scored with precision / recall.
    * Pending results is a yes/no question per note ("is anything still pending?"). A missed
      pending result is the costly mistake, because nobody then checks it, so recall matters most.
    """
    fu = [0, 0, 0]; wn = [0, 0, 0]; pend = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for item, out in results:
        g = item["gold"]
        for gold_list, pred_list, c in ((g["follow_up"], out["follow_up"], fu), (g["warnings"], out["warnings"], wn)):
            gs, ps = {s.strip() for s in gold_list}, {s.strip() for s in pred_list}
            c[0] += len(gs & ps); c[1] += len(ps - gs); c[2] += len(gs - ps)
        pred_has = bool(out["pending"].strip())
        key = ("tp" if pred_has else "fn") if g["has_pending"] else ("fp" if pred_has else "tn")
        pend[key] += 1
    p = prf(pend["tp"], pend["fp"], pend["fn"]); n = sum(pend.values())
    p["accuracy"] = round((pend["tp"] + pend["tn"]) / n, 3); p["tn"] = pend["tn"]
    return {"follow_up": prf(*fu), "warning_signs": prf(*wn), "pending_results": p}


# ----------------------------------------------------------------------------------------
# METRIC 5: Speed
# ----------------------------------------------------------------------------------------
def speed_metrics(data):
    """A tool that takes a minute per note will not be used. We time the whole pipeline per note."""
    analyze(data[0]["text"])                                    # warm-up (loads the NLP tools once)
    times = []
    for item in data:
        t = time.perf_counter(); analyze(item["text"]); times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return {"median_ms": round(statistics.median(times), 1), "p95_ms": round(times[int(0.95 * len(times)) - 1], 1),
            "notes_timed": len(times)}


# ----------------------------------------------------------------------------------------
# Put it all together
# ----------------------------------------------------------------------------------------
def evaluate(k=4, data=None, with_speed=True):
    data = data or load_eval()
    results = [(item, analyze(item["text"], n_sentences=k)) for item in data]
    report = {
        "n_notes": len(data),
        "n_medication_lines": sum(len(i["gold"]["medications"]) for i in data),
        "sections": section_metrics(results),
        "medications": medication_metrics(results),
        "summary": summary_metrics(data, k, results),
        "other": other_metrics(results),
    }
    if with_speed:
        report["speed"] = speed_metrics(data)
    return report


def print_report(m):
    p = lambda d: f"P {d['precision']:.2f}  R {d['recall']:.2f}  F1 {d['f1']:.2f}   (tp {d['tp']}, fp {d['fp']}, fn {d['fn']})"
    print(f"\nEVALUATION ON {m['n_notes']} SYNTHETIC NOTES ({m['n_medication_lines']} medication lines)")
    print("-" * 72)
    print("Sections found:            ", p(m["sections"]["overall"]))
    print("Medications (name only):   ", p(m["medications"]["name_only"]))
    print("Medications (name+status): ", p(m["medications"]["name_and_status"]))
    print("CHANGED drugs only:        ", p(m["medications"]["changed_drugs"]), " <- safety number")
    print(f"Status right when found:    {m['medications']['status_accuracy_when_found']:.2f}"
          f"   Dose right when found: {m['medications']['dose_accuracy_when_found']:.2f}")
    print("\nRecall by medication line format (where does the regex struggle?)")
    for k, v in m["medications"]["recall_by_line_format"].items():
        print(f"  {k:<14} {v:.2f}   ({m['medications']['line_counts_by_format'][k]} lines)")
    print("\nSummary vs baselines (notes where the course was found):")
    for name, d in m["summary"]["ranking_only"].items():
        print(f"  {name:<10} precision@k {d['precision_at_k']:.2f}   key-sentence recall {d['key_sentence_recall']:.2f}"
              f"   changed drugs mentioned {d['changed_drugs_mentioned']:.2f}")
    print(f"  end-to-end key-sentence recall (all notes): {m['summary']['end_to_end_key_recall']:.2f}")
    print("\nFollow-up:         ", p(m["other"]["follow_up"]))
    print("Warning signs:     ", p(m["other"]["warning_signs"]))
    print("Pending results:   ", p(m["other"]["pending_results"]))
    if "speed" in m:
        print(f"\nSpeed: median {m['speed']['median_ms']} ms per note, 95th percentile {m['speed']['p95_ms']} ms")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure the discharge assistant on the synthetic test set")
    ap.add_argument("--k", type=int, default=4, help="summary length in sentences")
    args = ap.parse_args()
    report = evaluate(k=args.k)
    print_report(report)
    Path(REPORT_FILE).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nSaved {REPORT_FILE}")

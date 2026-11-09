"""Calibration, coverage, and decision-error rates for the margin-mass screen.

The paper reports leave-one-corpus-out MAE.  A deployment screen also needs
calibration (does a predicted 0.2 mean 0.2?), interval coverage, and the cost of
the two decision errors.  This script reads ``out/paper_stats/loco_margin.json``
and reports, for the raw corpora and the held-out deployment suites:

* calibration slope/intercept and R^2 of actual q on predicted q;
* leave-one-corpus-out residual standard deviation and 95% interval coverage;
* the 2x2 decision table at q*: true positive (pred and actual above),
  false negative (pred at/below, actual above), false positive (pred above,
  actual at/below), true negative;
* small-prefix simulation coverage (does the 95% draw interval contain q?).

Reads existing results only.
"""
from __future__ import annotations

import json
import os

import numpy as np


def fit_calibration(pred, actual):
    A = np.vstack([np.ones_like(pred), pred]).T
    coef = np.linalg.lstsq(A, actual, rcond=None)[0]
    fitted = A @ coef
    ss_res = float(((actual - fitted) ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    # Pearson on predicted vs actual
    r = float(np.corrcoef(pred, actual)[0, 1]) if len(pred) > 2 else None
    return {"slope": float(coef[1]), "intercept": float(coef[0]),
            "r2": r2, "pearson": r}


def decision_table(rows):
    tp = sum(1 for r in rows if r["pred_above"] and r["actual_above"])
    fn = sum(1 for r in rows if (not r["pred_above"]) and r["actual_above"])
    fp = sum(1 for r in rows if r["pred_above"] and (not r["actual_above"]))
    tn = sum(1 for r in rows if (not r["pred_above"]) and (not r["actual_above"]))
    n = max(tp + fn + fp + tn, 1)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "false_negative_rate": fn / max(tp + fn, 1),
            "false_positive_rate": fp / max(fp + tn, 1),
            "accuracy": (tp + tn) / n}


def main():
    d = json.load(open("out/paper_stats/loco_margin.json"))
    out = {"widths": {}, "raw_pooled": {}, "external_pooled": {},
           "small_prefix": {}}

    raw_rows = []
    for w, r in d["widths"].items():
        rows = r["folds"]
        pred = np.array([x["q_pred"] for x in rows])
        actual = np.array([x["q"] for x in rows])
        resid = actual - pred
        sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else None
        cover = (float(np.mean(np.abs(resid) <= 1.96 * sigma))
                 if sigma is not None else None)
        out["widths"][w] = {
            "n": len(rows), "mae": r["mae"], "bias": r["bias"],
            "calibration": fit_calibration(pred, actual),
            "residual_sd": sigma,
            "interval_95_coverage": cover,
            "decision": decision_table(rows),
            "mae_2feat": r["mae_2feat"],
            "calibration_2feat": fit_calibration(
                np.array([x["q_pred_2feat"] for x in rows]), actual),
            "decision_2feat": decision_table(
                [{"pred_above": x["pred_above_2feat"],
                  "actual_above": x["actual_above"]} for x in rows]),
        }
        for x in rows:
            raw_rows.append(x)

    pred = np.array([x["q_pred"] for x in raw_rows])
    actual = np.array([x["q"] for x in raw_rows])
    resid = actual - pred
    sigma = float(np.std(resid, ddof=1))
    out["raw_pooled"] = {
        "n": len(raw_rows),
        "mae": float(np.mean(np.abs(resid))),
        "bias": float(np.mean(resid)),
        "calibration": fit_calibration(pred, actual),
        "residual_sd": sigma,
        "interval_95_coverage": float(np.mean(np.abs(resid) <= 1.96 * sigma)),
        "decision": decision_table(raw_rows),
    }

    ext_rows = []
    for w, rows in d["external"].items():
        for x in rows:
            ext_rows.append(x)
    if ext_rows:
        pred = np.array([x["q_pred"] for x in ext_rows])
        actual = np.array([x["q"] for x in ext_rows])
        resid = actual - pred
        out["external_pooled"] = {
            "n": len(ext_rows),
            "mae": float(np.mean(np.abs(resid))),
            "bias": float(np.mean(resid)),
            "calibration": fit_calibration(pred, actual),
            "decision": decision_table(ext_rows),
            "rows": [{"suite": x["suite"], "width": x["width"],
                      "q": x["q"], "q_pred": x["q_pred"],
                      "error": x["error"], "q_star": x["q_star"],
                      "pred_above": x["pred_above"],
                      "actual_above": x["actual_above"]} for x in ext_rows],
        }

    for w, rows in d["deploy"].items():
        cover = []
        for x in rows:
            lo, hi = x["q_pred_ci95"]
            cover.append({"suite": x["suite"],
                          "contains": bool(lo <= x["q_true"] <= hi),
                          "q_true": x["q_true"], "lo": lo, "hi": hi})
        out["small_prefix"][w] = {
            "n": len(cover),
            "coverage": float(np.mean([c["contains"] for c in cover])) if cover else None,
            "rows": cover,
        }

    os.makedirs("out/paper_stats", exist_ok=True)
    with open("out/paper_stats/loco_calibration.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in
                      ["raw_pooled", "external_pooled"]}, indent=1))
    print("per-width decision tables:")
    for w, r in out["widths"].items():
        print(w, r["calibration"], r["decision"])


if __name__ == "__main__":
    main()

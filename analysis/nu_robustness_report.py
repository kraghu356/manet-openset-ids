"""Combine the two robustness campaigns into results/nu_robustness.csv.

Endpoint detection comes from the campaign run with the corrected generator
(own address excluded from nu). Honest-node false positives come from the
larger campaign run before the correction: honest nodes never list themselves,
so their nu is identical under both builds (checked seed by seed), and the
larger campaign gives a tighter estimate.

    python3 analysis/nu_robustness_report.py --fixed ~/runs_nu_fix --honest ~/runs_nu
"""
import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nu_robustness as R

ap = argparse.ArgumentParser()
ap.add_argument("--fixed", required=True)
ap.add_argument("--honest", required=True)
a = ap.parse_args()
f = R.summarise(a.fixed, list(range(1, 11))).set_index("condition")
h = R.summarise(a.honest, list(range(1, 11))).set_index("condition")
# Honest nodes: whichever campaign has more runs for the condition (the
# corrected campaign was later extended to eight seeds for some conditions).
h = h.reindex(f.index)
use_f = f.runs > h.runs.fillna(0)
for c in ["runs", "honest", "fpr_node", "fpr_win", "fpr_node_2w"]:
    h.loc[use_f, c] = f.loc[use_f, c]


def wilson(k, n, z=1.96):
    import numpy as np
    if not n:
        return (float("nan"), float("nan"))
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; w = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - w), min(1.0, c + w))


tl, tu, fl, fu = [], [], [], []
for c in f.index:
    a1 = wilson(round(f.tpr_node[c] * f.endpoints[c]), f.endpoints[c]); tl.append(a1[0]); tu.append(a1[1])
    a2 = wilson(round(h.fpr_node[c] * h.honest[c]), h.honest[c]); fl.append(a2[0]); fu.append(a2[1])
out = pd.DataFrame({
    "runs_endpoint": f.runs, "endpoints": f.endpoints,
    "tpr_node": f.tpr_node, "tpr_node_lo": tl, "tpr_node_hi": tu, "tpr_win": f.tpr_win,
    "t_detect_median": f.t_detect_median,
    "tpr_node_2w": f.tpr_node_2w, "t_detect_2w_median": f.t_detect_2w_median,
    "runs_honest": h.runs, "honest": h.honest,
    "fpr_node": h.fpr_node, "fpr_node_lo": fl, "fpr_node_hi": fu, "fpr_win": h.fpr_win, "fpr_node_2w": h.fpr_node_2w,
    "honest_source": ["corrected" if use_f[c] else "honest-campaign" for c in f.index],
}, index=f.index).reindex([c for c in R.CONDITIONS if c in f.index]).reset_index().rename(columns={"index": "condition"})
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "nu_robustness.csv")
out.to_csv(p, index=False)
print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

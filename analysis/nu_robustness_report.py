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
out = pd.DataFrame({
    "runs_endpoint": f.runs, "endpoints": f.endpoints,
    "tpr_node": f.tpr_node, "tpr_win": f.tpr_win, "t_detect_median": f.t_detect_median,
    "tpr_node_2w": f.tpr_node_2w, "t_detect_2w_median": f.t_detect_2w_median,
    "runs_honest": h.runs, "honest": h.honest,
    "fpr_node": h.fpr_node, "fpr_win": h.fpr_win, "fpr_node_2w": h.fpr_node_2w,
}).reindex([c for c in R.CONDITIONS if c in f.index]).reset_index().rename(columns={"index": "condition"})
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "nu_robustness.csv")
out.to_csv(p, index=False)
print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

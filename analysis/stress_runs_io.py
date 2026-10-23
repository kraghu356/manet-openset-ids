"""Pack the wormhole stress runs into one released file, or unpack it back into the
run directories that analysis/nu_robustness_report.py and
analysis/fixed_detector_eval.py read.

    python3 analysis/stress_runs_io.py pack   --fixed runs_nu_fix --honest runs_nu
    python3 analysis/stress_runs_io.py unpack --fixed runs_nu_fix --honest runs_nu

The file is results/stress_runs.csv.gz, with columns campaign (corrected or
honest), condition and seed added to the generator's per-node output.
"""
import argparse, glob, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "stress_runs.csv.gz")

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["pack", "unpack"])
ap.add_argument("--fixed", required=True)
ap.add_argument("--honest", required=True)
a = ap.parse_args()
roots = {"corrected": a.fixed, "honest": a.honest}
if a.mode == "pack":
    parts = []
    for camp, root in roots.items():
        for done in sorted(glob.glob(os.path.join(root, "*", "s*.done"))):
            d = done[:-5]
            f = pd.read_csv(os.path.join(d, "features.csv"))
            f.insert(0, "seed", int(os.path.basename(d)[1:]))
            f.insert(0, "condition", os.path.basename(os.path.dirname(d)))
            f.insert(0, "campaign", camp)
            parts.append(f)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print("rows", len(df), "runs", df.groupby(["campaign", "condition", "seed"]).ngroups)
else:
    df = pd.read_csv(OUT)
    for (camp, cond, seed), g in df.groupby(["campaign", "condition", "seed"]):
        d = os.path.join(roots[camp], cond, f"s{seed}")
        os.makedirs(d, exist_ok=True)
        g.drop(columns=["campaign", "condition", "seed"]).to_csv(os.path.join(d, "features.csv"), index=False)
        open(d + ".done", "w").close()
    print("unpacked", df.groupby(["campaign", "condition", "seed"]).ngroups, "runs")

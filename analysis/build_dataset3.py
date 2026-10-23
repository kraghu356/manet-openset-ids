"""Build results/dataset3.csv.gz: dataset2 with the wormhole scenario regenerated.

dataset2 was generated before the unheard-neighbour count excluded a node's own
address (see ns3/manet-gen.cc). Honest nodes and the attackers of the other
four scenarios never list themselves, so their rows are unchanged and are kept.
The 20 wormhole runs are regenerated with the corrected generator
(analysis/run_campaign.py --scenarios worm) and replace the old ones.

    python3 analysis/build_dataset3.py --worm-runs ~/runs_worm_fix
"""
import argparse, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")

ap = argparse.ArgumentParser()
ap.add_argument("--worm-runs", required=True)
a = ap.parse_args()

d2 = pd.read_csv(os.path.join(R, "dataset2.csv.gz"))
keep = d2[d2.scenario != "worm"]
parts = [keep]
for s in range(1, 21):
    f = pd.read_csv(os.path.join(a.worm_runs, f"worm_s{s}", "features.csv"))
    f["label"] = f.label  # generator writes "BHA" for every malicious node, as in dataset2
    f["run"] = f"worm_s{s}"; f["seed"] = s; f["scenario"] = "worm"
    f["cls"] = f.is_malicious.map({0: "normal", 1: "worm"})
    parts.append(f[d2.columns])
d3 = pd.concat(parts, ignore_index=True)
d3.to_csv(os.path.join(R, "dataset3.csv.gz"), index=False, compression="gzip")
w = d3[d3.scenario == "worm"]
print("rows", len(d3), "worm rows", len(w))
print("endpoint nu max per node:", w[w.is_malicious == 1].groupby(["seed", "node_id"]).nb_no_rssi.max().value_counts().to_dict())
print("honest nu max:", int(d3[d3.is_malicious == 0].nb_no_rssi.max()))
# every routing neighbour of an honest node is learned from a decoded frame
assert d3[d3.is_malicious == 0].nb_no_rssi.max() == 0, "an honest node has nu > 0"
assert d3[(d3.is_malicious == 1) & (d3.scenario != "worm")].nb_no_rssi.max() == 0

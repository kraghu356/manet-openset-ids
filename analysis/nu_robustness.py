"""Robustness of the unheard-neighbour invariant nu under channel and network stress.

Runs the ns-3 generator for a set of stress conditions (wormhole scenario),
then reports, per condition, over all seeds:
  tpr_node   fraction of wormhole endpoints with nu > 0 at any time
  fpr_node   fraction of honest nodes with nu > 0 at any time
  tpr_win    fraction of endpoint windows (after first activity) with nu > 0
  fpr_win    fraction of honest windows with nu > 0
  t_detect   median time of first nu > 0 at an endpoint (s)
  pdr        mean packet delivery ratio is not recorded here; see manet-gen output

Usage:
  python3 analysis/nu_robustness.py --ns3 ~/ns-3.48 --out runs_nu --seeds 10 [--jobs 2]
  python3 analysis/nu_robustness.py --out runs_nu --report-only
"""
import argparse, json, os, subprocess, itertools
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

BASE = ["--nNodes=100", "--areaX=1500", "--areaY=1500", "--simTime=300",
        "--nFlows=10", "--pktRate=4", "--txPower=23", "--plExp=2.5",
        "--standard=80211b", "--dataMode=DsssRate11Mbps",
        "--nMalicious=2", "--wormhole=1", "--dropProb=0", "--forge=0"]

CONDITIONS = {
    "baseline":          [],
    "shadow_2dB":        ["--shadowSigma=2"],
    "shadow_4dB":        ["--shadowSigma=4"],
    "shadow_6dB":        ["--shadowSigma=6"],
    "loss_5pct":         ["--rxLoss=0.05"],
    "loss_10pct":        ["--rxLoss=0.10"],
    "speed_5_15":        ["--minSpeed=5", "--maxSpeed=15"],
    "speed_10_20":       ["--minSpeed=10", "--maxSpeed=20"],
    "dense_1000m":       ["--areaX=1000", "--areaY=1000"],
    "large_200n":        ["--nNodes=200", "--areaX=2100", "--areaY=2100", "--nFlows=20"],
    "hello_on":          ["--hello=1"],
    "relay_wormhole":    ["--wormTransparent=1"],
}


def run_one(ns3, out, cond, seed):
    d = os.path.join(out, cond, f"s{seed}")
    f = os.path.join(d, "features.csv")
    if os.path.exists(f) and os.path.getsize(f) > 0 and os.path.exists(d + ".done"):
        return d
    os.makedirs(d, exist_ok=True)
    exe = os.path.join(ns3, "build", "scratch", "ns3.48-manet-gen-optimized")
    args = [exe] + BASE + CONDITIONS[cond] + [f"--seed={seed}", f"--outDir={d}"]
    with open(os.path.join(d, "stdout.txt"), "w") as fo:
        subprocess.run(args, stdout=fo, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, LD_LIBRARY_PATH=os.path.join(ns3, "build", "lib")))
    open(d + ".done", "w").close()
    return d


def summarise(out, seeds):
    rows = []
    for cond in CONDITIONS:
        tn = fn = tw = fw = 0; Tn = Fn = Tw = Fw = 0; tdet = []
        n_runs = 0
        for s in seeds:
            f = os.path.join(out, cond, f"s{s}", "features.csv")
            if not os.path.exists(f):
                continue
            n_runs += 1
            df = pd.read_csv(f)
            for nid, g in df.groupby("node_id"):
                mal = g.is_malicious.iloc[0] == 1
                hit = g.nb_no_rssi > 0
                # windows after the node first took part in routing
                act = g[(g.rreq_recv.cumsum() > 0)]
                hw = (act.nb_no_rssi > 0)
                if mal:
                    Tn += 1; tn += int(hit.any())
                    Tw += len(act); tw += int(hw.sum())
                    if hit.any():
                        tdet.append(float(g.time_s[hit].iloc[0]))
                else:
                    Fn += 1; fn += int(hit.any())
                    Fw += len(act); fw += int(hw.sum())
        if n_runs == 0:
            continue
        rows.append(dict(condition=cond, runs=n_runs,
                         endpoints=Tn, honest=Fn,
                         tpr_node=tn / Tn if Tn else np.nan,
                         fpr_node=fn / Fn if Fn else np.nan,
                         tpr_win=tw / Tw if Tw else np.nan,
                         fpr_win=fw / Fw if Fw else np.nan,
                         t_detect_median=float(np.median(tdet)) if tdet else np.nan))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns3", default=os.path.expanduser("~/ns-3.48"))
    ap.add_argument("--out", default="runs_nu")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    seeds = list(range(1, a.seeds + 1))
    conds = a.conditions.split(",")
    if not a.report_only:
        jobs = list(itertools.product(conds, seeds))
        with ThreadPoolExecutor(a.jobs) as ex:
            for i, d in enumerate(ex.map(lambda cs: run_one(a.ns3, a.out, *cs), jobs), 1):
                print(f"[{i}/{len(jobs)}] {d}", flush=True)
    res = summarise(a.out, seeds)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    os.makedirs("results", exist_ok=True)
    res.to_csv("results/nu_robustness.csv", index=False)
    print("written results/nu_robustness.csv")

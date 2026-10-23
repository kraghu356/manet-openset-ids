"""Independent test campaign, generated after the method was fixed.

None of these runs were seen while choosing scores, thresholds or the k-NN + nu
combination. Two conditions:

  same_env   the paper's settings, fresh seeds 101-104 (6 scenarios x 4 seeds)
  shift_env  2 dB log-normal shadowing and node speeds 2-8 m/s, seeds 201-203
             (6 scenarios x 3 seeds)
  shift_cal  six further normal runs (seeds 204-209) in the shifted environment,
             used only to recalibrate thresholds (analysis/recalibration.py)

Runs are resumable: a run with a .done marker is skipped.

  python3 analysis/independent_campaign.py --ns3 ~/ns-3.48 --out runs_indep [--only same_env]
"""
import argparse, itertools, os, subprocess
from concurrent.futures import ThreadPoolExecutor

from run_campaign import BASE, SCEN

CONDITIONS = {
    "same_env": ([], list(range(101, 105))),
    "shift_env": (["--shadowSigma=2", "--minSpeed=2", "--maxSpeed=8"], list(range(201, 204))),
    "shift_cal": (["--shadowSigma=2", "--minSpeed=2", "--maxSpeed=8"], [204, 205, 206, 207, 208, 209]),
}
SCEN_OF = {"shift_cal": ["normal"]}


def run(ns3, out, cond, sc, seed):
    d = os.path.join(out, cond, f"{sc}_s{seed}")
    if os.path.exists(d + ".done"):
        return d
    os.makedirs(d, exist_ok=True)
    exe = os.path.join(ns3, "build", "scratch", "ns3.48-manet-gen-optimized")
    args = ["setarch", os.uname().machine, "-R", exe] + BASE + SCEN[sc] + CONDITIONS[cond][0] \
        + [f"--seed={seed}", f"--outDir={d}"]
    with open(os.path.join(d, "stdout.txt"), "w") as fo:
        subprocess.run(args, stdout=fo, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, LD_LIBRARY_PATH=os.path.join(ns3, "build", "lib")))
    open(d + ".done", "w").close()
    print("done", d, flush=True)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns3", default=os.path.expanduser("~/ns-3.48"))
    ap.add_argument("--out", default="runs_indep")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", default=",".join(CONDITIONS))
    a = ap.parse_args()
    jobs = [(c, sc, s) for c in a.only.split(",") for sc, s in itertools.product(SCEN_OF.get(c, SCEN), CONDITIONS[c][1])]
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda j: run(a.ns3, a.out, *j), jobs))

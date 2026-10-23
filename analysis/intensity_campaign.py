"""Attack-intensity campaign: the same scenario settings as analysis/run_campaign.py
with one attack parameter changed, new seeds 301-304. Used only for testing the
frozen detectors (analysis/intensity_eval.py); never for training or selection.

  gha_p025   grey hole, drop probability 0.25 (paper: 0.5)
  gha_p075   grey hole, drop probability 0.75
  flood_r1   flooding, 1 bogus route request per second per attacker (paper: 5)
  flood_r2   flooding, 2 per second
  bha_n2     black hole, 2 attackers (paper: 5)
  bha_n10    black hole, 10 attackers

  python3 analysis/intensity_campaign.py --ns3 ~/ns-3.48 --out runs_intensity
"""
import argparse, itertools, os, subprocess
from concurrent.futures import ThreadPoolExecutor

from run_campaign import BASE

COND = {
    "gha_p025": ["--nMalicious=5", "--forge=0", "--dropProb=0.25"],
    "gha_p075": ["--nMalicious=5", "--forge=0", "--dropProb=0.75"],
    "flood_r1": ["--nMalicious=5", "--forge=0", "--dropProb=0.0", "--floodRate=1"],
    "flood_r2": ["--nMalicious=5", "--forge=0", "--dropProb=0.0", "--floodRate=2"],
    "bha_n2":   ["--nMalicious=2", "--forge=1", "--dropProb=1.0"],
    "bha_n10":  ["--nMalicious=10", "--forge=1", "--dropProb=1.0"],
}
SEEDS = [301, 302, 303, 304]


def run(ns3, out, cond, seed):
    d = os.path.join(out, f"{cond}_s{seed}")
    if os.path.exists(d + ".done"):
        return d
    os.makedirs(d, exist_ok=True)
    exe = os.path.join(ns3, "build", "scratch", "ns3.48-manet-gen-optimized")
    with open(os.path.join(d, "stdout.txt"), "w") as fo:
        subprocess.run(["setarch", os.uname().machine, "-R", exe] + BASE + COND[cond] + [f"--seed={seed}", f"--outDir={d}"],
                       stdout=fo, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, LD_LIBRARY_PATH=os.path.join(ns3, "build", "lib")))
    open(d + ".done", "w").close()
    print("done", d, flush=True)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns3", default=os.path.expanduser("~/ns-3.48"))
    ap.add_argument("--out", default="runs_intensity")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda j: run(a.ns3, a.out, *j), itertools.product(COND, SEEDS)))

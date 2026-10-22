"""Regenerate the 120-run campaign (6 scenarios x 20 seeds).

Scenario flags (malicious nodes are always the highest node ids):
  normal  no attacker
  bha     5 attackers, forge RREPs, drop every forwarded packet
  gha     5 attackers, honest routing, drop forwarded packets with p = 0.5
  sink    5 attackers, forge RREPs, forward everything
  flood   5 attackers, originate 5 bogus RREQs per second, forward everything
  worm    2 attackers joined by an out-of-band tunnel, forward everything

Runs execute with address-space randomisation disabled (setarch -R) so a
seed reproduces exactly on a given build; see analysis/nu_robustness.py.

  python3 analysis/run_campaign.py --ns3 ~/ns-3.48 --out runs --seeds 20
  python3 analysis/run_campaign.py --ns3 ~/ns-3.48 --out runs --seeds 3 --pdr
"""
import argparse, itertools, os, re, subprocess
from concurrent.futures import ThreadPoolExecutor

BASE = ["--nNodes=100", "--areaX=1500", "--areaY=1500", "--simTime=300",
        "--nFlows=10", "--pktRate=4", "--txPower=23", "--plExp=2.5",
        "--standard=80211b", "--dataMode=DsssRate11Mbps"]
SCEN = {
    "normal": ["--nMalicious=0"],
    "bha":    ["--nMalicious=5", "--forge=1", "--dropProb=1.0"],
    "gha":    ["--nMalicious=5", "--forge=0", "--dropProb=0.5"],
    "sink":   ["--nMalicious=5", "--forge=1", "--dropProb=0.0"],
    "flood":  ["--nMalicious=5", "--forge=0", "--dropProb=0.0", "--floodRate=5"],
    "worm":   ["--nMalicious=2", "--wormhole=1", "--forge=0", "--dropProb=0.0"],
}


def run(ns3, out, sc, seed):
    d = os.path.join(out, f"{sc}_s{seed}")
    os.makedirs(d, exist_ok=True)
    exe = os.path.join(ns3, "build", "scratch", "ns3.48-manet-gen-optimized")
    r = subprocess.run(["setarch", os.uname().machine, "-R", exe] + BASE + SCEN[sc]
                       + [f"--seed={seed}", f"--outDir={d}"], capture_output=True, text=True,
                       env=dict(os.environ, LD_LIBRARY_PATH=os.path.join(ns3, "build", "lib")), check=True)
    open(os.path.join(d, "stdout.txt"), "w").write(r.stdout + r.stderr)
    m = re.search(r"PDR=([\d.]+).*hops=([\d.]+)", r.stdout)
    return sc, seed, float(m.group(1)), float(m.group(2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns3", default=os.path.expanduser("~/ns-3.48"))
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--pdr", action="store_true", help="print delivery ratio and hops per scenario")
    a = ap.parse_args()
    jobs = list(itertools.product(SCEN, range(1, a.seeds + 1)))
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(lambda j: run(a.ns3, a.out, *j), jobs))
    if a.pdr:
        import statistics as st
        for sc in SCEN:
            p = [r[2] for r in res if r[0] == sc]; h = [r[3] for r in res if r[0] == sc]
            print(f"{sc:7s} PDR {st.mean(p):.3f} +- {st.pstdev(p):.3f}  hops {st.mean(h):.2f}")

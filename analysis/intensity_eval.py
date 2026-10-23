"""Frozen detectors on the attack-intensity campaign (analysis/intensity_campaign.py).

For each condition the detector that never saw that attack family (fitted on
dataset3 seeds 1-16 with the family held out, thresholds from validation-normal
seeds 17-20) scores every node of the four intensity runs; positives are the
attackers, negatives the honest nodes of the same runs. Writes
results/intensity.json.
"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset_eval as E
import fixed_detector_eval as F

FAMILY = {"gha_p025": "gha", "gha_p075": "gha", "flood_r1": "flood", "flood_r2": "flood", "bha_n2": "bha", "bha_n10": "bha"}


def main(root=os.path.expanduser("~/runs_intensity")):
    a = E.load()
    ae = F.fit_ae(a, list(range(1, 17)))
    dets = {h: F.Detector(a, list(range(1, 17)), list(range(17, 21)), h, ae) for h in sorted(set(FAMILY.values()))}
    res = {}
    for cond, h in FAMILY.items():
        paths = sorted(p[:-5] for p in glob.glob(os.path.join(root, f"{cond}_s*.done")))
        if not paths:
            continue
        df = F.read_runs(paths, lambda n: h, lambda n: int(n.split("_s")[1]), lambda n: h)
        S, c, runs = dets[h].score(df)
        res[cond] = F.metrics(dets[h], S, c == h, c == "normal", runs)
        res[cond]["runs"] = len(paths)
        print(cond, len(paths), {m: (round(res[cond][m]["auroc"], 3), round(res[cond][m]["p95"]["tpr"], 2), round(res[cond][m]["p95"]["fpr"], 3))
                                 for m in F.METHODS}, flush=True)
    json.dump(res, open(os.path.join(E.HERE, "..", "results", "intensity.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])

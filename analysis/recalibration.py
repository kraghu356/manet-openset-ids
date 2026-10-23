"""Recalibration in a shifted environment using normal traffic only.

The frozen detectors of analysis/fixed_detector_eval.py (fitted on dataset3
seeds 1-16, one per held-out attack) are kept unchanged: same scaler, same
k-NN, forest and autoencoder. Only the reference distributions are replaced:
the rank references of Eqs. 9-10 and the threshold percentiles are recomputed
from honest nodes of two normal runs in the shifted environment
(runs_indep/shift_cal, seeds 204-205), which contain no attack and are not
used for testing. Test data are the 18 shifted-environment runs
(runs_indep/shift_env, seeds 201-203). Both the frozen and the recalibrated
detectors are reported.

Writes results/recalibration.json.
"""
import copy, glob, json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset_eval as E
import fixed_detector_eval as F

PCTS = [95, 99]


def evaluate(det, S, c, h, runs):
    pos, neg = c == h, c == "normal"
    out = {}
    for m in F.METHODS:
        p, n = S[m][pos], S[m][neg]
        d = {"auroc": float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]))}
        for q in PCTS:
            t = np.percentile(det.val[m], q)
            tp, fp = int((p > t).sum()), int((n > t).sum())
            d[f"p{q}"] = {"tpr": tp / len(p), "fpr": fp / len(n), "fpr_ci": F.wilson(fp, len(n)),
                          "false_alerts_per_run": fp / len(np.unique(runs[pos | neg]))}
        out[m] = d
    return out


def main(indep=os.path.expanduser("~/runs_indep")):
    a = E.load()
    ae = F.fit_ae(a, list(range(1, 17)))
    dets = {h: F.Detector(a, list(range(1, 17)), list(range(17, 21)), h, ae) for h in E.ATTACKS}
    name = lambda n: n.split("_s")[0]; seed = lambda n: int(n.split("_s")[1])
    cal = F.read_runs(sorted(p[:-5] for p in glob.glob(os.path.join(indep, "shift_cal", "*.done"))), name, seed, name)
    test = F.read_runs(sorted(p[:-5] for p in glob.glob(os.path.join(indep, "shift_env", "*.done"))), name, seed, name)
    Xc, Mc, cc, nuc, _ = E.node_table(cal, F.FEATS)
    res = {"calibration_honest_nodes": int((cc == "normal").sum()), "test_runs": int(test.run.nunique()),
           "frozen": {}, "recalibrated": {}}
    for h in E.ATTACKS:
        det = dets[h]
        S, c, runs = det.score(test)
        res["frozen"][h] = evaluate(det, S, c, h, runs)
        rc = copy.copy(det)
        raw = det.raw(Xc, Mc); ok = cc == "normal"
        rc.ref = {m: v[ok] for m, v in raw.items()}
        rc.val = {m: v[ok] for m, v in rc.combine(raw, nuc).items()}
        X, M, c2, nu, _ = E.node_table(test, F.FEATS)
        S2 = rc.combine(rc.raw(X, M), nu)
        res["recalibrated"][h] = evaluate(rc, S2, c2, h, runs)
        print(h, {k: {m: (round(res[k][h][m]["auroc"], 3), round(res[k][h][m]["p99"]["tpr"], 2), round(res[k][h][m]["p99"]["fpr"], 3))
                      for m in F.METHODS} for k in ["frozen", "recalibrated"]}, flush=True)
    for k in ["frozen", "recalibrated"]:
        res[k]["summary"] = {m: {f"p{q}": {"mean_tpr": float(np.mean([res[k][h][m][f"p{q}"]["tpr"] for h in E.ATTACKS])),
                                          "max_fpr": float(max(res[k][h][m][f"p{q}"]["fpr"] for h in E.ATTACKS)),
                                          "mean_false_alerts_per_run": float(np.mean([res[k][h][m][f"p{q}"]["false_alerts_per_run"] for h in E.ATTACKS]))}
                                 for q in PCTS} | {"mean_auroc": float(np.mean([res[k][h][m]["auroc"] for h in E.ATTACKS]))}
                             for m in F.METHODS}
    p = os.path.join(E.HERE, "..", "results", "recalibration.json")
    json.dump(res, open(p, "w"), indent=1)
    print(json.dumps({k: res[k]["summary"] for k in ["frozen", "recalibrated"]}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Evaluate the finished detectors, frozen, on data that played no part in choosing them.

Detectors (k-NN distance, k-NN + nu of Eq. 10, hybrid of Eq. 9) are fitted on
dataset3 seeds 1-16 and thresholded on seeds 17-20 exactly as in
analysis/openset_eval.py, once per held-out attack. Nothing is refitted on
the data below.

  independent  runs_indep/same_env and runs_indep/shift_env
               (analysis/independent_campaign.py): every held-out attack
  stress       the wormhole stress runs of analysis/nu_robustness.py, scored
               by the detector that never saw a wormhole. Endpoint nodes come
               from the corrected generator (runs_nu_fix), honest nodes from
               the larger campaign (runs_nu); honest rows are identical
               between the two generator versions.
  thresholds   the four dataset3 folds, held-out flooding and the mean over
               attacks: detection, false-positive rate, precision and false
               alerts per run at the 90th, 95th, 99th and 99.5th percentile
               of validation-normal scores.

Writes results/fixed_detector.json.

  python3 analysis/fixed_detector_eval.py --indep ~/runs_indep --stress-fix ~/runs_nu_fix --stress-honest ~/runs_nu
"""
import argparse, glob, json, os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset_eval as E

FEATS = E.F_OBS
PCTS = [90, 95, 99, 99.5]
METHODS = ["knn_dist", "knn_nu", "hybrid"]


def prep(a):
    a = a.sort_values(["run", "node_id", "time_s"]).copy()
    for c in ["rreq_sent", "rreq_recv", "no_route", "fwd_seen"]:
        a[c] = np.log1p(a[c].clip(lower=0))
    g = a.groupby(["run", "node_id"])
    for c in E.EWMA:
        a[c + "_ewma"] = g[c].transform(lambda x: x.ewm(alpha=0.3).mean())
    return a


def read_runs(paths, scen_of, seed_of, cls_of):
    parts = []
    for p in paths:
        f = pd.read_csv(os.path.join(p, "features.csv"))
        name = os.path.basename(p)
        f["run"] = p; f["scenario"] = scen_of(name); f["seed"] = seed_of(name)
        f["cls"] = np.where(f.is_malicious == 1, cls_of(name), "normal")
        parts.append(f)
    return prep(pd.concat(parts, ignore_index=True))


class Detector:
    def __init__(s, a, trs, vas, h, ae):
        s.h = h
        tr = a[a.seed.isin(trs) & (a.scenario != h)]
        va = a[a.seed.isin(vas) & (a.scenario != h)]
        Xtr, Mtr, ctr, _, _ = E.node_table(tr, FEATS)
        Xva, Mva, cva, nuva, _ = E.node_table(va, FEATS)
        known = sorted([c for c in E.ATTACKS if c != h] + ["normal"]); ki = {c: i for i, c in enumerate(known)}
        s.sc = StandardScaler().fit(Xtr.reshape(-1, len(FEATS)))
        Ftr = E.flat(s.f(Xtr), Mtr)
        s.kn = KNeighborsClassifier(5).fit(Ftr, np.array([ki[c] for c in ctr]))
        s.rf = RandomForestClassifier(300, random_state=0, class_weight="balanced", n_jobs=-1).fit(
            Ftr, np.array([ki[c] for c in ctr]))
        s.ae, s.sca = ae
        vneg = cva == "normal"
        raw = s.raw(Xva, Mva)
        s.ref = {m: raw[m][vneg] for m in raw}
        s.val = s.combine(raw, nuva)
        s.val = {m: v[vneg] for m, v in s.val.items()}

    def f(s, X):
        return s.sc.transform(X.reshape(-1, len(FEATS))).reshape(X.shape).astype(np.float32)

    def raw(s, X, M):
        F = E.flat(s.f(X), M)
        Xa = s.sca.transform(X.reshape(-1, len(FEATS))).reshape(X.shape).astype(np.float32)
        return {"knn_dist": s.kn.kneighbors(F)[0].mean(1), "rf_vote": 1 - E.vote(s.rf, F).max(1),
                "autoencoder": E.recon(s.ae, Xa, M)}

    def combine(s, raw, nu):
        r = lambda m: E.rank_against(s.ref[m], raw[m]) if hasattr(s, "ref") else None
        ov = (nu > 0) * (1.0 + nu)
        return {"knn_dist": raw["knn_dist"],
                "knn_nu": np.maximum(r("knn_dist"), ov),
                "hybrid": np.maximum(np.maximum(r("rf_vote"), r("autoencoder")), ov)}

    def score(s, df):
        X, M, c, nu, _ = E.node_table(df, FEATS)
        runs = np.array([r for (r, _), q in df.groupby(["run", "node_id"], sort=False) if q.fwd_seen.sum() > 0])
        return s.combine(s.raw(X, M), nu), c, runs


def fit_ae(a, trs):
    Xn, Mn, _, _, _ = E.node_table(a[a.seed.isin(trs) & (a.cls == "normal") & (a.scenario == "normal")], FEATS)
    sca = StandardScaler().fit(Xn.reshape(-1, len(FEATS)))
    return E.train_ae(sca.transform(Xn.reshape(-1, len(FEATS))).reshape(Xn.shape).astype(np.float32), Mn, 40), sca


def metrics(det, S, pos, neg, runs):
    out = {}
    for m in METHODS:
        s = S[m]; p, n = s[pos], s[neg]
        y = np.r_[np.ones(len(p)), np.zeros(len(n))]; z = np.r_[p, n]
        d = {"n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
        d["auroc"] = float(roc_auc_score(y, z)) if len(p) and len(n) and len(np.unique(z)) > 1 else (0.5 if len(p) and len(n) else None)
        d["ap"] = float(average_precision_score(y, z)) if len(p) and len(n) else None
        nruns = len(np.unique(runs[pos | neg]))
        for q in PCTS:
            t = np.percentile(det.val[m], q)
            tp, fp = int((p > t).sum()), int((n > t).sum())
            d[f"p{q}"] = {"tpr": tp / max(len(p), 1), "fpr": fp / max(len(n), 1),
                          "precision": tp / (tp + fp) if tp + fp else None,
                          "false_alerts_per_run": fp / max(nruns, 1)}
        out[m] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indep", default=os.path.expanduser("~/runs_indep"))
    ap.add_argument("--stress-fix", default=os.path.expanduser("~/runs_nu_fix"))
    ap.add_argument("--stress-honest", default=os.path.expanduser("~/runs_nu"))
    ap.add_argument("--skip-thresholds", action="store_true")
    a_ = ap.parse_args()
    a = E.load()
    out_path = os.path.join(E.HERE, "..", "results", "fixed_detector.json")
    # sections that are not recomputed in this call are kept from an earlier run
    res = json.load(open(out_path)) if os.path.exists(out_path) else {"independent": {}, "stress": {}, "thresholds": {}}
    res["protocol"] = "detectors fitted on dataset3 seeds 1-16, thresholds from validation-normal seeds 17-20; frozen"
    ae = fit_ae(a, list(range(1, 17)))
    dets = {h: Detector(a, list(range(1, 17)), list(range(17, 21)), h, ae) for h in E.ATTACKS}
    print("detectors fitted", flush=True)

    # ---- independent campaign ----
    for cond in ["same_env", "shift_env"]:
        paths = sorted(p[:-5] for p in glob.glob(os.path.join(a_.indep, cond, "*.done")))
        if not paths:
            continue
        df = read_runs(paths, lambda n: n.split("_s")[0], lambda n: int(n.split("_s")[1]), lambda n: n.split("_s")[0])
        res["independent"][cond] = {"runs": len(paths)}
        for h in E.ATTACKS:
            S, c, runs = dets[h].score(df)
            res["independent"][cond][h] = metrics(dets[h], S, c == h, c == "normal", runs)
            print(cond, h, {m: round(res["independent"][cond][h][m]["auroc"] or 0, 3) for m in METHODS}, flush=True)

    # ---- full-detector stress test (wormhole held out) ----
    import nu_robustness as NR
    det = dets["worm"]
    for cond in NR.CONDITIONS:
        fx = sorted(p[:-5] for p in glob.glob(os.path.join(a_.stress_fix, cond, "s*.done")))
        hn = sorted(p[:-5] for p in glob.glob(os.path.join(a_.stress_honest, cond, "s*.done")))
        if not fx or not hn:
            continue
        lab = lambda n: "worm"
        dfx = read_runs(fx, lambda n: "worm", lambda n: 1000 + int(n[1:]), lab)
        dhn = read_runs(hn, lambda n: "worm", lambda n: 2000 + int(n[1:]), lab)
        Sx, cx, rx = det.score(dfx); Sh, ch, rh = det.score(dhn)
        S = {m: np.r_[Sx[m][cx == "worm"], Sh[m][ch == "normal"]] for m in METHODS}
        pos = np.r_[np.ones((cx == "worm").sum(), bool), np.zeros((ch == "normal").sum(), bool)]
        runs = np.r_[rx[cx == "worm"], rh[ch == "normal"]]
        res["stress"][cond] = metrics(det, S, pos, ~pos, runs)
        res["stress"][cond]["runs"] = {"endpoint_runs": len(fx), "honest_runs": len(hn)}
        print("stress", cond, {m: (round(res["stress"][cond][m]["auroc"] or 0, 3),
                                   round(res["stress"][cond][m]["p95"]["fpr"], 3)) for m in METHODS}, flush=True)

    # ---- operating points on the four dataset3 folds ----
    if not a_.skip_thresholds:
        acc = {h: {m: {q: {"tp": 0, "fp": 0, "P": 0, "N": 0, "runs": 0} for q in PCTS} for m in METHODS} for h in E.ATTACKS}
        for fi, (trs, vas, tes) in enumerate(E.FOLDS):
            aef = fit_ae(a, trs)
            te = a[a.seed.isin(tes)]
            for h in E.ATTACKS:
                d = Detector(a, trs, vas, h, aef)
                S, c, runs = d.score(te)
                pos, neg = c == h, c == "normal"
                for m in METHODS:
                    for q in PCTS:
                        t = np.percentile(d.val[m], q); z = acc[h][m][q]
                        z["tp"] += int((S[m][pos] > t).sum()); z["fp"] += int((S[m][neg] > t).sum())
                        z["P"] += int(pos.sum()); z["N"] += int(neg.sum()); z["runs"] += len(np.unique(runs[pos | neg]))
            print("thresholds fold", fi + 1, flush=True)
        for h in E.ATTACKS:
            res["thresholds"][h] = {m: {str(q): {"tpr": z["tp"] / z["P"], "fpr": z["fp"] / z["N"],
                                                 "precision": z["tp"] / (z["tp"] + z["fp"]) if z["tp"] + z["fp"] else None,
                                                 "false_alerts_per_run": z["fp"] / z["runs"]}
                                        for q, z in acc[h][m].items()} for m in METHODS}
    json.dump(res, open(out_path, "w"), indent=1)
    print("written", os.path.abspath(out_path))


if __name__ == "__main__":
    main()

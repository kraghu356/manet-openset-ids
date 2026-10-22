"""Supplementary metrics requested in the pre-submission audit.

For the non-neural scores (forest vote, k-NN distance, reconstruction error, nu,
the hybrid of Eq. 9 and k-NN + nu of Eq. 10), under the same folds, held-out
attacks, scalers and validation-only thresholds as analysis/openset_eval.py:

  * AUROC and average precision (PR-AUC) at the natural test prevalence
  * paired fold-level AUROC of k-NN + nu against k-NN alone
  * a run-clustered bootstrap 95% interval for AUROC, resampling test runs
    (not nodes) with replacement within each fold, 1,000 replicates
  * a no-speed ablation: the same scores on the 14 observable features that
    need no positioning or inertial source (onboard-only set)
  * per-fold, per-class counts of test node sequences (leakage audit)

Writes results/extra_metrics.json.
"""
import json, os, sys
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset_eval as E

rng = np.random.default_rng(0)
B = 1000
F_ONBOARD = [f for f in E.F_OBS if f != "speed"]


def node_runs(df):
    runs = []
    for (run, _), q in df.groupby(["run", "node_id"], sort=False):
        if q.fwd_seen.sum() == 0:
            continue
        runs.append(run)
    return np.array(runs)


def auc(p, n):
    if len(np.unique(np.r_[p, n])) == 1:
        return 0.5
    return float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]))


def ap(p, n):
    return float(average_precision_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]))


def run(feats, tag, a):
    out = {"auroc": {}, "ap": {}, "boot": {}, "counts": {}}
    methods = ["rf_vote", "knn_dist", "autoencoder", "nu", "hybrid", "knn_nu"]
    per = {h: {m: {"auroc": [], "ap": []} for m in methods} for h in E.ATTACKS}
    pooled = {h: {m: [] for m in methods} for h in E.ATTACKS}   # (fold, run, score, label) for bootstrap
    for fi, (trs, vas, tes) in enumerate(E.FOLDS):
        Xn, Mn, _, _, _ = E.node_table(a[a.seed.isin(trs) & (a.cls == "normal") & (a.scenario == "normal")], feats)
        sca = StandardScaler().fit(Xn.reshape(-1, len(feats)))
        fa = lambda X: sca.transform(X.reshape(-1, len(feats))).reshape(X.shape).astype(np.float32)
        ae = E.train_ae(fa(Xn), Mn, 40)
        te = a[a.seed.isin(tes)]
        Xte, Mte, cte, nute, _ = E.node_table(te, feats); rte = node_runs(te)
        ae_te = E.recon(ae, fa(Xte), Mte)
        out["counts"][f"fold{fi + 1}"] = {c: int((cte == c).sum()) for c in sorted(set(cte))}
        for h in E.ATTACKS:
            known = sorted([c for c in E.ATTACKS if c != h] + ["normal"]); ki = {c: i for i, c in enumerate(known)}
            tr = a[a.seed.isin(trs) & (a.scenario != h)]; va = a[a.seed.isin(vas) & (a.scenario != h)]
            Xtr, Mtr, ctr, _, _ = E.node_table(tr, feats); Xva, Mva, cva, nuva, _ = E.node_table(va, feats)
            assert not np.isin(ctr, [h]).any() and not np.isin(cva, [h]).any()
            ytr = np.array([ki[c] for c in ctr])
            sc = StandardScaler().fit(Xtr.reshape(-1, len(feats)))
            f = lambda X: sc.transform(X.reshape(-1, len(feats))).reshape(X.shape).astype(np.float32)
            Ftr, Fva, Fte = E.flat(f(Xtr), Mtr), E.flat(f(Xva), Mva), E.flat(f(Xte), Mte)
            rf = RandomForestClassifier(300, random_state=0, class_weight="balanced", n_jobs=-1).fit(Ftr, ytr)
            kn = KNeighborsClassifier(5).fit(Ftr, ytr)
            ae_va = E.recon(ae, fa(Xva), Mva)
            S_te = {"rf_vote": 1 - E.vote(rf, Fte).max(1), "knn_dist": kn.kneighbors(Fte)[0].mean(1),
                    "autoencoder": ae_te, "nu": nute}
            S_va = {"rf_vote": 1 - E.vote(rf, Fva).max(1), "knn_dist": kn.kneighbors(Fva)[0].mean(1),
                    "autoencoder": ae_va, "nu": nuva}
            vneg = cva == "normal"
            r = lambda m, s: E.rank_against(S_va[m][vneg], s)
            ov_te, ov_va = (nute > 0) * (1.0 + nute), (nuva > 0) * (1.0 + nuva)
            S_te["hybrid"] = np.maximum(np.maximum(r("rf_vote", S_te["rf_vote"]), r("autoencoder", S_te["autoencoder"])), ov_te)
            S_te["knn_nu"] = np.maximum(r("knn_dist", S_te["knn_dist"]), ov_te)
            pos, neg = cte == h, cte == "normal"
            for m in methods:
                s = S_te[m]
                per[h][m]["auroc"].append(auc(s[pos], s[neg])); per[h][m]["ap"].append(ap(s[pos], s[neg]))
                keep = pos | neg
                pooled[h][m].append((fi, rte[keep], s[keep], pos[keep].astype(int)))
            print(f"  {tag} fold {fi + 1} {h}: knn={per[h]['knn_dist']['auroc'][-1]:.3f} "
                  f"knn_nu={per[h]['knn_nu']['auroc'][-1]:.3f} hybrid={per[h]['hybrid']['auroc'][-1]:.3f}", flush=True)
    # run-clustered bootstrap of the four-fold mean AUROC
    for h in E.ATTACKS:
        out["boot"][h] = {}
        for m in methods:
            reps = []
            for _ in range(B):
                vals = []
                for fi, runs, s, y in pooled[h][m]:
                    ur = np.unique(runs); pick = rng.choice(ur, len(ur), replace=True)
                    idx = np.concatenate([np.where(runs == q)[0] for q in pick])
                    yy, ss = y[idx], s[idx]
                    if yy.min() == yy.max():
                        continue
                    vals.append(auc(ss[yy == 1], ss[yy == 0]))
                if vals:
                    reps.append(np.mean(vals))
            out["boot"][h][m] = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
    out["auroc"] = {h: {m: per[h][m]["auroc"] for m in methods} for h in E.ATTACKS}
    out["ap"] = {h: {m: per[h][m]["ap"] for m in methods} for h in E.ATTACKS}
    return out


if __name__ == "__main__":
    a = E.load()
    res = {"F_observable": run(E.F_OBS, "obs", a), "F_onboard_no_speed": run(F_ONBOARD, "onboard", a),
           "prevalence_note": "AP is computed at the natural test prevalence of each held-out attack"}
    p = os.path.join(E.HERE, "..", "results", "extra_metrics.json")
    json.dump(res, open(p, "w"), indent=1)
    print("written", os.path.abspath(p))

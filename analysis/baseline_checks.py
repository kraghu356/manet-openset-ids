"""Checks on the two open-set baselines of analysis/openset_baselines.py.

1. Reciprocal-point learning (RPL). For six hyperparameter settings, per fold
   and held-out attack: final training loss, closed-set accuracy on the test
   nodes of the known classes (convergence check), AUROC of the score used in
   the paper (minus the largest reciprocal-point distance) and of the reversed
   score (plus the largest distance).

2. Mahalanobis distance against k-NN + nu (Eq. 10) and k-NN distance:
   fold-level mean AUROC and a run-clustered paired bootstrap (1,000
   replicates) of the difference in four-fold mean AUROC, resampling test runs
   within each fold, the same runs for both detectors.

Writes results/baseline_checks.json.
"""
import json, os, sys
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset_eval as E
from openset_baselines import mahalanobis, RPL

FEATS = E.F_OBS
torch.set_num_threads(2)
CONFIGS = [  # (lambda, hidden, epochs)
    (0.1, 64, 60), (0.01, 64, 60), (1.0, 64, 60), (0.1, 128, 60), (0.1, 64, 150), (0.1, 32, 30)]


def auc(p, n):
    z = np.r_[p, n]
    return 0.5 if len(np.unique(z)) == 1 else float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], z))


def train_rpl(Ftr, ytr, k, lam, h, ep, seed=0):
    torch.manual_seed(seed)
    m = RPL(Ftr.shape[1], k, h); opt = torch.optim.Adam(m.parameters(), 1e-3)
    X = torch.tensor(Ftr, dtype=torch.float32); Y = torch.tensor(ytr)
    w = torch.tensor(len(ytr) / (k * np.bincount(ytr, minlength=k).clip(1)), dtype=torch.float32)
    for _ in range(ep):
        perm = torch.randperm(len(X)); tot = 0.0
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]; d = m(X[b])
            loss = nn.functional.cross_entropy(d, Y[b], weight=w) + lam * ((d.gather(1, Y[b][:, None]).squeeze(1) - m.R) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
    m.eval()
    with torch.no_grad():
        dmax = lambda F: m(torch.tensor(F, dtype=torch.float32)).max(1)
    return m, tot / len(X), dmax


def node_runs(df):
    return np.array([r for (r, _), q in df.groupby(["run", "node_id"], sort=False) if q.fwd_seen.sum() > 0])


def main():
    a = E.load(); rng = np.random.default_rng(0)
    rpl = {str(c): {h: {"loss": [], "closed_acc": [], "auroc": [], "auroc_reversed": []} for h in E.ATTACKS} for c in CONFIGS}
    pooled = {h: [] for h in E.ATTACKS}
    fold_auc = {m: [] for m in ["knn_dist", "mahalanobis", "knn_nu"]}
    for fi, (trs, vas, tes) in enumerate(E.FOLDS):
        te = a[a.seed.isin(tes)]
        Xte, Mte, cte, nute, _ = E.node_table(te, FEATS); rte = node_runs(te)
        per = {m: [] for m in fold_auc}
        for h in E.ATTACKS:
            known = sorted([c for c in E.ATTACKS if c != h] + ["normal"]); ki = {c: i for i, c in enumerate(known)}
            Xtr, Mtr, ctr, _, _ = E.node_table(a[a.seed.isin(trs) & (a.scenario != h)], FEATS)
            Xva, Mva, cva, _, _ = E.node_table(a[a.seed.isin(vas) & (a.scenario != h)], FEATS)
            ytr = np.array([ki[c] for c in ctr])
            sc = StandardScaler().fit(Xtr.reshape(-1, len(FEATS)))
            f = lambda X: sc.transform(X.reshape(-1, len(FEATS))).reshape(X.shape).astype(np.float32)
            Ftr, Fva, Fte = E.flat(f(Xtr), Mtr), E.flat(f(Xva), Mva), E.flat(f(Xte), Mte)
            pos, neg = cte == h, cte == "normal"
            known_te = np.isin(cte, known)
            yk = np.array([ki[c] for c in cte[known_te]])
            for cfg in CONFIGS:
                m, loss, dmax = train_rpl(Ftr, ytr, len(known), *cfg)
                with torch.no_grad():
                    D = m(torch.tensor(Fte, dtype=torch.float32)).numpy()
                s = -D.max(1)
                r = rpl[str(cfg)][h]
                r["loss"].append(loss); r["closed_acc"].append(float((D[known_te].argmax(1) == yk).mean()))
                r["auroc"].append(auc(s[pos], s[neg])); r["auroc_reversed"].append(auc(-s[pos], -s[neg]))
            # Mahalanobis vs k-NN + nu
            kn = KNeighborsClassifier(5).fit(Ftr, ytr); mh = mahalanobis(Ftr, ytr)
            vneg = cva == "normal"
            kv = kn.kneighbors(Fva)[0].mean(1)[vneg]; kt = kn.kneighbors(Fte)[0].mean(1)
            S = {"knn_dist": kt, "mahalanobis": mh(Fte),
                 "knn_nu": np.maximum(E.rank_against(kv, kt), (nute > 0) * (1.0 + nute))}
            keep = pos | neg
            pooled[h].append((rte[keep], {k: v[keep] for k, v in S.items()}, pos[keep].astype(int)))
            for k in per:
                per[k].append(auc(S[k][pos], S[k][neg]))
            print(f"fold {fi + 1} {h}: rpl default auroc={rpl[str(CONFIGS[0])][h]['auroc'][-1]:.3f} "
                  f"reversed={rpl[str(CONFIGS[0])][h]['auroc_reversed'][-1]:.3f} acc={rpl[str(CONFIGS[0])][h]['closed_acc'][-1]:.3f} "
                  f"maha={per['mahalanobis'][-1]:.3f} knn_nu={per['knn_nu'][-1]:.3f}", flush=True)
        for k in per:
            fold_auc[k].append(float(np.mean(per[k])))
    # paired run-clustered bootstrap of mean AUROC differences
    pairs = [("knn_nu", "mahalanobis"), ("mahalanobis", "knn_dist"), ("knn_nu", "knn_dist")]
    reps = {f"{x}-{y}": [] for x, y in pairs}
    reps_worm = {f"{x}-{y}": [] for x, y in pairs}
    for _ in range(1000):
        acc = {k: [] for k in reps}; accw = {k: [] for k in reps}
        for h in E.ATTACKS:
            for runs, S, y in pooled[h]:
                ur = np.unique(runs); pick = rng.choice(ur, len(ur), replace=True)
                idx = np.concatenate([np.where(runs == q)[0] for q in pick])
                if y[idx].min() == y[idx].max():
                    continue
                for x, z in pairs:
                    d = auc(S[x][idx][y[idx] == 1], S[x][idx][y[idx] == 0]) - auc(S[z][idx][y[idx] == 1], S[z][idx][y[idx] == 0])
                    acc[f"{x}-{z}"].append(d)
                    if h == "worm":
                        accw[f"{x}-{z}"].append(d)
        for k in reps:
            reps[k].append(np.mean(acc[k])); reps_worm[k].append(np.mean(accw[k]))
    out = {"rpl": {}, "fold_auroc": fold_auc, "paired_bootstrap": {}}
    for c in CONFIGS:
        r = rpl[str(c)]
        out["rpl"][str(c)] = {"lambda": c[0], "hidden": c[1], "epochs": c[2],
                              "mean_auroc": float(np.mean([np.mean(r[h]["auroc"]) for h in E.ATTACKS])),
                              "mean_auroc_reversed": float(np.mean([np.mean(r[h]["auroc_reversed"]) for h in E.ATTACKS])),
                              "closed_acc": float(np.mean([np.mean(r[h]["closed_acc"]) for h in E.ATTACKS])),
                              "final_loss": float(np.mean([np.mean(r[h]["loss"]) for h in E.ATTACKS])),
                              "per_attack_auroc": {h: float(np.mean(r[h]["auroc"])) for h in E.ATTACKS}}
    for k in reps:
        out["paired_bootstrap"][k] = {"mean_diff_ci": [float(np.percentile(reps[k], 2.5)), float(np.percentile(reps[k], 97.5))],
                                     "worm_diff_ci": [float(np.percentile(reps_worm[k], 2.5)), float(np.percentile(reps_worm[k], 97.5))]}
    p = os.path.join(E.HERE, "..", "results", "baseline_checks.json")
    json.dump(out, open(p, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()

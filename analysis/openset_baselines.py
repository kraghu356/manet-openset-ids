"""Two recent open-set baselines and a score-fusion ablation, under the protocol of
analysis/openset_eval.py (four seed folds, each attack held out of training and
validation in turn, thresholds at the 95th percentile of validation-normal scores).

Baselines, both on the same standardised, masked-mean node features as k-NN:
  mahalanobis  class-conditional Gaussian with a shared covariance; novelty
               score is the smallest Mahalanobis distance to a known-class mean
               (Lee et al., NeurIPS 2018)
  rpl          reciprocal-point learning (Chen et al., ECCV 2020; adversarial
               variant ARPL, TPAMI 2022, without the generator): a two-layer
               embedding with one learnable reciprocal point per known class;
               the class logit is the squared distance to its reciprocal point,
               trained with cross-entropy plus the open-space margin loss;
               novelty score is minus the largest logit

Fusion ablation for the two combined detectors (Eqs. 9 and 10):
  max   the rule used in the paper
  mean  mean of the learned ranks, then the nu override
  sum   sum of the learned ranks and the nu term (I[nu>0](1+nu))

Writes results/openset_baselines.json.
"""
import json, os, sys
import numpy as np
import torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset_eval as E

FEATS = E.F_OBS
torch.set_num_threads(2)


def auc(p, n):
    z = np.r_[p, n]
    if len(np.unique(z)) == 1:
        return 0.5
    return float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], z))


def mahalanobis(Ftr, ytr):
    cls = np.unique(ytr)
    mu = np.stack([Ftr[ytr == c].mean(0) for c in cls])
    R = np.concatenate([Ftr[ytr == c] - mu[i] for i, c in enumerate(cls)])
    P = np.linalg.pinv(np.cov(R, rowvar=False) + 1e-6 * np.eye(R.shape[1]))
    def score(F):
        d = np.stack([np.einsum("ij,jk,ik->i", F - m, P, F - m) for m in mu], 1)
        return d.min(1)
    return score


class RPL(nn.Module):
    def __init__(s, f, k, h=64):
        super().__init__()
        s.emb = nn.Sequential(nn.Linear(f, h), nn.ReLU(), nn.Linear(h, h))
        s.P = nn.Parameter(0.1 * torch.randn(k, h))
        s.R = nn.Parameter(torch.zeros(1))

    def forward(s, x):
        z = s.emb(x)
        return ((z[:, None, :] - s.P[None]) ** 2).mean(-1)   # (n, k) squared distances


def rpl(Ftr, ytr, k, ep=60, lam=0.1, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = RPL(Ftr.shape[1], k)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    X = torch.tensor(Ftr, dtype=torch.float32); Y = torch.tensor(ytr)
    w = torch.tensor(len(ytr) / (k * np.bincount(ytr, minlength=k).clip(1)), dtype=torch.float32)
    for _ in range(ep):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            d = m(X[b])
            loss = nn.functional.cross_entropy(d, Y[b], weight=w)
            dy = d.gather(1, Y[b][:, None]).squeeze(1)
            loss = loss + lam * ((dy - m.R) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    def score(F):
        with torch.no_grad():
            return -m(torch.tensor(F, dtype=torch.float32)).max(1).values.numpy()
    return score


def main():
    a = E.load()
    names = ["knn_dist", "mahalanobis", "rpl",
             "knn_nu_max", "knn_nu_sum", "hybrid_max", "hybrid_mean", "hybrid_sum"]
    res = {m: {h: {"auroc": [], "tpr95": [], "fpr95": []} for h in E.ATTACKS} for m in names}
    for fi, (trs, vas, tes) in enumerate(E.FOLDS):
        Xn, Mn, _, _, _ = E.node_table(a[a.seed.isin(trs) & (a.cls == "normal") & (a.scenario == "normal")], FEATS)
        sca = StandardScaler().fit(Xn.reshape(-1, len(FEATS)))
        fa = lambda X: sca.transform(X.reshape(-1, len(FEATS))).reshape(X.shape).astype(np.float32)
        ae = E.train_ae(fa(Xn), Mn, 40)
        Xte, Mte, cte, nute, _ = E.node_table(a[a.seed.isin(tes)], FEATS)
        ae_te = E.recon(ae, fa(Xte), Mte)
        for h in E.ATTACKS:
            known = sorted([c for c in E.ATTACKS if c != h] + ["normal"]); ki = {c: i for i, c in enumerate(known)}
            Xtr, Mtr, ctr, _, _ = E.node_table(a[a.seed.isin(trs) & (a.scenario != h)], FEATS)
            Xva, Mva, cva, nuva, _ = E.node_table(a[a.seed.isin(vas) & (a.scenario != h)], FEATS)
            ytr = np.array([ki[c] for c in ctr])
            sc = StandardScaler().fit(Xtr.reshape(-1, len(FEATS)))
            f = lambda X: sc.transform(X.reshape(-1, len(FEATS))).reshape(X.shape).astype(np.float32)
            Ftr, Fva, Fte = E.flat(f(Xtr), Mtr), E.flat(f(Xva), Mva), E.flat(f(Xte), Mte)
            kn = KNeighborsClassifier(5).fit(Ftr, ytr)
            rf = RandomForestClassifier(300, random_state=0, class_weight="balanced", n_jobs=-1).fit(Ftr, ytr)
            mh = mahalanobis(Ftr, ytr); rp = rpl(Ftr, ytr, len(known))
            vneg = cva == "normal"
            raw_va = {"knn_dist": kn.kneighbors(Fva)[0].mean(1), "mahalanobis": mh(Fva), "rpl": rp(Fva),
                      "rf": 1 - E.vote(rf, Fva).max(1), "ae": E.recon(ae, fa(Xva), Mva)}
            raw_te = {"knn_dist": kn.kneighbors(Fte)[0].mean(1), "mahalanobis": mh(Fte), "rpl": rp(Fte),
                      "rf": 1 - E.vote(rf, Fte).max(1), "ae": ae_te}
            rk = lambda m, s: E.rank_against(raw_va[m][vneg], s)
            def combos(raw, nu):
                ov = (nu > 0) * (1.0 + nu)
                rkn, rrf, rae = rk("knn_dist", raw["knn_dist"]), rk("rf", raw["rf"]), rk("ae", raw["ae"])
                return {"knn_dist": raw["knn_dist"], "mahalanobis": raw["mahalanobis"], "rpl": raw["rpl"],
                        "knn_nu_max": np.maximum(rkn, ov), "knn_nu_sum": rkn + ov,
                        "hybrid_max": np.maximum(np.maximum(rrf, rae), ov),
                        "hybrid_mean": np.maximum((rrf + rae) / 2, ov),
                        "hybrid_sum": rrf + rae + ov}
            S_va, S_te = combos(raw_va, nuva), combos(raw_te, nute)
            pos, neg = cte == h, cte == "normal"
            for m in names:
                t = np.percentile(S_va[m][vneg], 95)
                res[m][h]["auroc"].append(auc(S_te[m][pos], S_te[m][neg]))
                res[m][h]["tpr95"].append(float((S_te[m][pos] > t).mean()))
                res[m][h]["fpr95"].append(float((S_te[m][neg] > t).mean()))
            print(f"fold {fi + 1} {h}: " + " ".join(f"{m}={res[m][h]['auroc'][-1]:.3f}" for m in names), flush=True)
    out = {m: {h: {k: float(np.mean(v)) for k, v in res[m][h].items()} for h in E.ATTACKS} for m in names}
    for m in names:
        out[m]["mean_auroc"] = float(np.mean([out[m][h]["auroc"] for h in E.ATTACKS]))
        out[m]["mean_tpr95"] = float(np.mean([out[m][h]["tpr95"] for h in E.ATTACKS]))
        out[m]["max_fpr95"] = float(max(out[m][h]["fpr95"] for h in E.ATTACKS))
    out["folds_raw"] = res
    p = os.path.join(E.HERE, "..", "results", "openset_baselines.json")
    json.dump(out, open(p, "w"), indent=1)
    for m in names:
        print(m, round(out[m]["mean_auroc"], 3), {h: round(out[m][h]["auroc"], 3) for h in E.ATTACKS},
              "DR95", round(out[m]["mean_tpr95"], 2), "maxFPR95", round(out[m]["max_fpr95"], 3))
    print("written", os.path.abspath(p))


if __name__ == "__main__":
    main()

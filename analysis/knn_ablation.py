"""Ablations of the distance-plus-invariant detector (Eq. 10), four folds,
leave-one-attack-out, thresholds at the validation-normal 95th percentile.

  k        number of neighbours: 1, 3, 5 (paper), 10, 20
  groups   remove one feature group at a time from the 15 observable features:
           forwarding (fwd_seen, fwd_ratio, fwd_ratio_ewma), route discovery
           (rreq_recv, rreq_sent, rreq_sent_ewma, no_route, no_route_ewma),
           neighbourhood (nb_count, nb_churn, nb_churn_ewma), radio (rssi_min,
           rssi_dist_max), mobility (speed), and nu as a feature (nb_no_rssi;
           the nu override of Eq. 10 is kept)
  override the invariant term of Eq. 10 removed (k-NN distance alone) and the
           two-window rule for nu

Writes results/knn_ablation.json.
"""
import json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset_eval as E

GROUPS = {"forwarding": ["fwd_seen", "fwd_ratio", "fwd_ratio_ewma"],
          "route discovery": ["rreq_recv", "rreq_sent", "rreq_sent_ewma", "no_route", "no_route_ewma"],
          "neighbourhood": ["nb_count", "nb_churn", "nb_churn_ewma"],
          "radio": ["rssi_min", "rssi_dist_max"], "mobility": ["speed"], "nu as a feature": ["nb_no_rssi"]}


def auc(p, n):
    z = np.r_[p, n]
    return 0.5 if len(np.unique(z)) == 1 else float(roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], z))


def run(a, feats, k=5, override="nu"):
    per = {h: {"auroc": [], "tpr": [], "fpr": []} for h in E.ATTACKS}
    for trs, vas, tes in E.FOLDS:
        Xte, Mte, cte, nute, _ = E.node_table(a[a.seed.isin(tes)], feats); nu2te = E.node_table.nu2
        for h in E.ATTACKS:
            known = sorted([c for c in E.ATTACKS if c != h] + ["normal"]); ki = {c: i for i, c in enumerate(known)}
            Xtr, Mtr, ctr, _, _ = E.node_table(a[a.seed.isin(trs) & (a.scenario != h)], feats)
            Xva, Mva, cva, nuva, _ = E.node_table(a[a.seed.isin(vas) & (a.scenario != h)], feats); nu2va = E.node_table.nu2
            sc = StandardScaler().fit(Xtr.reshape(-1, len(feats)))
            f = lambda X: sc.transform(X.reshape(-1, len(feats))).reshape(X.shape).astype(np.float32)
            Ftr, Fva, Fte = E.flat(f(Xtr), Mtr), E.flat(f(Xva), Mva), E.flat(f(Xte), Mte)
            kn = KNeighborsClassifier(k).fit(Ftr, [ki[c] for c in ctr])
            vneg = cva == "normal"
            dv, dt = kn.kneighbors(Fva)[0].mean(1), kn.kneighbors(Fte)[0].mean(1)
            rv, rt = E.rank_against(dv[vneg], dv), E.rank_against(dv[vneg], dt)
            if override == "nu":
                sv, st = np.maximum(rv, (nuva > 0) * (1 + nuva)), np.maximum(rt, (nute > 0) * (1 + nute))
            elif override == "nu2w":
                sv, st = np.maximum(rv, 2 * nu2va), np.maximum(rt, 2 * nu2te)
            else:
                sv, st = rv, rt
            t = np.percentile(sv[vneg], 95); pos, neg = cte == h, cte == "normal"
            per[h]["auroc"].append(auc(st[pos], st[neg]))
            per[h]["tpr"].append(float((st[pos] > t).mean())); per[h]["fpr"].append(float((st[neg] > t).mean()))
    out = {h: {m: float(np.mean(v)) for m, v in d.items()} for h, d in per.items()}
    out["mean_auroc"] = float(np.mean([out[h]["auroc"] for h in E.ATTACKS]))
    out["mean_tpr"] = float(np.mean([out[h]["tpr"] for h in E.ATTACKS]))
    out["max_fpr"] = float(max(out[h]["fpr"] for h in E.ATTACKS))
    return out


def main():
    a = E.load(); res = {"k": {}, "groups": {}, "override": {}}
    for k in [1, 3, 5, 10, 20]:
        res["k"][k] = run(a, E.F_OBS, k=k); print("k", k, round(res["k"][k]["mean_auroc"], 3), flush=True)
    for g, cols in GROUPS.items():
        feats = [c for c in E.F_OBS if c not in cols]
        res["groups"][g] = run(a, feats); print("without", g, round(res["groups"][g]["mean_auroc"], 3), flush=True)
    for o in ["none", "nu2w"]:
        res["override"][o] = run(a, E.F_OBS, override=o); print("override", o, round(res["override"][o]["mean_auroc"], 3), flush=True)
    p = os.path.join(E.HERE, "..", "results", "knn_ablation.json")
    json.dump(res, open(p, "w"), indent=1); print("written", p)


if __name__ == "__main__":
    main()

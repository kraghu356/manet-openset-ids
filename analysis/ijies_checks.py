"""Analyses requested in the IJIES review, under the four-fold leave-one-attack-out
protocol of analysis/openset_eval.py (thresholds from validation-normal scores
only; the held-out attack never enters training or validation).

  python3 analysis/ijies_checks.py allnodes   every node, active or not, with the
                                             fraction of active windows as an extra input
  python3 analysis/ijies_checks.py gate       end-to-end decision: random-forest classifier
                                             plus the k-NN + nu gate, with time to first alarm
                                             from prefix scoring
  python3 analysis/ijies_checks.py dilution   per-feature reconstruction error of the
                                             autoencoder on wormhole endpoints
  python3 analysis/ijies_checks.py seeds      training seeds 0-2 for the recurrent classifier
                                             (softmax and energy), the autoencoder and the forest
  python3 analysis/ijies_checks.py dump       node-level scores of the non-neural detectors and the
                                             autoencoder for every fold and held-out attack, and
                                             autoencoder checkpoints (recurrent-classifier
                                             checkpoints for fold 1 are written by `seeds`)

Each writes results/ijies_<name>.json (dump writes results/node_scores.csv.gz and
results/checkpoints/).
"""
import json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openset_eval as E

FEATS = E.F_OBS
NU = FEATS.index("nb_no_rssi")
OUT = os.path.join(E.HERE, "..", "results")
torch.set_num_threads(2)


def auc(p, n):
    z = np.r_[p, n]
    return 0.5 if len(p) == 0 or len(n) == 0 or len(np.unique(z)) == 1 else float(
        roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], z))


def node_table_all(df, feats, active_input=False):
    """Like E.node_table but keeps every node; optionally appends the fraction of
    windows in which the node was offered packets as a constant input."""
    X, M, cls, nu, keys = [], [], [], [], []
    k = len(feats) + (1 if active_input else 0)
    for (run, nid), q in df.groupby(["run", "node_id"], sort=False):
        v = q[feats].to_numpy(np.float32)
        if active_input:
            v = np.c_[v, np.full(len(v), float((q.fwd_seen > 0).mean()), np.float32)]
        n = min(len(v), E.SEQ)
        p = np.zeros((E.SEQ, k), np.float32); m = np.zeros(E.SEQ, np.float32)
        p[:n] = v[:n]; m[:n] = 1
        X.append(p); M.append(m); cls.append(q.cls.iloc[0]); nu.append(float(q.nb_no_rssi.max())); keys.append((run, nid))
    return np.stack(X), np.stack(M), np.array(cls), np.array(nu), keys


def fit_scaler(X, k):
    sc = StandardScaler().fit(X.reshape(-1, k))
    return lambda Z: sc.transform(Z.reshape(-1, k)).reshape(Z.shape).astype(np.float32)


def known_of(h):
    known = sorted([c for c in E.ATTACKS if c != h] + ["normal"])
    return known, {c: i for i, c in enumerate(known)}


# ---------------------------------------------------------------- all nodes
def allnodes(a):
    res = {}
    for variant, act in [("active_only", None), ("all_nodes", False), ("all_nodes_with_activity_input", True)]:
        per = {h: {m: {"auroc": [], "tpr": [], "fpr": []} for m in ["knn_dist", "knn_nu", "hybrid"]} for h in E.ATTACKS}
        counts = {}
        for fi, (trs, vas, tes) in enumerate(E.FOLDS):
            nt = (lambda df: E.node_table(df, FEATS)[:4]) if act is None else (lambda df: node_table_all(df, FEATS, act)[:4])
            k = len(FEATS) + (1 if act else 0)
            Xn, Mn, cn, _ = nt(a[a.seed.isin(trs) & (a.scenario == "normal")])
            fa = fit_scaler(Xn, k); ae = E.train_ae(fa(Xn), Mn, 40)
            Xte, Mte, cte, nute = nt(a[a.seed.isin(tes)])
            counts[f"fold{fi + 1}"] = {c: int((cte == c).sum()) for c in sorted(set(cte))}
            ae_te = E.recon(ae, fa(Xte), Mte)
            for h in E.ATTACKS:
                known, ki = known_of(h)
                Xtr, Mtr, ctr, _ = nt(a[a.seed.isin(trs) & (a.scenario != h)])
                Xva, Mva, cva, nuva = nt(a[a.seed.isin(vas) & (a.scenario != h)])
                f = fit_scaler(Xtr, k); Ftr, Fva, Fte = E.flat(f(Xtr), Mtr), E.flat(f(Xva), Mva), E.flat(f(Xte), Mte)
                ytr = np.array([ki[c] for c in ctr])
                kn = KNeighborsClassifier(5).fit(Ftr, ytr)
                rf = RandomForestClassifier(300, random_state=0, class_weight="balanced", n_jobs=-1).fit(Ftr, ytr)
                vneg = cva == "normal"
                dv, dt = kn.kneighbors(Fva)[0].mean(1), kn.kneighbors(Fte)[0].mean(1)
                vv, vt = 1 - E.vote(rf, Fva).max(1), 1 - E.vote(rf, Fte).max(1)
                av, at = E.recon(ae, fa(Xva), Mva), ae_te
                r = lambda ref, s: E.rank_against(ref[vneg], s)
                ov = lambda nu: (nu > 0) * (1.0 + nu)
                Sv = {"knn_dist": dv, "knn_nu": np.maximum(r(dv, dv), ov(nuva)), "hybrid": np.maximum(np.maximum(r(vv, vv), r(av, av)), ov(nuva))}
                St = {"knn_dist": dt, "knn_nu": np.maximum(r(dv, dt), ov(nute)), "hybrid": np.maximum(np.maximum(r(vv, vt), r(av, at)), ov(nute))}
                pos, neg = cte == h, cte == "normal"
                for m in St:
                    t = np.percentile(Sv[m][vneg], 95)
                    per[h][m]["auroc"].append(auc(St[m][pos], St[m][neg]))
                    per[h][m]["tpr"].append(float((St[m][pos] > t).mean())); per[h][m]["fpr"].append(float((St[m][neg] > t).mean()))
            print(variant, "fold", fi + 1, flush=True)
        out = {m: {h: {k2: float(np.mean(v)) for k2, v in per[h][m].items()} for h in E.ATTACKS} for m in ["knn_dist", "knn_nu", "hybrid"]}
        for m in out:
            out[m]["mean_auroc"] = float(np.mean([out[m][h]["auroc"] for h in E.ATTACKS]))
            out[m]["mean_tpr"] = float(np.mean([out[m][h]["tpr"] for h in E.ATTACKS]))
            out[m]["max_fpr"] = float(max(out[m][h]["fpr"] for h in E.ATTACKS))
        out["test_counts"] = counts
        res[variant] = out
        print(variant, {m: round(out[m]["mean_auroc"], 3) for m in ["knn_dist", "knn_nu", "hybrid"]}, flush=True)
    return res


# ---------------------------------------------------------------- end-to-end gate
def gate(a):
    """Classifier (random forest) plus the k-NN + nu gate, run window by window.

    After each window w the node is scored on windows 1..w only. The distance
    is ranked against validation-normal nodes scored on the same w windows, so
    early, noisy prefixes are normalised; a node with nu > 0 so far scores 2.
    The running statistic is the maximum over w, and the threshold is a
    percentile of that statistic over validation-normal nodes, which fixes the
    rate at which a normal node raises an alarm at any time in a 290 s run.
    Final label: unknown if the gate has fired, otherwise the classifier's
    label on the full sequence. Latency is measured from the start of the run,
    when every attack starts, to the end of the first window that raises an
    alarm (gate, or the classifier predicting an attack class).
    """
    res = {q: {h: {} for h in E.ATTACKS} for q in ["p95", "p99"]}
    for fi, (trs, vas, tes) in enumerate(E.FOLDS):
        Xte, Mte, cte, nute, _ = E.node_table(a[a.seed.isin(tes)], FEATS)
        for h in E.ATTACKS:
            known, ki = known_of(h)
            Xtr, Mtr, ctr, _, _ = E.node_table(a[a.seed.isin(trs) & (a.scenario != h)], FEATS)
            Xva, Mva, cva, nuva, _ = E.node_table(a[a.seed.isin(vas) & (a.scenario != h)], FEATS)
            f = fit_scaler(Xtr, len(FEATS))
            Ftr = E.flat(f(Xtr), Mtr); ytr = np.array([ki[c] for c in ctr])
            kn = KNeighborsClassifier(5).fit(Ftr, ytr)
            rf = RandomForestClassifier(300, random_state=0, class_weight="balanced", n_jobs=-1).fit(Ftr, ytr)
            vneg = cva == "normal"; Xvn, Mvn = Xva[vneg], Mva[vneg]
            inormal = ki["normal"]
            Rv = np.zeros((len(Xvn), E.SEQ)); Rt = np.zeros((len(Xte), E.SEQ)); Pt = np.zeros((len(Xte), E.SEQ), int)
            for w in range(1, E.SEQ + 1):
                Mw_v = Mvn.copy(); Mw_v[:, w:] = 0; Mw_t = Mte.copy(); Mw_t[:, w:] = 0
                Fv, Ft = E.flat(f(Xvn), Mw_v), E.flat(f(Xte), Mw_t)
                dv, dt = kn.kneighbors(Fv)[0].mean(1), kn.kneighbors(Ft)[0].mean(1)
                nv, nt = Xvn[:, :w, NU].max(1), Xte[:, :w, NU].max(1)
                Rv[:, w - 1] = np.maximum(E.rank_against(dv, dv), (nv > 0) * 2.0)
                Rt[:, w - 1] = np.maximum(E.rank_against(dv, dt), (nt > 0) * 2.0)
                Pt[:, w - 1] = rf.predict(Ft)
            Sv, St = Rv.max(1), np.maximum.accumulate(Rt, 1)
            full_pred = rf.predict(E.flat(f(Xte), Mte))
            pos, neg = cte == h, cte == "normal"
            kn_mask = np.isin(cte, known); katt = kn_mask & ~neg
            yk = np.array([ki[c] for c in cte[kn_mask]])
            for q, pct in [("p95", 95), ("p99", 99)]:
                tau = np.percentile(Sv, pct)
                fire = St > tau; g = fire.any(1)
                label = np.where(g, -1, full_pred)
                gate_t = np.where(g, fire.argmax(1), E.SEQ)
                rf_att = Pt != inormal
                rf_t = np.where(rf_att.any(1), rf_att.argmax(1), E.SEQ)
                any_t = np.minimum(gate_t, rf_t)
                alarm = any_t < E.SEQ
                lat = lambda m: (float(np.median((any_t[m & alarm] + 1) * 10)) if (m & alarm).any() else None)
                d = {"unknown_detection": float(g[pos].mean()),
                     "any_alarm_unseen": float(alarm[pos].mean()),
                     "false_alarm_normal": float(alarm[neg].mean()),
                     "gate_false_alarm_normal": float(g[neg].mean()),
                     "known_attack_alarm": float(alarm[katt].mean()),
                     "known_attack_flagged_unknown": float(g[katt].mean()),
                     "known_macro_f1_classifier": float(f1_score(yk, full_pred[kn_mask], average="macro")),
                     "known_macro_f1_with_gate": float(f1_score(yk, label[kn_mask], average="macro", labels=list(range(len(known))))),
                     "latency_s_median_unseen": lat(pos),
                     "latency_s_p90_unseen": (float(np.percentile((any_t[pos & alarm] + 1) * 10, 90)) if (pos & alarm).any() else None)}
                res[q][h].setdefault("folds", []).append(d)
            print("gate fold", fi + 1, h, {k: round(v, 3) if isinstance(v, float) else v for k, v in res["p95"][h]["folds"][-1].items()}, flush=True)
    for q in res:
        for h in E.ATTACKS:
            fl = res[q][h]["folds"]
            res[q][h]["mean"] = {k: (float(np.mean([x[k] for x in fl if x[k] is not None])) if any(x[k] is not None for x in fl) else None) for k in fl[0]}
    return res


# ---------------------------------------------------------------- dilution
def dilution(a):
    rows = {f: [] for f in FEATS}; tot, nuw, nuonly = [], [], []
    for fi, (trs, vas, tes) in enumerate(E.FOLDS):
        Xn, Mn, _, _, _ = E.node_table(a[a.seed.isin(trs) & (a.cls == "normal") & (a.scenario == "normal")], FEATS)
        fa = fit_scaler(Xn, len(FEATS)); ae = E.train_ae(fa(Xn), Mn, 40)
        Xte, Mte, cte, _, _ = E.node_table(a[a.seed.isin(tes)], FEATS)
        Z = fa(Xte)
        with torch.no_grad():
            R = ae(torch.from_numpy(Z)).numpy()
        k = Mte[:, :, None]
        per = (((R - Z) ** 2) * k).sum(1) / k.sum(1).clip(1)        # (nodes, features)
        pos, neg = cte == "worm", cte == "normal"
        for j, fname in enumerate(FEATS):
            rows[fname].append(auc(per[pos, j], per[neg, j]))
        tot.append(auc(per[pos].mean(1), per[neg].mean(1)))
        w = np.ones(len(FEATS)); w[NU] = len(FEATS)                 # nu given the weight of all others
        nuw.append(auc((per[pos] * w).sum(1), (per[neg] * w).sum(1)))
        nuonly.append(auc(per[pos, NU], per[neg, NU]))
        print("dilution fold", fi + 1, round(tot[-1], 3), round(nuonly[-1], 3), round(nuw[-1], 3), flush=True)
    return {"wormhole_auroc_total_residual": float(np.mean(tot)), "wormhole_auroc_nu_residual_only": float(np.mean(nuonly)),
            "wormhole_auroc_nu_weighted": float(np.mean(nuw)), "per_feature": {f: float(np.mean(v)) for f, v in rows.items()},
            "folds": {"total": tot, "nu_only": nuonly, "nu_weighted": nuw},
            "scaler_note": "StandardScaler on benign training nodes; a zero-variance feature keeps scale 1 and mean 0, so nu enters unscaled"}


# ---------------------------------------------------------------- training seeds
def seeds(a):
    res = {s: {"msp_gru": {}, "energy_gru": {}, "autoencoder": {}, "hybrid": {}} for s in [0, 1, 2]}
    for s in [0, 1, 2]:
        for fi, (trs, vas, tes) in enumerate(E.FOLDS):
            Xn, Mn, _, _, _ = E.node_table(a[a.seed.isin(trs) & (a.cls == "normal") & (a.scenario == "normal")], FEATS)
            fa = fit_scaler(Xn, len(FEATS)); ae = E.train_ae(fa(Xn), Mn, 40, seed=s)
            Xte, Mte, cte, nute, _ = E.node_table(a[a.seed.isin(tes)], FEATS)
            ae_te = E.recon(ae, fa(Xte), Mte)
            for h in E.ATTACKS:
                known, ki = known_of(h)
                Xtr, Mtr, ctr, _, _ = E.node_table(a[a.seed.isin(trs) & (a.scenario != h)], FEATS)
                Xva, Mva, cva, nuva, _ = E.node_table(a[a.seed.isin(vas) & (a.scenario != h)], FEATS)
                f = fit_scaler(Xtr, len(FEATS))
                ytr = np.array([ki[c] for c in ctr]); yva = np.array([ki[c] for c in cva])
                pos, neg = cte == h, cte == "normal"
                if fi == 0:   # recurrent classifier: fold 1 only, to bound compute
                    m = E.train_seq("gru", f(Xtr), Mtr, ytr, f(Xva), Mva, yva, len(known), 30, seed=s)
                    lg, _ = E.logits_and_features(m, f(Xte), Mte)
                    msp, en = E.msp(lg), E.energy(lg)
                    res[s]["msp_gru"].setdefault(h, []).append(auc(msp[pos], msp[neg]))
                    res[s]["energy_gru"].setdefault(h, []).append(auc(en[pos], en[neg]))
                    if s == 0:
                        os.makedirs(os.path.join(OUT, "checkpoints"), exist_ok=True)
                        torch.save(m.state_dict(), os.path.join(OUT, "checkpoints", f"gru_fold1_heldout_{h}.pt"))
                res[s]["autoencoder"].setdefault(h, []).append(auc(ae_te[pos], ae_te[neg]))
                Ftr, Fva, Fte = E.flat(f(Xtr), Mtr), E.flat(f(Xva), Mva), E.flat(f(Xte), Mte)
                rf = RandomForestClassifier(300, random_state=s, class_weight="balanced", n_jobs=-1).fit(Ftr, ytr)
                vneg = cva == "normal"
                vv, vt = 1 - E.vote(rf, Fva).max(1), 1 - E.vote(rf, Fte).max(1)
                av = E.recon(ae, fa(Xva), Mva)
                hy = np.maximum(np.maximum(E.rank_against(vv[vneg], vt), E.rank_against(av[vneg], ae_te)), (nute > 0) * (1.0 + nute))
                res[s]["hybrid"].setdefault(h, []).append(auc(hy[pos], hy[neg]))
            print("seed", s, "fold", fi + 1, {k: round(np.mean([np.mean(v) for v in res[s][k].values()]), 3) for k in res[s] if res[s][k]}, flush=True)
    summ = {}
    for k in ["msp_gru", "energy_gru", "autoencoder", "hybrid"]:
        means = [float(np.mean([np.mean(res[s][k][h]) for h in E.ATTACKS])) for s in [0, 1, 2]]
        summ[k] = {"folds": "fold 1 only" if k in ("msp_gru", "energy_gru") else "all four", "mean_auroc_per_seed": means, "mean": float(np.mean(means)), "sd": float(np.std(means)),
                   "per_attack": {h: [float(np.mean(res[s][k][h])) for s in [0, 1, 2]] for h in E.ATTACKS}}
    return summ


# ---------------------------------------------------------------- node-level scores and checkpoints
def dump(a):
    os.makedirs(os.path.join(OUT, "checkpoints"), exist_ok=True)
    rows = []
    from openset_baselines import mahalanobis
    for fi, (trs, vas, tes) in enumerate(E.FOLDS):
        Xn, Mn, _, _, _ = E.node_table(a[a.seed.isin(trs) & (a.cls == "normal") & (a.scenario == "normal")], FEATS)
        fa = fit_scaler(Xn, len(FEATS)); ae = E.train_ae(fa(Xn), Mn, 40)
        torch.save(ae.state_dict(), os.path.join(OUT, "checkpoints", f"autoencoder_fold{fi + 1}.pt"))
        te = a[a.seed.isin(tes)]
        Xte, Mte, cte, nute, _ = E.node_table(te, FEATS)
        keys = [(r, n) for (r, n), q in te.groupby(["run", "node_id"], sort=False) if q.fwd_seen.sum() > 0]
        ae_te = E.recon(ae, fa(Xte), Mte)
        for h in E.ATTACKS:
            known, ki = known_of(h)
            Xtr, Mtr, ctr, _, _ = E.node_table(a[a.seed.isin(trs) & (a.scenario != h)], FEATS)
            Xva, Mva, cva, nuva, _ = E.node_table(a[a.seed.isin(vas) & (a.scenario != h)], FEATS)
            f = fit_scaler(Xtr, len(FEATS)); ytr = np.array([ki[c] for c in ctr]); yva = np.array([ki[c] for c in cva])
            Ftr, Fva, Fte = E.flat(f(Xtr), Mtr), E.flat(f(Xva), Mva), E.flat(f(Xte), Mte)
            kn = KNeighborsClassifier(5).fit(Ftr, ytr)
            rf = RandomForestClassifier(300, random_state=0, class_weight="balanced", n_jobs=-1).fit(Ftr, ytr)
            mh = mahalanobis(Ftr, ytr)
            vneg = cva == "normal"
            dv, dt = kn.kneighbors(Fva)[0].mean(1), kn.kneighbors(Fte)[0].mean(1)
            vv, vt = 1 - E.vote(rf, Fva).max(1), 1 - E.vote(rf, Fte).max(1)
            av = E.recon(ae, fa(Xva), Mva)
            ov = (nute > 0) * (1.0 + nute)
            sc = {"rf_vote": vt, "knn_dist": dt, "mahalanobis": mh(Fte),
                  "autoencoder": ae_te, "nu": nute,
                  "knn_nu": np.maximum(E.rank_against(dv[vneg], dt), ov),
                  "hybrid": np.maximum(np.maximum(E.rank_against(vv[vneg], vt), E.rank_against(av[vneg], ae_te)), ov)}
            pred = np.array(known)[rf.predict(Fte)]
            df = pd.DataFrame({"fold": fi + 1, "held_out": h, "run": [k[0] for k in keys], "node_id": [k[1] for k in keys],
                               "cls": cte, "rf_prediction": pred, **{k: v for k, v in sc.items()}})
            rows.append(df)
            print("dump fold", fi + 1, h, flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(os.path.join(OUT, "node_scores.csv.gz"), index=False, compression="gzip")
    return {"rows": len(out)}


if __name__ == "__main__":
    what = sys.argv[1]
    a = E.load()
    r = {"allnodes": allnodes, "gate": gate, "dilution": dilution, "seeds": seeds, "dump": dump}[what](a)
    json.dump(r, open(os.path.join(OUT, f"ijies_{what}.json"), "w"), indent=1)
    print("written", what)

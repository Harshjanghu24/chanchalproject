"""CDAS Step 4: Bayesian Belief Network for AI anxiety (correlational structure, not causal).
Structure learned with HillClimbSearch + BIC on discretised scale scores.
Assumptions encoded as constraints: anxiety has no outgoing edges (it is the outcome),
and demographics (gender, country, year) have no parents among the attitude scales.
Evaluated with the same repeated user-grouped CV as cv_eval.py, so numbers are comparable."""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.inference import VariableElimination
try:
    from pgmpy.estimators import BIC as BicScore
except ImportError:  # older pgmpy
    from pgmpy.estimators import BicScore
try:
    from pgmpy.models import DiscreteBayesianNetwork as BN
except ImportError:  # older pgmpy
    from pgmpy.models import BayesianNetwork as BN

import data_prep as dp

warnings.filterwarnings("ignore")
SEED, REPS, K = 42, 5, 5
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS, exist_ok=True)

TARGET = "anxiety"
SCALES = ["RE", "R", "SG", "L", "CM"]
DEMO = ["Gender", "Country3", "year"]
CONT = [TARGET] + SCALES


def build():
    raw = dp.load()
    ycol = dp._find(raw.columns, "year")
    d = raw[["user_id", TARGET] + SCALES + ["Gender", "Country"]].copy()
    d["year"] = raw[ycol].astype(str)
    top3 = d["Country"].value_counts().index[:3]
    d["Country3"] = d["Country"].where(d["Country"].isin(top3), "other")
    return d.drop(columns="Country").dropna().reset_index(drop=True)


def discretise(train, test=None):
    """Tertile edges are fitted on TRAIN only, then applied to test (no leakage)."""
    edges = {c: np.unique(np.quantile(train[c], [1 / 3, 2 / 3])) for c in CONT}

    def apply(df):
        out = df[CONT + DEMO].copy()
        for c in CONT:
            out[c] = np.digitize(df[c].to_numpy(), edges[c])
        return out

    return apply(train), (apply(test) if test is not None else None), edges


def learn(disc):
    cols = list(disc.columns)
    black = [(TARGET, c) for c in cols if c != TARGET]
    black += [(s, dm) for s in SCALES for dm in DEMO]
    hc = HillClimbSearch(disc)
    try:  # older pgmpy
        dag = hc.estimate(scoring_method=BicScore(disc), max_indegree=3,
                          black_list=black, show_progress=False)
    except TypeError:  # pgmpy >= 1.0 uses expert knowledge
        from pgmpy.causal_discovery import ExpertKnowledge
        dag = hc.estimate(scoring_method=BicScore(disc), max_indegree=3,
                          expert_knowledge=ExpertKnowledge(forbidden_edges=black),
                          show_progress=False)
    model = BN(list(dag.edges()))
    model.add_nodes_from(cols)
    est = BayesianEstimator(model, disc)  # works across pgmpy versions
    model.add_cpds(*[est.estimate_cpd(n, prior_type="BDeu", equivalent_sample_size=10)
                     for n in model.nodes()])
    return model


def predict(model, test_disc, bin_means):
    infer = VariableElimination(model)
    states = {n: list(model.get_cpds(n).state_names[n]) for n in model.nodes()}
    cache, exp, cls = {}, [], []
    for _, row in test_disc.drop(columns=[TARGET]).iterrows():
        ev = {n: row[n] for n in row.index if n in states and row[n] in states[n]}
        key = tuple(sorted(ev.items(), key=lambda kv: kv[0]))
        if key not in cache:
            q = infer.query([TARGET], evidence=ev, show_progress=False)
            st, pr = list(q.state_names[TARGET]), np.asarray(q.values, dtype=float)
            cache[key] = (sum(p * bin_means[s] for s, p in zip(st, pr)), st[int(pr.argmax())])
        e, c = cache[key]
        exp.append(e)
        cls.append(c)
    return np.array(exp), np.array(cls)


def cv_bn(d):
    rng = np.random.default_rng(SEED)
    users = np.unique(d["user_id"])
    r2s, accs, majs = [], [], []
    for rep in range(REPS):
        perm = rng.permutation(users)
        fold_of = {u: i % K for i, u in enumerate(perm)}
        folds = np.array([fold_of[u] for u in d["user_id"]])
        pred, base = np.zeros(len(d)), np.zeros(len(d))
        hit = maj = 0
        for f in range(K):
            tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
            train, test = d.iloc[tr], d.iloc[te]
            trd, ted, _ = discretise(train, test)
            model = learn(trd)
            bin_means = pd.Series(train[TARGET].to_numpy()).groupby(trd[TARGET].to_numpy()).mean().to_dict()
            e, c = predict(model, ted, bin_means)
            pred[te], base[te] = e, train[TARGET].mean()
            hit += int((c == ted[TARGET].to_numpy()).sum())
            maj += int((ted[TARGET].to_numpy() == trd[TARGET].mode().iloc[0]).sum())
        y = d[TARGET].to_numpy()
        r2s.append(1 - ((y - pred) ** 2).sum() / ((y - base) ** 2).sum())
        accs.append(hit / len(d))
        majs.append(maj / len(d))
        print(f"rep {rep + 1}/{REPS}: R2={r2s[-1]:.3f}  bin acc={accs[-1]:.3f}  majority={majs[-1]:.3f}")
    return r2s, accs, majs


def draw(model, path):
    g = nx.DiGraph(model.edges())
    g.add_nodes_from(model.nodes())
    pos = nx.spring_layout(g, seed=SEED, k=1.2)
    colors = ["#e07b39" if n == TARGET else ("#7fb3d5" if n in SCALES else "#b8b8b8") for n in g.nodes()]
    plt.figure(figsize=(8, 6))
    nx.draw_networkx(g, pos, node_color=colors, node_size=2200, font_size=9, arrows=True, arrowsize=18)
    plt.title("Learned Bayesian network (statistical dependencies, not causal effects)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


if __name__ == "__main__":
    d = build()
    print(f"BN data: n={len(d)}, users={d['user_id'].nunique()}")
    r2s, accs, majs = cv_bn(d)
    print(f"\nBayesian Network out-of-sample R2 = {np.mean(r2s):.3f} (sd {np.std(r2s):.3f})")
    print(f"3-bin accuracy = {np.mean(accs):.3f} vs majority-class baseline {np.mean(majs):.3f}")
    pd.DataFrame({"r2": r2s, "bin_acc": accs, "majority_acc": majs}).to_csv(
        os.path.join(RESULTS, "bn_cv.csv"), index=False)

    full, _, _ = discretise(d)
    model = learn(full)
    draw(model, os.path.join(RESULTS, "bn_graph.png"))
    print("\nLearned edges (parent -> child):")
    for a, b in sorted(model.edges()):
        print(f"  {a} -> {b}")
    print("Parents of anxiety:", sorted(model.get_parents(TARGET)))
    print(f"Saved graph to {RESULTS}")
"""CDAS Step 5: the three extra public AI datasets (outcomes are proxies, see notes in output).
Same out-of-sample R2 protocol as cv_eval.py (10 x 5-fold, vs train-mean baseline).
These files have no respondent id, so rows are treated as independent.
Correlational only. Run:  python multi_datasets.py"""
import glob
import os
import re
import sys
import traceback
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from cv_eval import oos_r2
from train_models import make_pre, RESULTS, SEED

warnings.filterwarnings("ignore")
BASE = sys.argv[1] if len(sys.argv) > 1 else r"D:\chanchal project\datasets"
ROWS = []


def first(pattern):
    hits = glob.glob(os.path.join(BASE, pattern))
    if not hits:
        raise FileNotFoundError(f"no file matching {pattern} in {BASE}")
    return hits[0]


def makers(num, cat):
    return {
        "Ridge": lambda: Pipeline([("pre", make_pre(num, cat)), ("m", Ridge(alpha=10))]),
        "RandomForest": lambda: Pipeline([("pre", make_pre(num, cat)), ("m", RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=10, random_state=SEED, n_jobs=-1))]),
        "XGBoost": lambda: Pipeline([("pre", make_pre(num, cat)), ("m", XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=2, subsample=0.8,
            tree_method="hist", random_state=SEED))]),
    }


def evaluate(name, d, num, cat, ycol):
    d = d.reset_index(drop=True)
    X, y = d[num + cat], d[ycol].to_numpy(float)
    print(f"\n{'=' * 70}\n{name}: n={len(d)}, outcome='{ycol}' (mean {y.mean():.2f}, sd {y.std():.2f})")
    print("Out-of-sample R2 (10 x 5-fold):")
    for mname, mk in makers(num, cat).items():
        m, s = oos_r2(X, y, np.arange(len(d)), mk)
        print(f"  {mname:<13} {m:.3f} (sd {s:.3f})")
        ROWS.append(dict(dataset=name, model=mname, n=len(d), r2_mean=m, r2_sd=s))

    ridge = makers(num, cat)["Ridge"]()
    ridge.fit(X, y)
    names = ridge.named_steps["pre"].get_feature_names_out()
    coef = pd.Series(ridge.named_steps["m"].coef_, index=names)
    nc = coef[[f"num__{c}" for c in num]].rename(lambda s: s.replace("num__", ""))
    print("Ridge: change in outcome per +1 SD of each numeric predictor (association):")
    print(nc.sort_values(key=abs, ascending=False).round(3).to_string())

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED)
    rf = makers(num, cat)["RandomForest"]()
    rf.fit(Xtr, ytr)
    pi = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=SEED, scoring="r2")
    imp = pd.Series(pi.importances_mean, index=num + cat).sort_values(ascending=False)
    print("RF permutation importance (holdout), top 6:")
    print(imp.head(6).round(4).to_string())


# ------------------------------------------------------------------ career anxiety
def run_career():
    c = pd.read_excel(first("Career_Anxiety*.xlsx"))
    c["y"] = c["career_anxiety"].map({"No Anxiety": 0, "Low": 1, "Medium": 2, "High": 3})
    c["knowledge"] = c["ai_knowledge"].map({"Low": 0, "Medium": 1, "High": 2})
    c["replace_jobs"] = c["ai_replace_jobs"].map({"No": 0, "Partially": 1, "Fully": 2})
    c["year"] = c["academic_year"].astype(str).str.extract(r"(\d)")[0].astype(float)
    c = c.rename(columns={"ai_takeover_time": "takeover_time", "ai_future_perspective": "future_view"})
    num = ["knowledge", "replace_jobs", "year", "age"]
    cat = ["takeover_time", "future_view", "gender", "university", "department"]
    d = c[num + cat + ["y", "career_anxiety", "ai_replace_jobs"]].dropna()
    print("\nCAREER ANXIETY: mean anxiety score (0=none .. 3=high) by 'will AI replace jobs?'")
    print(d.groupby("ai_replace_jobs")["y"].agg(["mean", "count"]).round(2).to_string())
    print("(ai_tools_used / career_path / ai_tool_perception are not used: near-constant or free text.)")
    evaluate("Career anxiety (perceived risk -> anxiety)", d, num, cat, "y")


# ------------------------------------------------------------------ cognitive strain / regret
def run_ex():
    d0 = pd.read_csv(first("Data_set_Ex.csv"))

    def col(n):
        for c in d0.columns:
            if re.match(rf"\s*{n}\.", str(c)):
                return d0[c]
        raise KeyError(f"item {n} not found")

    def item(n):
        return pd.to_numeric(col(n), errors="coerce")

    def mean_of(ns):
        return pd.concat([item(n) for n in ns], axis=1).mean(axis=1)

    d = pd.DataFrame({
        "regret": mean_of([14, 15, 16]),            # outcome: remorse/regret about over-reliance (proxy)
        "benefit": mean_of([17, 18, 19, 20, 21, 22]),  # time saved, confidence, efficiency, satisfaction, emotion
        "ease": mean_of([23, 24]),                   # lower mental effort / cognitive burden
        "cost": mean_of([11, 12, 13]),               # fear/awareness of weaker independent thinking
        "reliance": mean_of([6, 7]),
    })
    d["careful"] = pd.concat([item(8), item(9), 8 - item(10)], axis=1).mean(axis=1)
    d["experience"] = col(4).astype(str).str.strip().map(
        {"Less than 6 months": 0, "6-12 months": 1, "1-2 years": 2, "More than 2 years": 3})
    d["gender"] = col(1).astype(str).str.strip()
    d["field"] = np.where(col(3).astype(str).str.strip() == "CSE", "CSE", "other")
    d = d.dropna()
    num = ["benefit", "ease", "cost", "reliance", "careful", "experience"]
    cat = ["gender", "field"]
    print("\nNOTE: 'cost' (items 11-13) and 'regret' (14-16) are worded very similarly and sit in the same")
    print("survey block, so some association is built in. Treat as proxies, not independent constructs.")
    evaluate("Cognitive strain survey (benefit/cost -> regret)", d, num, cat, "regret")
    tipping(d)


def tipping(d):
    z = d.copy()
    z["benefit_c"], z["cost_c"] = z["benefit"] - z["benefit"].mean(), z["cost"] - z["cost"].mean()
    fit = smf.ols("regret ~ benefit_c * cost_c + ease + reliance + careful + experience",
                  data=z).fit(cov_type="HC3")
    print("\nBENEFIT x COST interaction (OLS, robust SEs):")
    print(fit.summary().tables[1])
    b, V = fit.params, fit.cov_params()
    sd = z["benefit_c"].std()
    print("Effect of +1 point of 'cost' on regret, at different benefit levels:")
    for lab, k in [("-1 SD", -1), ("mean", 0), ("+1 SD", 1)]:
        slope = b["cost_c"] + b["benefit_c:cost_c"] * k * sd
        var = (V.loc["cost_c", "cost_c"] + (k * sd) ** 2 * V.loc["benefit_c:cost_c", "benefit_c:cost_c"]
               + 2 * k * sd * V.loc["cost_c", "benefit_c:cost_c"])
        se = np.sqrt(var)
        print(f"  benefit {lab:<5}: {slope:+.3f}  (95% CI {slope - 1.96 * se:+.3f} to {slope + 1.96 * se:+.3f})")

    gb = np.linspace(z["benefit"].quantile(.05), z["benefit"].quantile(.95), 40)
    gc = np.linspace(z["cost"].quantile(.05), z["cost"].quantile(.95), 40)
    BB, CC = np.meshgrid(gb, gc)
    grid = pd.DataFrame({"benefit_c": BB.ravel() - z["benefit"].mean(), "cost_c": CC.ravel() - z["cost"].mean(),
                         "ease": z["ease"].mean(), "reliance": z["reliance"].mean(),
                         "careful": z["careful"].mean(), "experience": z["experience"].mean()})
    pred = fit.predict(grid).to_numpy().reshape(BB.shape)
    plt.figure(figsize=(7, 5.5))
    im = plt.contourf(BB, CC, pred, levels=14, cmap="viridis")
    plt.colorbar(im, label="Predicted regret (1-7)")
    if pred.min() < 4 < pred.max():
        plt.contour(BB, CC, pred, levels=[4], colors="white", linestyles="--")
    plt.xlabel("Perceived benefit of AI (1-7)")
    plt.ylabel("Awareness/fear of weaker independent thinking (1-7)")
    plt.title("Association, not causation. Dashed line = scale midpoint (4),\na reference level, not a discovered threshold", fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS, "tipping_cognitive_strain.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


# ------------------------------------------------------------------ cognitive dependence
def run_dependence():
    b = pd.read_csv(first("*AI_Cognitive_Dependence*.csv"))
    lik = {"Strongly Disagree": 1, "Disagree": 2, "Neutral": 3, "Agree": 4, "Strongly Agree": 5}
    cols = [c for c in b.columns if " [" in c]
    for c in cols:
        b[c] = b[c].astype(str).str.strip().map(lik)

    def grp(g):
        return [c for c in cols if c.startswith(g + " [")]

    usage_items = [c for c in grp("AI Usage") if "depend on AI tools" not in c]  # avoids near-duplicate of outcome
    b["usage"] = b[usage_items].mean(axis=1)
    b["dependence"] = b[grp("Cognitive Dependence")].mean(axis=1)
    b["trust"] = b[grp("Trust in AI")].mean(axis=1)
    b["decision"] = b[grp("Academic Decision-Making")].mean(axis=1)
    num = ["usage", "trust", "decision", "Age"]
    cat = ["Gender", "University Type", "Academic Level", "Department/Discipline"]
    d = b[num + cat + ["dependence"]].dropna()
    print(f"\nCOGNITIVE DEPENDENCE: {len(d)} usable rows of {len(b)}")
    print("NOTE: outcome = cognitive dependence index (proxy for loss of autonomy / external reliance).")
    print("Usage, trust and decision items measure closely related behaviour, so R2 may be high because")
    print("of construct overlap in one questionnaire, not because AI use 'causes' dependence.")
    evaluate("Cognitive dependence survey (usage/trust -> dependence)", d, num, cat, "dependence")


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    for fn in (run_career, run_ex, run_dependence):
        try:
            fn()
        except Exception:
            print(f"\n!! {fn.__name__} failed:")
            traceback.print_exc()
    if ROWS:
        out = pd.DataFrame(ROWS)
        out.to_csv(os.path.join(RESULTS, "multi_datasets_metrics.csv"), index=False)
        print("\nSUMMARY\n", out.round(3).to_string(index=False))
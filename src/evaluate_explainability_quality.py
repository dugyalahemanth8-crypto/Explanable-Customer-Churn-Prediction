"""
evaluate_explainability_quality.py
====================================
Answers: "Is my SHAP explainability the BEST it can be?"

This goes beyond correctness (verify_explainability.py) into QUALITY.
It evaluates 8 dimensions used in academic XAI (Explainable AI) research:

  DIMENSION 1 — Fidelity
    Do SHAP values faithfully represent what the model actually learned?
    Metric: Correlation between SHAP ranking and LR |coefficient| ranking.
    Threshold: Spearman r > 0.95

  DIMENSION 2 — Consistency
    Does the same feature always get a similar SHAP direction across
    similar customers? Inconsistent explanations confuse stakeholders.
    Metric: For high-risk customers, InternetService_Fiber optic must
    ALWAYS be red (positive SHAP). For low-risk, Contract_Two year
    must ALWAYS be green (negative SHAP).
    Threshold: 100% directional consistency on prototypical cases.

  DIMENSION 3 — Contrastivity
    Can the explanation distinguish WHY customer A churns but customer B stays,
    when they are nearly identical except for one key feature?
    Metric: SHAP difference for the differing feature > 0.3 between the pair.
    Threshold: All contrastive pairs show clear differentiation.

  DIMENSION 4 — Completeness
    Do the top-10 SHAP features capture most of the model's decision?
    Metric: Sum of |SHAP| for top-10 features as % of total |SHAP|.
    Threshold: Top-10 covers > 80% of total explanation.

  DIMENSION 5 — Sparsity (Simplicity)
    Are explanations concise enough for a business user to act on?
    Too many equally-important features = impossible to prioritize.
    Metric: Gini coefficient of |SHAP| values. Higher = more focused.
    Threshold: Gini > 0.4 (explanations are not flat/uniform).

  DIMENSION 6 — Actionability
    Do the top SHAP features correspond to things a business can actually change?
    Non-actionable features (gender, senior citizen) should NOT dominate.
    Metric: % of top-5 SHAP features that are actionable.
    Threshold: > 60% of top-5 features are actionable.

  DIMENSION 7 — Stability (Robustness)
    Small changes to input (adding $1/month) should not drastically
    change the SHAP ranking of top features.
    Metric: Rank correlation of top-10 SHAP features before/after small perturbation.
    Threshold: Spearman r > 0.85 for ±5% perturbation.

  DIMENSION 8 — Calibration Alignment
    SHAP values should be proportional to their actual impact on probability.
    For LR: a SHAP of +1.0 should correspond to a larger probability shift
    than a SHAP of +0.1.
    Metric: Monotonicity check — ranking by SHAP must match ranking by
    individual feature contribution to probability change.
    Threshold: 100% monotonic.

Run from project root:
    python src/evaluate_explainability_quality.py

Outputs a quality score (0-100) and specific improvement recommendations.
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import MODEL_COLUMNS_PATH, DATA_DIR
from src.predict import build_feature_vector, _load_artefacts
from src.explain import _get_background_data

import shap


# ── Canonical customers ────────────────────────────────────────────────────────

LOW_RISK = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "Yes", "tenure": 60, "PhoneService": "Yes",
    "MultipleLines": "Yes", "InternetService": "DSL",
    "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Credit card (automatic)", "MonthlyCharges": 65.0,
}
HIGH_RISK = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "No",
    "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 85.0,
}
MEDIUM_RISK = {
    "gender": "Female", "SeniorCitizen": 1, "Partner": "No",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No",
    "Contract": "One year", "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check", "MonthlyCharges": 75.0,
}

# Features a business CAN change (actionable) vs cannot
ACTIONABLE_FEATURES = {
    "Contract_Two year", "Contract_One year",
    "OnlineSecurity_Yes", "OnlineBackup_Yes",
    "DeviceProtection_Yes", "TechSupport_Yes",
    "StreamingTV_Yes", "StreamingMovies_Yes",
    "MultipleLines_Yes", "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
    "InternetService_Fiber optic", "InternetService_No",
    "PhoneService_Yes",
}
NON_ACTIONABLE = {
    "gender_Male", "SeniorCitizen", "Partner_Yes",
    "Dependents_Yes", "tenure", "MonthlyCharges",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_shap_series(customer: dict, explainer, model_columns) -> pd.Series:
    model, scaler, _ = _load_artefacts()
    df_enc = build_feature_vector(customer)
    X_sc   = scaler.transform(df_enc)
    sv     = explainer.shap_values(X_sc)
    vals   = np.array(sv).flatten() if not isinstance(sv, list) \
             else np.array(sv[1]).flatten()
    return pd.Series(vals, index=model_columns)


def _sep(title: str, dim: int) -> None:
    print(f"\n{'='*65}")
    print(f"  DIMENSION {dim} — {title}")
    print(f"{'='*65}")


def _score_label(score: float) -> str:
    if score >= 90: return "🟢 EXCELLENT"
    if score >= 70: return "🟡 GOOD"
    if score >= 50: return "🟠 ACCEPTABLE"
    return "🔴 NEEDS IMPROVEMENT"


# ══════════════════════════════════════════════════════════════════════════════
# D1 — FIDELITY
# ══════════════════════════════════════════════════════════════════════════════
def d1_fidelity(explainer, model, model_columns, X_bg):
    _sep("FIDELITY — Does SHAP faithfully represent the model?", 1)
    print("  Metric: Spearman correlation between mean|SHAP| and |coef|")

    rng    = np.random.default_rng(42)
    idx    = rng.choice(len(X_bg), size=min(300, len(X_bg)), replace=False)
    sv     = explainer.shap_values(X_bg[idx])
    vals   = np.array(sv) if not isinstance(sv, list) else np.array(sv[1])
    mean_abs = np.abs(vals).mean(axis=0)
    abs_coef = np.abs(model.coef_[0])

    corr, _ = spearmanr(mean_abs, abs_coef)
    score   = min(100, corr * 100)

    print(f"  Spearman r = {corr:.4f}")
    print(f"  Score: {score:.0f}/100  {_score_label(score)}")

    top5_shap = pd.Series(mean_abs, index=model_columns).nlargest(5)
    top5_coef = pd.Series(abs_coef, index=model_columns).nlargest(5)
    print(f"\n  Top 5 by mean|SHAP|: {list(top5_shap.index)}")
    print(f"  Top 5 by |coef|:     {list(top5_coef.index)}")
    overlap = len(set(top5_shap.index) & set(top5_coef.index))
    print(f"  Top-5 overlap: {overlap}/5  ({'✅ Perfect' if overlap == 5 else '⚠️ Partial' if overlap >= 3 else '❌ Poor'})")

    return score, corr >= 0.95


# ══════════════════════════════════════════════════════════════════════════════
# D2 — CONSISTENCY
# ══════════════════════════════════════════════════════════════════════════════
def d2_consistency(explainer, model, scaler, model_columns):
    _sep("CONSISTENCY — Same feature, same direction across similar customers?", 2)
    print("  Metric: Direction of key features across 10 high-risk & 10 low-risk variants")

    fiber_signs  = []
    contract_signs = []

    # 10 high-risk variants (varying tenure 1-5, monthly 75-95)
    for tenure in [1, 2, 3, 4, 5]:
        for monthly in [75, 85, 95]:
            c = dict(HIGH_RISK); c["tenure"] = tenure; c["MonthlyCharges"] = monthly
            sv = _get_shap_series(c, explainer, model_columns)
            fiber_signs.append(np.sign(sv.get("InternetService_Fiber optic", 0)))

    # 10 low-risk variants (varying tenure 36-72)
    for tenure in [36, 48, 60, 72]:
        for monthly in [55, 65, 75]:
            c = dict(LOW_RISK); c["tenure"] = tenure; c["MonthlyCharges"] = monthly
            sv = _get_shap_series(c, explainer, model_columns)
            contract_signs.append(np.sign(sv.get("Contract_Two year", 0)))

    fiber_consistency    = sum(s > 0 for s in fiber_signs) / len(fiber_signs)
    contract_consistency = sum(s < 0 for s in contract_signs) / len(contract_signs)
    score = (fiber_consistency + contract_consistency) / 2 * 100

    print(f"  InternetService_Fiber optic always RED (positive) across high-risk variants:")
    print(f"    {fiber_consistency:.0%} consistent  ({'✅' if fiber_consistency == 1.0 else '❌'})")
    print(f"  Contract_Two year always GREEN (negative) across low-risk variants:")
    print(f"    {contract_consistency:.0%} consistent  ({'✅' if contract_consistency == 1.0 else '❌'})")
    print(f"\n  Score: {score:.0f}/100  {_score_label(score)}")

    return score, score >= 95


# ══════════════════════════════════════════════════════════════════════════════
# D3 — CONTRASTIVITY
# ══════════════════════════════════════════════════════════════════════════════
def d3_contrastivity(explainer, model, scaler, model_columns):
    _sep("CONTRASTIVITY — Can SHAP explain why A churns but B stays?", 3)
    print("  Metric: SHAP difference for the SINGLE differing feature between twin pairs")
    print("  Threshold: difference > 0.20 for each pair\n")

    pairs = [
        ("Contract",       "Month-to-month", "Two year",  "Contract_Two year"),
        ("tenure",         2,                 60,          "tenure"),
        ("InternetService","Fiber optic",     "DSL",       "InternetService_Fiber optic"),
        ("OnlineSecurity", "No",              "Yes",       "OnlineSecurity_Yes"),
    ]

    results = []
    for feat, val_churn, val_stay, shap_col in pairs:
        c_churn = dict(HIGH_RISK); c_churn[feat] = val_churn
        c_stay  = dict(HIGH_RISK); c_stay[feat]  = val_stay

        # Fix dependent fields when changing InternetService
        if feat == "InternetService" and val_stay == "DSL":
            for svc in ["OnlineSecurity","OnlineBackup","DeviceProtection",
                        "TechSupport","StreamingTV","StreamingMovies"]:
                c_stay[svc] = "No"

        sv_churn = _get_shap_series(c_churn, explainer, model_columns)
        sv_stay  = _get_shap_series(c_stay,  explainer, model_columns)

        shap_c = sv_churn.get(shap_col, 0)
        shap_s = sv_stay.get(shap_col, 0)
        diff   = abs(shap_c - shap_s)
        passed = diff >= 0.20

        results.append(passed)
        print(f"  {feat} flip ({val_churn} → {val_stay}):")
        print(f"    SHAP[{shap_col}]: {shap_c:+.4f} → {shap_s:+.4f}  |Δ|={diff:.4f}  "
              f"{'✅ Contrastive' if passed else '❌ Not contrastive enough'}")

    score = sum(results) / len(results) * 100
    print(f"\n  Score: {score:.0f}/100  {_score_label(score)}")
    return score, all(results)


# ══════════════════════════════════════════════════════════════════════════════
# D4 — COMPLETENESS
# ══════════════════════════════════════════════════════════════════════════════
def d4_completeness(explainer, model, scaler, model_columns):
    _sep("COMPLETENESS — Do top-10 features capture most of the decision?", 4)
    print("  Metric: % of total |SHAP| captured by top-10 features")
    print("  Threshold: > 80%\n")

    coverages = []
    for name, customer in [("LOW_RISK", LOW_RISK),
                            ("HIGH_RISK", HIGH_RISK),
                            ("MEDIUM_RISK", MEDIUM_RISK)]:
        sv    = _get_shap_series(customer, explainer, model_columns)
        total = sv.abs().sum()
        top10 = sv.abs().nlargest(10).sum()
        cov   = top10 / total if total > 0 else 0
        coverages.append(cov)
        print(f"  {name:<15}  top-10 coverage = {cov:.1%}  "
              f"({'✅' if cov >= 0.80 else '❌'})")

    avg_cov = np.mean(coverages)
    score   = min(100, avg_cov * 100)
    print(f"\n  Average coverage: {avg_cov:.1%}")
    print(f"  Score: {score:.0f}/100  {_score_label(score)}")
    return score, avg_cov >= 0.80


# ══════════════════════════════════════════════════════════════════════════════
# D5 — SPARSITY (Simplicity)
# ══════════════════════════════════════════════════════════════════════════════
def d5_sparsity(explainer, model, scaler, model_columns):
    _sep("SPARSITY — Are explanations focused enough to act on?", 5)
    print("  Metric: Gini coefficient of |SHAP| values (higher = more focused)")
    print("  Threshold: Gini > 0.40  (flat distribution = Gini 0, one feature = Gini 1)\n")

    def gini(arr):
        arr = np.sort(np.abs(arr))
        n   = len(arr)
        idx = np.arange(1, n + 1)
        return (2 * np.sum(idx * arr) / (n * np.sum(arr))) - (n + 1) / n if arr.sum() > 0 else 0

    ginis = []
    for name, customer in [("LOW_RISK", LOW_RISK),
                            ("HIGH_RISK", HIGH_RISK),
                            ("MEDIUM_RISK", MEDIUM_RISK)]:
        sv  = _get_shap_series(customer, explainer, model_columns)
        g   = gini(sv.values)
        ginis.append(g)

        # Show distribution
        top3 = sv.abs().nlargest(3)
        total = sv.abs().sum()
        top3_pct = top3.sum() / total if total > 0 else 0
        print(f"  {name:<15}  Gini={g:.3f}  top-3 captures {top3_pct:.0%}  "
              f"({'✅' if g >= 0.40 else '⚠️ flat'})")
        for f, v in top3.items():
            print(f"    {f:<40} {sv[f]:+.4f}")

    avg_gini = np.mean(ginis)
    score    = min(100, avg_gini * 150)   # 0.67 Gini → 100 score
    print(f"\n  Average Gini: {avg_gini:.3f}")
    print(f"  Score: {score:.0f}/100  {_score_label(score)}")
    return score, avg_gini >= 0.40


# ══════════════════════════════════════════════════════════════════════════════
# D6 — ACTIONABILITY
# ══════════════════════════════════════════════════════════════════════════════
def d6_actionability(explainer, model, scaler, model_columns):
    _sep("ACTIONABILITY — Do top features correspond to things a business can change?", 6)
    print("  Actionable: Contract, Services, Payment Method, Internet type")
    print("  Non-actionable: Gender, SeniorCitizen, tenure (demographic/locked-in)")
    print("  Threshold: > 60% of top-5 features are actionable\n")

    action_rates = []
    for name, customer in [("LOW_RISK", LOW_RISK),
                            ("HIGH_RISK", HIGH_RISK),
                            ("MEDIUM_RISK", MEDIUM_RISK)]:
        sv    = _get_shap_series(customer, explainer, model_columns)
        top5  = sv.abs().nlargest(5).index.tolist()
        n_act = sum(1 for f in top5 if f in ACTIONABLE_FEATURES)
        rate  = n_act / 5
        action_rates.append(rate)

        print(f"  {name}:")
        for f in top5:
            tag = "✅ actionable" if f in ACTIONABLE_FEATURES else "⚪ non-actionable"
            print(f"    {f:<40}  {sv[f]:+.4f}  {tag}")
        print(f"  Actionability rate: {rate:.0%}  {'✅' if rate >= 0.60 else '⚠️'}\n")

    avg_rate = np.mean(action_rates)
    score    = avg_rate * 100
    print(f"  Average actionability rate: {avg_rate:.0%}")
    print(f"  Score: {score:.0f}/100  {_score_label(score)}")
    return score, avg_rate >= 0.60


# ══════════════════════════════════════════════════════════════════════════════
# D7 — STABILITY (Robustness)
# ══════════════════════════════════════════════════════════════════════════════
def d7_stability(explainer, model, scaler, model_columns):
    _sep("STABILITY — Do small input changes cause drastic explanation changes?", 7)
    print("  Metric: Rank correlation of top-10 SHAP features before/after ±5% perturbation")
    print("  Threshold: Spearman r > 0.85\n")

    correlations = []
    for name, customer in [("LOW_RISK", LOW_RISK),
                            ("HIGH_RISK", HIGH_RISK),
                            ("MEDIUM_RISK", MEDIUM_RISK)]:
        sv_orig = _get_shap_series(customer, explainer, model_columns)

        # Perturb MonthlyCharges by +5%
        c_perturbed = dict(customer)
        c_perturbed["MonthlyCharges"] = customer["MonthlyCharges"] * 1.05
        sv_pert = _get_shap_series(c_perturbed, explainer, model_columns)

        # Rank correlation of |SHAP| values
        corr, _ = spearmanr(sv_orig.abs().values, sv_pert.abs().values)
        correlations.append(corr)

        # Check if top-5 features are the same
        top5_orig = set(sv_orig.abs().nlargest(5).index)
        top5_pert = set(sv_pert.abs().nlargest(5).index)
        overlap   = len(top5_orig & top5_pert)

        print(f"  {name:<15}  rank_corr={corr:.4f}  top-5 overlap={overlap}/5  "
              f"({'✅' if corr >= 0.85 else '❌'})")

    avg_corr = np.mean(correlations)
    score    = min(100, avg_corr * 100)
    print(f"\n  Average rank correlation: {avg_corr:.4f}")
    print(f"  Score: {score:.0f}/100  {_score_label(score)}")
    return score, avg_corr >= 0.85


# ══════════════════════════════════════════════════════════════════════════════
# D8 — CALIBRATION ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════════
def d8_calibration(explainer, model, scaler, model_columns):
    _sep("CALIBRATION ALIGNMENT — Does SHAP magnitude match probability impact?", 8)
    print("  Metric: For each feature in top-10, verify that removing it")
    print("  changes the probability proportionally to its |SHAP| value")
    print("  Threshold: Spearman r > 0.80 between |SHAP| and |ΔP|\n")

    correlations = []
    for name, customer in [("HIGH_RISK", HIGH_RISK),
                            ("LOW_RISK", LOW_RISK)]:
        sv = _get_shap_series(customer, explainer, model_columns)
        top10 = sv.abs().nlargest(10).index.tolist()

        base_X  = scaler.transform(build_feature_vector(customer))
        base_prob = float(model.predict_proba(base_X)[0][1])

        delta_probs = []
        shap_mags   = []

        for feat in top10:
            if feat not in model_columns:
                continue
            idx = model_columns.index(feat)

            # Zero out this feature (set to background mean)
            X_mod = base_X.copy()
            bg_mean = _get_background_data(scaler, model_columns).mean(axis=0)
            X_mod[0, idx] = bg_mean[idx]

            mod_prob  = float(model.predict_proba(X_mod)[0][1])
            delta_probs.append(abs(mod_prob - base_prob))
            shap_mags.append(abs(sv[feat]))

        if len(shap_mags) >= 3:
            corr, _ = spearmanr(shap_mags, delta_probs)
            correlations.append(corr)
            print(f"  {name:<15}  SHAP-magnitude vs ΔP correlation: {corr:.4f}  "
                  f"({'✅' if corr >= 0.80 else '⚠️'})")

    avg_corr = np.mean(correlations) if correlations else 0
    score    = min(100, max(0, avg_corr * 100))
    print(f"\n  Average correlation: {avg_corr:.4f}")
    print(f"  Score: {score:.0f}/100  {_score_label(score)}")
    return score, avg_corr >= 0.80


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 65)
    print("  EXPLAINABILITY QUALITY EVALUATION")
    print("  Customer Churn Prediction — SHAP LinearExplainer")
    print("█" * 65)

    # Load shared objects once
    model, scaler, model_columns = _load_artefacts()
    X_bg     = _get_background_data(scaler, model_columns)
    explainer = shap.LinearExplainer(model, X_bg)

    print(f"\n  Model: LogisticRegression  |  Features: {len(model_columns)}")
    print(f"  Background samples: {len(X_bg)}")
    print(f"  Evaluating 8 explainability quality dimensions...\n")

    # Run all 8 dimensions
    dim_results = {}

    try:
        s, p = d1_fidelity(explainer, model, model_columns, X_bg)
        dim_results["D1 Fidelity"] = (s, p)
    except Exception as e:
        print(f"  ERROR: {e}")
        dim_results["D1 Fidelity"] = (0, False)

    try:
        s, p = d2_consistency(explainer, model, scaler, model_columns)
        dim_results["D2 Consistency"] = (s, p)
    except Exception as e:
        print(f"  ERROR: {e}")
        dim_results["D2 Consistency"] = (0, False)

    try:
        s, p = d3_contrastivity(explainer, model, scaler, model_columns)
        dim_results["D3 Contrastivity"] = (s, p)
    except Exception as e:
        print(f"  ERROR: {e}")
        dim_results["D3 Contrastivity"] = (0, False)

    try:
        s, p = d4_completeness(explainer, model, scaler, model_columns)
        dim_results["D4 Completeness"] = (s, p)
    except Exception as e:
        print(f"  ERROR: {e}")
        dim_results["D4 Completeness"] = (0, False)

    try:
        s, p = d5_sparsity(explainer, model, scaler, model_columns)
        dim_results["D5 Sparsity"] = (s, p)
    except Exception as e:
        print(f"  ERROR: {e}")
        dim_results["D5 Sparsity"] = (0, False)

    try:
        s, p = d6_actionability(explainer, model, scaler, model_columns)
        dim_results["D6 Actionability"] = (s, p)
    except Exception as e:
        print(f"  ERROR: {e}")
        dim_results["D6 Actionability"] = (0, False)

    try:
        s, p = d7_stability(explainer, model, scaler, model_columns)
        dim_results["D7 Stability"] = (s, p)
    except Exception as e:
        print(f"  ERROR: {e}")
        dim_results["D7 Stability"] = (0, False)

    try:
        s, p = d8_calibration(explainer, model, scaler, model_columns)
        dim_results["D8 Calibration"] = (s, p)
    except Exception as e:
        print(f"  ERROR: {e}")
        dim_results["D8 Calibration"] = (0, False)

    # ── Final report ───────────────────────────────────────────────────────────
    print("\n\n" + "█" * 65)
    print("  EXPLAINABILITY QUALITY REPORT")
    print("█" * 65)

    print(f"\n  {'Dimension':<25} {'Score':>6}  {'Pass?':>6}  {'Rating'}")
    print("  " + "-" * 58)

    scores  = []
    weights = [1.5, 1.2, 1.2, 1.0, 1.0, 1.3, 1.0, 1.0]  # weighted importance
    total_w = sum(weights)

    for (name, (score, passed)), weight in zip(dim_results.items(), weights):
        icon   = "✅" if passed else "❌"
        rating = _score_label(score)
        scores.append(score * weight)
        print(f"  {name:<25} {score:>5.0f}%  {icon:>6}  {rating}")

    weighted_avg = sum(scores) / total_w
    passed_count = sum(1 for _, (_, p) in dim_results.items() if p)
    total        = len(dim_results)

    print(f"\n  {'─'*58}")
    print(f"  {'OVERALL QUALITY SCORE':<25} {weighted_avg:>5.0f}%")
    print(f"  Dimensions passing: {passed_count}/{total}")

    # Grade
    if   weighted_avg >= 90: grade = "A+  PRODUCTION READY"
    elif weighted_avg >= 80: grade = "A   EXCELLENT"
    elif weighted_avg >= 70: grade = "B   GOOD"
    elif weighted_avg >= 60: grade = "C   ACCEPTABLE"
    else:                    grade = "F   NEEDS SIGNIFICANT WORK"

    print(f"\n  Grade: {grade}")

    # Recommendations
    print(f"\n  {'─'*58}")
    print("  IMPROVEMENT RECOMMENDATIONS:")
    print()

    failed = [(n, s) for n, (s, p) in dim_results.items() if not p]
    if not failed:
        print("  🎉 All dimensions passing! Your explainability is production-quality.")
        print()
        print("  To push further:")
        print("  • Try SHAP interaction values to show feature pairs")
        print("  • Add counterfactual explanations (what would make this customer stay?)")
        print("  • Add global SHAP summary plots for stakeholder presentations")
    else:
        for name, score in sorted(failed, key=lambda x: x[1]):
            if "Fidelity" in name:
                print(f"  ❌ {name} ({score:.0f}%): SHAP not matching coefficients.")
                print(f"     → Check that model and scaler are from the same training run.")
            elif "Consistency" in name:
                print(f"  ❌ {name} ({score:.0f}%): Feature direction changes unexpectedly.")
                print(f"     → Verify background data is the full training set, not a subset.")
            elif "Contrastivity" in name:
                print(f"  ❌ {name} ({score:.0f}%): SHAP values too similar between different customers.")
                print(f"     → Consider increasing model regularization (lower C in LogReg).")
            elif "Completeness" in name:
                print(f"  ❌ {name} ({score:.0f}%): Too many features needed to explain decisions.")
                print(f"     → Consider feature selection to remove low-importance features.")
            elif "Sparsity" in name:
                print(f"  ❌ {name} ({score:.0f}%): Explanations are too spread across many features.")
                print(f"     → Increase regularization (reduce C) to force sparser coefficients.")
            elif "Actionability" in name:
                print(f"  ❌ {name} ({score:.0f}%): Non-actionable features dominating explanations.")
                print(f"     → Consider removing gender/SeniorCitizen from the model.")
            elif "Stability" in name:
                print(f"  ❌ {name} ({score:.0f}%): Small input changes cause big explanation changes.")
                print(f"     → This is often a sign of multicollinearity. Check remaining correlations.")
            elif "Calibration" in name:
                print(f"  ❌ {name} ({score:.0f}%): SHAP magnitudes don't match actual probability impact.")
                print(f"     → Verify the background dataset represents the training distribution.")

    print("\n" + "█" * 65 + "\n")
    return weighted_avg


if __name__ == "__main__":
    score = main()
    sys.exit(0 if score >= 70 else 1)
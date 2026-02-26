"""
streamlit_app.py — Customer Churn Prediction with SHAP Explainability
Run:  streamlit run app/streamlit_app.py
"""
import sys
import warnings
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# streamlit_app.py lives in  <root>/app/
# src/ lives in              <root>/src/
# So we need to add <root> (the parent of app/) to sys.path
_APP_DIR     = Path(__file__).resolve().parent          # <root>/app/
_PROJECT_ROOT = _APP_DIR.parent                          # <root>/
sys.path.insert(0, str(_PROJECT_ROOT))

warnings.filterwarnings("ignore")

from src.predict  import predict_churn
from src.explain  import explain_single_prediction, get_explanation_figure
from src.config   import (
    FEATURE_IMPORTANCE_PATH, FEATURE_IMPORTANCE_GLOBAL_PATH, SHAP_IMPORTANCE_PATH
)

# ── Page Setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 12px; padding: 16px 20px;
        border-left: 4px solid #2b6cb0; margin-bottom: 12px;
    }
    .churn-box {
        background: #fff5f5; border: 2px solid #fc8181; border-radius: 12px;
        padding: 20px; text-align: center;
    }
    .stay-box {
        background: #f0fff4; border: 2px solid #68d391; border-radius: 12px;
        padding: 20px; text-align: center;
    }
    h1 { font-size: 1.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Title ──────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction System")
st.markdown("Predict churn probability **and understand why** using SHAP explainability.")
st.markdown("---")

# ── Sidebar — Customer Inputs ──────────────────────────────────────────────────
st.sidebar.header("🔍 Customer Information")

st.sidebar.subheader("Demographics")
gender         = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1],
                                      format_func=lambda x: "Yes" if x == 1 else "No")
partner        = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents     = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])

st.sidebar.subheader("Account Information")
tenure        = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract      = st.sidebar.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)
paperless     = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment       = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)

st.sidebar.subheader("Services")
phone_svc = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multi_lines = (
    st.sidebar.selectbox("Multiple Lines", ["Yes", "No"])
    if phone_svc == "Yes" else "No phone service"
)
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

if internet != "No":
    online_sec  = st.sidebar.selectbox("Online Security",    ["Yes", "No"])
    online_bak  = st.sidebar.selectbox("Online Backup",      ["Yes", "No"])
    dev_protect = st.sidebar.selectbox("Device Protection",  ["Yes", "No"])
    tech_supp   = st.sidebar.selectbox("Tech Support",       ["Yes", "No"])
    stream_tv   = st.sidebar.selectbox("Streaming TV",       ["Yes", "No"])
    stream_mov  = st.sidebar.selectbox("Streaming Movies",   ["Yes", "No"])
else:
    online_sec = online_bak = dev_protect = tech_supp = \
        stream_tv = stream_mov = "No internet service"

st.sidebar.subheader("Charges")
monthly_charges = st.sidebar.number_input(
    "Monthly Charges ($)", 0.0, 200.0, 70.0, step=5.0
)
total_charges = st.sidebar.number_input(
    "Total Charges ($)", 0.0, 10000.0,
    float(monthly_charges * max(tenure, 1)), step=50.0
)

# ── Assemble customer dict ─────────────────────────────────────────────────────
customer_data = {
    "gender":           gender,
    "SeniorCitizen":    senior_citizen,
    "Partner":          partner,
    "Dependents":       dependents,
    "tenure":           tenure,
    "PhoneService":     phone_svc,
    "MultipleLines":    multi_lines,
    "InternetService":  internet,
    "OnlineSecurity":   online_sec,
    "OnlineBackup":     online_bak,
    "DeviceProtection": dev_protect,
    "TechSupport":      tech_supp,
    "StreamingTV":      stream_tv,
    "StreamingMovies":  stream_mov,
    "Contract":         contract,
    "PaperlessBilling": paperless,
    "PaymentMethod":    payment,
    "MonthlyCharges":   monthly_charges,
    "TotalCharges":     total_charges,
}

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 Predict Churn", type="primary")

# ── Main content ───────────────────────────────────────────────────────────────
if predict_button:
    with st.spinner("Analysing customer data and computing SHAP values …"):
        try:
            # ── Prediction ─────────────────────────────────────────────────────
            prediction, probability, model = predict_churn(customer_data)

            # ── SHAP explanation ───────────────────────────────────────────────
            explanation_df = explain_single_prediction(
                customer_data, top_n=10,
                customer_label=f"tenure={tenure}mo, {contract}"
            )

        except Exception as exc:
            st.error(f"❌ Error: {exc}")
            with st.expander("Full traceback"):
                import traceback
                st.code(traceback.format_exc())
            st.stop()

    # ── ─────────────────────────────────────────────────────────────────────────
    # RESULTS — two-column layout
    # ── ─────────────────────────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    # ── LEFT: Prediction summary ───────────────────────────────────────────────
    with left:
        st.markdown("### 🎯 Prediction Result")

        if prediction == 1:
            st.markdown(
                f"<div class='churn-box'>"
                f"<h2 style='color:#c53030;margin:0'>🔴 WILL CHURN</h2>"
                f"<p style='font-size:2rem;font-weight:800;color:#c53030;margin:8px 0'>"
                f"{probability:.1%}</p>"
                f"<p style='color:#718096;margin:0'>Churn Probability</p>"
                f"</div>", unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='stay-box'>"
                f"<h2 style='color:#276749;margin:0'>🟢 WILL STAY</h2>"
                f"<p style='font-size:2rem;font-weight:800;color:#276749;margin:8px 0'>"
                f"{probability:.1%}</p>"
                f"<p style='color:#718096;margin:0'>Churn Probability</p>"
                f"</div>", unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(probability, text=f"Churn probability gauge: {probability:.1%}")

        # Risk level badge
        if probability > 0.7:
            risk_label, risk_color = "🔴 HIGH RISK", "#c53030"
        elif probability > 0.4:
            risk_label, risk_color = "🟡 MEDIUM RISK", "#b7791f"
        else:
            risk_label, risk_color = "🟢 LOW RISK", "#276749"
        st.markdown(
            f"<p style='font-size:1.1rem;font-weight:700;color:{risk_color}'>"
            f"Risk Level: {risk_label}</p>",
            unsafe_allow_html=True
        )

        # ── Key metrics ────────────────────────────────────────────────────────
        st.markdown("#### Customer Snapshot")
        m1, m2, m3 = st.columns(3)
        m1.metric("Tenure",    f"{tenure} mo")
        m2.metric("Monthly",   f"${monthly_charges:.0f}")
        m3.metric("Contract",  contract.replace("-to-", "-"))

        # ── Recommendations ────────────────────────────────────────────────────
        st.markdown("---")
        if prediction == 1 or probability > 0.5:
            st.error("⚠️ **HIGH CHURN RISK — Immediate action recommended!**")
            st.markdown("**Retention strategies:**")
            st.markdown(
                "- 🎁 Offer loyalty discounts or promotions\n"
                "- 📞 Schedule a personal retention call\n"
                "- 📝 Propose contract upgrade with incentives\n"
                "- 🎯 Provide personalised service bundles"
            )
            # Specific risk factors
            factors = []
            if tenure < 12:              factors.append("⚠️ New customer (high baseline churn risk)")
            if contract == "Month-to-month": factors.append("⚠️ Month-to-month contract (no commitment)")
            if payment == "Electronic check": factors.append("⚠️ Electronic check payment (correlated with churn)")
            if internet == "Fiber optic": factors.append("⚠️ Fiber optic service (price-sensitive segment)")
            if monthly_charges > 80:     factors.append(f"⚠️ High monthly charges (${monthly_charges:.0f})")
            if online_sec == "No" and internet != "No": factors.append("⚠️ No online security add-on")
            if tech_supp == "No" and internet != "No":  factors.append("⚠️ No tech support add-on")
            if factors:
                st.markdown("**Detected risk factors:**")
                for f in factors:
                    st.markdown(f"- {f}")
        else:
            st.success("✅ **LOW CHURN RISK — Customer likely to stay!**")
            st.markdown("**Engagement strategies:**")
            st.markdown(
                "- 🎯 Consider upselling premium services\n"
                "- 💎 Reward loyalty with exclusive benefits\n"
                "- 📊 Request feedback and testimonials\n"
                "- 🤝 Maintain excellent service quality"
            )

    # ── RIGHT: SHAP Explanation ────────────────────────────────────────────────
    with right:
        st.markdown("### 🔍 Why This Prediction? (SHAP)")
        st.caption(
            "SHAP values show each feature's contribution to the churn probability. "
            "**Red bars → increase** churn risk. **Green bars → decrease** churn risk."
        )

        # ── SHAP bar chart ─────────────────────────────────────────────────────
        title = (
            f"Top Feature Contributions\n"
            f"Prediction: {'CHURN' if prediction == 1 else 'STAY'} "
            f"({probability:.1%} probability)"
        )
        fig = get_explanation_figure(explanation_df, title=title)
        st.pyplot(fig)
        plt.close(fig)

        # ── SHAP table — custom HTML (works on both light and dark themes) ────────
        st.markdown("#### Feature Impact Table")
        display_df = explanation_df[["Feature", "SHAP_Value", "Impact"]].copy()
        display_df["SHAP_Value"] = pd.to_numeric(display_df["SHAP_Value"], errors="coerce").round(4)

        rows_html = ""
        for _, row in display_df.iterrows():
            is_churn  = float(row["SHAP_Value"]) > 0
            bg_color  = "rgba(229,62,62,0.15)"   if is_churn else "rgba(56,161,105,0.15)"
            dot_color = "#e53e3e"                  if is_churn else "#38a169"
            arrow     = "▲"                        if is_churn else "▼"
            rows_html += f"""
            <tr style='background:{bg_color};'>
              <td style='padding:8px 12px;color:#e2e8f0;font-size:.9rem;'>{row['Feature']}</td>
              <td style='padding:8px 12px;color:#e2e8f0;font-size:.9rem;text-align:center;font-weight:600;'>{row['SHAP_Value']:.4f}</td>
              <td style='padding:8px 12px;color:{dot_color};font-size:.85rem;font-weight:600;'>{arrow} {row['Impact']}</td>
            </tr>"""

        table_html = f"""
        <table style='width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;'>
          <thead>
            <tr style='background:rgba(255,255,255,0.08);'>
              <th style='padding:10px 12px;text-align:left;color:#a0aec0;font-size:.8rem;text-transform:uppercase;letter-spacing:.5px;'>Feature</th>
              <th style='padding:10px 12px;text-align:center;color:#a0aec0;font-size:.8rem;text-transform:uppercase;letter-spacing:.5px;'>SHAP Value</th>
              <th style='padding:10px 12px;text-align:left;color:#a0aec0;font-size:.8rem;text-transform:uppercase;letter-spacing:.5px;'>Impact</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>"""

        st.markdown(table_html, unsafe_allow_html=True)
        st.caption(
            "💡 SHAP value magnitude shows how strongly each feature pushed the model "
            "toward or away from predicting churn for **this specific customer**."
        )

    # ── Full-width extra analysis ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Complete Customer Profile")
    profile_cols = st.columns(4)
    profile_items = [
        ("Gender", gender), ("Senior Citizen", "Yes" if senior_citizen else "No"),
        ("Partner", partner), ("Dependents", dependents),
        ("Phone Service", phone_svc), ("Multiple Lines", multi_lines),
        ("Internet Service", internet), ("Online Security", online_sec),
        ("Online Backup", online_bak), ("Device Protection", dev_protect),
        ("Tech Support", tech_supp), ("Streaming TV", stream_tv),
        ("Streaming Movies", stream_mov), ("Paperless Billing", paperless),
        ("Payment Method", payment), ("Total Charges", f"${total_charges:.2f}"),
    ]
    for i, (k, v) in enumerate(profile_items):
        profile_cols[i % 4].markdown(f"**{k}:** {v}")


# ── Home screen (before prediction) ───────────────────────────────────────────
else:
    st.info("👈 Fill in customer information in the sidebar and click **Predict Churn** to get started!")

    st.markdown("### 📊 Model Feature Importance")

    tab1, tab2, tab3 = st.tabs(
        ["🏆 Top 10 Features", "🌐 Global Top 15", "🔬 SHAP Importance"]
    )
    with tab1:
        if FEATURE_IMPORTANCE_PATH.exists():
            st.image(FEATURE_IMPORTANCE_PATH.read_bytes(), width=700)
        else:
            st.warning("feature_importance.png not found — run `python src/train.py` first.")

    with tab2:
        if FEATURE_IMPORTANCE_GLOBAL_PATH.exists():
            st.image(FEATURE_IMPORTANCE_GLOBAL_PATH.read_bytes(), width=700)
        else:
            st.warning("feature_importance_global.png not found — run `python src/train.py` first.")

    with tab3:
        if SHAP_IMPORTANCE_PATH.exists():
            st.image(SHAP_IMPORTANCE_PATH.read_bytes(), width=700)
        else:
            st.warning("shap_importance.png not found — run `python src/explain.py` first.")

    st.markdown("---")
    st.markdown("### ℹ️ About This System")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Model:** Logistic Regression (30 features)

**Encoding:** One-hot encoding with `drop_first=True`

**Features analysed:**
- Demographics (age, gender, family)
- Account info (tenure, contract, payment)
- Services (internet, phone, add-ons)
- Charges (monthly and total)
        """)
    with col2:
        st.markdown("""
**Explainability:** SHAP (SHapley Additive exPlanations)

**SHAP method:** `LinearExplainer` — exact for linear models

**Key findings from the data:**
- Tenure is the strongest predictor of retention
- Month-to-month contracts have 3× higher churn rate
- Fiber optic customers are more price-sensitive
- Electronic check payment correlates with churn
        """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#a0aec0;font-size:.85rem'>"
    "Customer Churn Prediction System v2.0 · "
    "Logistic Regression + SHAP Explainability · Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
# 📊 Explainable Customer Churn Prediction (XAI)

An end-to-end, production-grade Machine Learning system for the telecommunications industry. This project goes beyond simple binary prediction by implementing a **Verified Explainability Framework** to provide trustworthy, actionable insights for customer retention.

### 🚀 [Click Here to View the Live Demo](https://explanable-customer-churn-prediction-nqpcbyktblkyyasxch82hk.streamlit.app/)

---

## 🚀 Key Innovations & Technical Rigor
Unlike standard "black-box" churn models found in many research papers, this project implements:

* **Multicollinearity Resolution:** Mathematically identified and resolved the "TotalCharges Trap." By dropping redundant features, we ensured SHAP values remain stable and prevent feature importance hijacking.
* **8-Dimension XAI Quality Suite:** A dedicated evaluation framework (`evaluate_explainability_quality.py`) that benchmarks explanations across academic metrics: *Fidelity, Stability, Sparsity, Consistency, Contrastivity, Completeness, Actionability, and Calibration.*
* **Explainer Factory:** A modular backend that dynamically switches between `shap.LinearExplainer` and `shap.TreeExplainer` based on the active model architecture.
* **Production-Ready Inference:** Custom `_manual_encode` logic to handle single-row real-time predictions without the common "missing column" errors found in standard `get_dummies` implementations.

## 🛠️ Tech Stack
- **Core ML:** Scikit-learn (Logistic Regression, Random Forest)
- **Explainability:** SHAP (Shapley Additive Explanations)
- **Interface:** Streamlit (Interactive Dashboard)
- **Data:** Pandas, NumPy, Matplotlib, Seaborn

## 📂 Project Structure
```text
├── app/              # Interactive Streamlit Web Dashboard
├── data/             # Raw dataset, processed arrays, and serialized models
├── src/              # Core Logic (Train, Predict, Explain, Quality Benchmarking)
├── notebooks/        # Exploratory Data Analysis (EDA)
└── requirements.txt  # Project dependencies
```

⚙️ How to Run

Install Dependencies:

```Bash
pip install -r requirements.txt
```
Train & Evaluate:

```Bash
python src/train.py
python src/evaluate_explainability_quality.py
```

Launch the Dashboard:
```Bash
streamlit run app/streamlit_app.py
```

📈 Research Comparison
This project was benchmarked against recent (2022-2026) XAI research in telecom. While many papers focus on raw AUC, this implementation prioritizes Explanation Fidelity, ensuring that the reasons provided for a customer leaving are mathematically sound and business-actionable.


---

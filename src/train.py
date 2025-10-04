import pandas as pd
from catboost import CatBoostClassifier
import shap
from preprocess import load_and_preprocess
from evaluate import evaluate_model

def run_experiments():
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess()
    results = []

    cat = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=5,
        random_state=42,
        verbose=100,
        eval_metric='F1',
        early_stopping_rounds=500
    )
    cat.fit(X_train, y_train, eval_set=(X_test, y_test))

    # Predições
    y_probs = cat.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > 0.5).astype(int)

    # Avaliar modelo
    evaluate_model("CatBoost", y_test, y_pred, y_probs, results)
    df_results = pd.DataFrame(results, columns=["Modelo", "Specificity", "Sensitivity", "Accuracy", "AUC"])

    # SHAP
    explainer = shap.TreeExplainer(cat)
    shap_values = explainer.shap_values(X_test)

    # Gráficos SHAP (Summary + Decision + Waterfall)
    shap.summary_plot(shap_values, X_test, show=False)  # Summary
    shap.decision_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], show=False)  # Decision
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0,:], feature_names=X_test.columns)  # Waterfall

    return df_results, explainer, shap_values, X_test

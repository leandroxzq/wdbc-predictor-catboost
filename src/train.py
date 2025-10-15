import pandas as pd
from catboost import CatBoostClassifier
from sklearn.svm import SVC
import shap
from preprocess import load_and_preprocess
from evaluate import evaluate_model

def run_experiments():
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess()
    results = []

    # ---------------- CatBoost ----------------
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

    # Predições CatBoost
    y_probs_cat = cat.predict_proba(X_test)[:, 1]
    y_pred_cat = (y_probs_cat > 0.5).astype(int)

    # Avaliar CatBoost
    evaluate_model("CatBoost", y_test, y_pred_cat, y_probs_cat, results)

    # ---------------- SVM ----------------
    svm = SVC(
        C=10,                # regularização mais flexível
        kernel='rbf',        # captura não linearidades
        gamma='scale',       # boa escala padrão
        probability=True,    # necessário para AUC
        random_state=42
    )
    svm.fit(X_train, y_train)

    # Predições SVM
    y_probs_svm = svm.predict_proba(X_test)[:, 1]
    y_pred_svm = (y_probs_svm > 0.5).astype(int)

    # Avaliar SVM
    evaluate_model("SVM", y_test, y_pred_svm, y_probs_svm, results)

    # DataFrame com resultados
    df_results = pd.DataFrame(results, columns=["Modelo", "Specificity", "Sensitivity", "Accuracy", "AUC"])

    # ---------------- SHAP CatBoost ----------------
    explainer = shap.TreeExplainer(cat)
    shap_values = explainer.shap_values(X_test)

    # Gráficos SHAP (Summary + Decision + Waterfall)
    shap.summary_plot(shap_values, X_test, show=False)  # Summary
    shap.decision_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], show=False)  # Decision
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0,:], feature_names=X_test.columns)  # Waterfall

    return df_results, explainer, shap_values, X_test

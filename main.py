from src.train import run_experiments

if __name__ == "__main__":
    df_results, explainer, shap_values, X_test = run_experiments()
    print("\nResultados finais:")
    print(df_results)

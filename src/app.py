import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from train import run_experiments 

st.set_page_config(page_title="Breast Cancer CatBoost + SHAP", layout="wide")
st.title("Breast Cancer Wisconsin - CatBoost + SHAP")

# Executar experimento (Roda o treino/avaliação)

df_results, explainer, shap_values, X_test = run_experiments()

# Mostrar métricas

st.subheader("Resultados de Performance")
st.dataframe(df_results)

# SHAP Summary Plot (Gráfico Global)

st.subheader("1. Summary Plot (Interpretabilidade Global)")
st.caption("Importância geral das features")

fig_summary, ax_summary = plt.subplots(figsize=(10, 6))

shap.summary_plot(shap_values, X_test, show=False)

st.pyplot(fig_summary, bbox_inches='tight')
plt.close(fig_summary) # Fecha a figura para liberar memória

# SHAP Decision Plot (Amostra Local)

st.subheader("2. Decision Plot (Interpretabilidade Local - Amostra 0)")

fig_decision, ax_decision = plt.subplots(figsize=(10, 6))

shap.decision_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:], show=False)

st.pyplot(fig_decision, bbox_inches='tight')
plt.close(fig_decision)

# SHAP Waterfall Plot (Amostra Local - Mais Visual)

st.subheader("3. Waterfall Plot (Justificativa da Decisão - Amostra 0)")

fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))

feature_names = X_test.columns.tolist() 
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value, 
    shap_values[0,:], 
    feature_names=feature_names,
    show=False
)

st.pyplot(fig_waterfall, bbox_inches='tight')
plt.close(fig_waterfall)
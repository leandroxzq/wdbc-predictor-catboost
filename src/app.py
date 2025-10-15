import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from train import run_experiments

st.set_page_config(page_title="Breast Cancer CatBoost + SHAP", layout="wide")
st.title("Breast Cancer Wisconsin - CatBoost + SHAP")

# Sidebar
st.sidebar.header("Configurações do Dashboard")
show_summary = st.sidebar.checkbox("Mostrar Summary Plot (Global)", True)
show_decision = st.sidebar.checkbox("Mostrar Decision Plot (Local)", True)
show_waterfall = st.sidebar.checkbox("Mostrar Waterfall Plot (Local)", True)
sample_idx = st.sidebar.slider("Selecione a amostra para plots locais", 0, 113, 0)

# Inicializa sessão se não existir
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

# Botão para treinar modelos
if st.sidebar.button("Treinar modelos"):
    df_results, explainer, shap_values, X_test = run_experiments()
    
    st.session_state.df_results = df_results
    st.session_state.explainer = explainer
    st.session_state.shap_values = shap_values
    st.session_state.X_test = X_test
    st.session_state.results_ready = True
    st.success("Modelos treinados com sucesso!")

# Só mostra os resultados se já foram calculados
if st.session_state.results_ready:
    # Mostrar métricas
    st.subheader("Resultados de Performance")
    st.dataframe(st.session_state.df_results)

    # SHAP Summary Plot
    if show_summary:
        st.subheader("1. Summary Plot (Interpretabilidade Global)")
        fig_summary, ax_summary = plt.subplots(figsize=(10, 6))
        shap.summary_plot(st.session_state.shap_values, st.session_state.X_test, show=False)
        st.pyplot(fig_summary, bbox_inches='tight')
        plt.close(fig_summary)

    # SHAP Decision Plot
    if show_decision:
        st.subheader(f"2. Decision Plot (Amostra Local - {sample_idx})")
        fig_decision, ax_decision = plt.subplots(figsize=(10, 6))
        shap.decision_plot(
            st.session_state.explainer.expected_value,
            st.session_state.shap_values[sample_idx, :],
            st.session_state.X_test.iloc[sample_idx, :],
            show=False
        )
        st.pyplot(fig_decision, bbox_inches='tight')
        plt.close(fig_decision)

    # SHAP Waterfall Plot
    if show_waterfall:
        st.subheader(f"3. Waterfall Plot (Amostra Local - {sample_idx})")
        fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 6))
        feature_names = st.session_state.X_test.columns.tolist()
        shap.plots._waterfall.waterfall_legacy(
            st.session_state.explainer.expected_value,
            st.session_state.shap_values[sample_idx, :],
            feature_names=feature_names,
            show=False
        )
        st.pyplot(fig_waterfall, bbox_inches='tight')
        plt.close(fig_waterfall)

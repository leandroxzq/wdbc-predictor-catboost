# Breast Cancer Wisconsin Predictor - CatBoost

### Este projeto utiliza o algoritmo CatBoost para classificação de câncer de mama com base no dataset Wisconsin Diagnostic Breast Cancer (WDBC). Inclui interpretação de modelo usando SHAP e uma interface interativa via Streamlit.

## 🚀 Funcionalidades

### - Treino e avaliação de modelos:

- CatBoost
- SVM (Support Vector Machine)

### - Métricas de performance:

- Accuracy
- Sensitivity
- Specificity
- AUC (Area Under Curve)

### - Interpretação de modelo com SHAP:

- Summary Plot (importância global das features)
- Decision Plot (análise de decisão de uma amostra)
- Waterfall Plot (justificativa detalhada da decisão de uma amostra)

## 🛠️ Instalação

### 1. Clone o repositório

```bash
    git clone https://github.com/leandroxzq/wdbc-predictor-catboost.git
    cd wdbc-predictor-catboost
```

### 2. Crie um ambiente virtual e ative-o

# Windows:

```bash
    python -m venv venv
    venv\Scripts\activate
```

# Linux/Mac:

```bash
    python -m venv venv
    source venv/bin/activate
```

### 3. Instale as dependências

```bash
    pip install -r requirements.txt
```

## 🏃‍♂️ Executando o App

# No diretório raiz do projeto, execute:

```bash
    streamlit run src/app.py
```

### O aplicativo estará disponível no seu navegador.

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuração da Página
st.set_page_config(page_title="Simulador de Risco de Crédito", layout="wide")

# Título e Descrição
st.title("🏦 Simulator de Inadimplência (Pessoa Física)")
st.markdown("""
Este painel utiliza um modelo de Machine Learning (Random Forest) para prever 
a taxa de inadimplência baseada em cenários econômicos.
""")

# --- 1. Carga dos Artefatos ---
@st.cache_resource
def load_assets():
    model = joblib.load("models/model_final.pkl")
    scaler = joblib.load("models/scaler_final.pkl")
    # Carrega a estrutura de colunas usada no treino para garantir compatibilidade
    sample_data = pd.read_csv("data/processed/X_train_sample.csv")
    return model, scaler, sample_data

try:
    model, scaler, sample_data = load_assets()
except FileNotFoundError:
    st.error("Erro: Arquivos do modelo não encontrados. Verifique se rodou o notebook e salvou em 'models/'.")
    st.stop()

# --- 2. Sidebar de Parâmetros (O "What-If") ---
st.sidebar.header("⚙️ Configurar Cenário")

# Vamos focar nas variáveis TOP IMPORTANCE que você descobriu
# O usuário mexe nessas, o resto usamos a média histórica

# Selic (Defasada 6 meses)
selic_input = st.sidebar.slider(
    "Selic (há 6 meses) %", 
    min_value=2.0, max_value=20.0, value=float(sample_data['selic_lag_6'].mean()), step=0.25
)

# Inadimplência Anterior (Inércia)
inad_anterior = st.sidebar.slider(
    "Inadimplência Mês Anterior %", 
    min_value=1.0, max_value=10.0, value=float(sample_data['target_lag_1'].iloc[-1]), step=0.1
)

# Spread Bancário PF
spread_input = st.sidebar.slider(
    "Spread Bancário PF", 
    min_value=10.0, max_value=50.0, value=float(sample_data['spread_pf'].mean()), step=0.5
)

# --- 3. Preparar os Dados para o Modelo ---
# Criamos um dataframe com 1 linha contendo as médias de tudo
input_data = pd.DataFrame([sample_data.mean()], columns=sample_data.columns)

# Substituímos pelos valores que o usuário escolheu
input_data['selic_lag_6'] = selic_input
input_data['target_lag_1'] = inad_anterior
input_data['spread_pf'] = spread_input

# Se tivermos outras variáveis importantes, poderíamos adicionar mais sliders.
# O restante das 50+ colunas ficará com a média histórica (Ceteris Paribus).

# --- 4. Previsão ---
# Escalar os dados (O modelo espera dados padronizados)
input_data_scaled = scaler.transform(input_data)

# Prever
prediction = model.predict(input_data_scaled)[0]

# --- 5. Exibição dos Resultados ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("Previsão de Inadimplência")
    st.metric(
        label="Taxa Esperada (Mês seguinte)", 
        value=f"{prediction:.2f}%",
        delta=f"{prediction - inad_anterior:.2f}% vs Mês Anterior"
    )

with col2:
    st.subheader("Análise de Sensibilidade")
    st.write("Impacto da Selic (Defasada) no resultado:")
    
    # Pequeno gráfico mostrando como a Selic afeta o resultado (mantendo o resto fixo)
    sensibilidade = []
    selic_range = np.linspace(2, 20, 20)
    
    for s in selic_range:
        temp_df = input_data.copy()
        temp_df['selic_lag_6'] = s
        # Escalar e prever
        pred_s = model.predict(scaler.transform(temp_df))[0]
        sensibilidade.append(pred_s)
    
    chart_data = pd.DataFrame({"Selic Lag 6": selic_range, "Inadimplência Prevista": sensibilidade})
    st.line_chart(chart_data.set_index("Selic Lag 6"))

st.info("Nota: Este modelo assume que as demais variáveis macroeconômicas permanecem constantes na média histórica.")
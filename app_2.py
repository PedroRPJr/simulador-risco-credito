import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- Configuração ---
st.set_page_config(page_title="Simulador Avançado de Risco", layout="wide", page_icon="🌾")

st.title("🌾 Simulador Avançado de Risco de Crédito")
st.markdown("Projeções dinâmicas com Sazonalidade (Safra/Varejo) e Tendências Econômicas.")

# --- Funções ---
def load_assets(segmento):
    try:
        model = joblib.load(f"models/model_{segmento}.pkl")
        scaler = joblib.load(f"models/scaler_{segmento}.pkl")
        cols = pd.read_csv(f"models/columns_{segmento}.csv").columns.tolist()
        last_vals = pd.read_csv(f"models/last_values_{segmento}.csv", index_col=0).squeeze()
        return model, scaler, cols, last_vals
    except:
        return None, None, None, None

def predict_dynamic(model, scaler, base_input, start_inad, selic_trend, months=18):
    """
    Simulação Dinâmica Corrigida:
    Passa um DataFrame com nomes de colunas para o Scaler para evitar Warnings.
    """
    predictions = []
    current_inad = start_inad
    
    # Pega o mês atual da última coleta (ou usa o atual do sistema)
    current_month = int(base_input.get('mes', datetime.now().month))
    
    # Cópia para manipulação
    current_input = base_input.copy()
    
    # Lista oficial de colunas que o scaler espera (na ordem correta)
    feature_names = scaler.feature_names_in_
    
    for i in range(months):
        # 1. Atualizar Variáveis Dinâmicas
        current_input['target_lag_1'] = current_inad
        
        # A Selic muda conforme a tendência definida pelo usuário
        # Pegamos o valor anterior e somamos a tendência
        old_selic = current_input.get('selic_lag_6', 10.0)
        new_selic = old_selic + selic_trend 
        
        # Trava limites lógicos
        new_selic = max(2.0, min(40.0, new_selic))
        current_input['selic_lag_6'] = new_selic
        
        # Avançar o calendário (Sazonalidade)
        current_month += 1
        if current_month > 12: current_month = 1
        
        # Atualiza mês e safra se as colunas existirem no modelo
        if 'mes' in list(current_input.index):
            current_input['mes'] = current_month
        if 'periodo_safra' in list(current_input.index):
            current_input['periodo_safra'] = 1 if current_month in [2,3,4,5] else 0
            
        # 2. Escalar e Prever (CORREÇÃO AQUI)
        # Criamos um DataFrame de 1 linha com as colunas na ordem exata que o Scaler aprendeu
        df_input = pd.DataFrame([current_input])
        
        # Reindex garante que se faltar alguma coluna, ele preenche com 0, 
        # e se tiver coluna sobrando, ele ignora. E põe na ordem certa.
        df_input = df_input.reindex(columns=feature_names, fill_value=0)
        
        # Passamos o DataFrame (com nomes!) para o transform
        scaled = scaler.transform(df_input)
        
        pred = model.predict(scaled)[0]
        predictions.append(pred)
        
        current_inad = pred
        
    return predictions

# --- Interface ---
tabs = st.tabs(["👤 Pessoa Física", "🏢 Pessoa Jurídica", "🚜 Rural PF", "🚜 Rural PJ"])
mapa = {"👤 Pessoa Física": "PF", "🏢 Pessoa Jurídica": "PJ", "🚜 Rural PF": "Rural_PF", "🚜 Rural PJ": "Rural_PJ"}

for tab_name, segmento in mapa.items():
    with tabs[list(mapa.keys()).index(tab_name)]:
        model, scaler, cols, last_vals = load_assets(segmento)
        
        if not model:
            st.error("Modelo não encontrado. Re-treine com os novos dados.")
            continue
            
        col_cfg, col_chart = st.columns([1, 2])
        
        with col_cfg:
            st.subheader("Parâmetros de Simulação")
            
            # 1. Ponto de Partida
            st.markdown("**1. Ponto de Partida**")
            inad_start = st.number_input("Inadimplência Inicial (%)", value=float(last_vals.get('target_lag_1', 3.0)), step=0.1, key=f"start_{segmento}")
            selic_start = st.number_input("Selic Inicial (%)", value=float(last_vals.get('selic_lag_6', 11.0)), step=0.25, key=f"selic_{segmento}")
            
            st.markdown("---")
            
            # 2. Tendência (A Mágica da Dinâmica)
            st.markdown("**2. Tendência Econômica (Próx. 18 meses)**")
            trend_selic = st.slider(
                "Evolução da Selic (pp/mês)", 
                min_value=-0.50, max_value=0.50, value=0.0, step=0.05,
                format="%.2f",
                key=f"trend_{segmento}",
                help="Ex: -0.10 significa que a Selic cairá 0.10% todo mês (Queda de juros)."
            )
            
            txt_trend = "Estável"
            if trend_selic < 0: txt_trend = "Queda de Juros (Otimista)"
            if trend_selic > 0: txt_trend = "Aumento de Juros (Pessimista)"
            st.caption(f"Cenário: **{txt_trend}**")

        with col_chart:
            # Base Input
            base_input = pd.Series(last_vals)
            base_input['selic_lag_6'] = selic_start
            
            # Simulação
            projecao = predict_dynamic(model, scaler, base_input, inad_start, trend_selic)
            
            # Gráfico
            meses = range(1, 19)
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Estilo dependendo do segmento
            color = 'green' if 'Rural' in segmento else 'blue'
            
            ax.plot(meses, projecao, marker='o', color=color, linewidth=2, label=f"Projeção {segmento}")
            
            # Títulos e Eixos
            ax.set_title(f"Projeção 18 Meses: {segmento}", fontsize=14)
            ax.set_ylabel("Inadimplência (%)")
            ax.set_xlabel("Meses à Frente")
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Destaque Final
            final = projecao[-1]
            ax.annotate(f"{final:.2f}%", (18, final), xytext=(18, final + (final*0.05)), 
                        ha='center', fontweight='bold', color=color)
            
            st.pyplot(fig)
            
            # Insights
            var_total = projecao[-1] - projecao[0]
            st.info(f"Neste cenário, a inadimplência varia **{var_total:+.2f} pp** em 18 meses.")
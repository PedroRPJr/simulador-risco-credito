import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

# Configuração
Image.MAX_IMAGE_PIXELS = None
st.set_page_config(page_title="Simulador Macro", layout="wide", page_icon="📈")

st.title("📈 Simulador de Cenários Econômicos")
st.markdown("Sensibilidade: Selic (Geral), IPCA (Consumo) e Dólar (Rural).")

# Funções
def load_assets(segmento):
    try:
        model = joblib.load(f"models/model_{segmento}.pkl")
        scaler = joblib.load(f"models/scaler_{segmento}.pkl")
        cols = pd.read_csv(f"models/columns_{segmento}.csv").columns.tolist()
        last_vals = pd.read_csv(f"models/last_values_{segmento}.csv", index_col=0).squeeze()
        return model, scaler, cols, last_vals
    except FileNotFoundError:
        return None, None, None, None

def predict_scenario(model, scaler, feature_names, inputs_iniciais, selic_trend, ipca_trend, dolar_trend, months=18, is_decimal=False):
    """
    Simulação Completa com correção de escala (Decimal vs Porcentagem)
    """
    predictions = []
    current_input = inputs_iniciais.copy()
    
    # --- AJUSTE DE ESCALA (TRADUÇÃO) ---
    # Se o modelo foi treinado em decimais (0.10), mas o usuário digitou (10.0),
    # nós convertemos as entradas iniciais para decimal antes de começar.
    if is_decimal:
        base_selic = float(current_input.get('selic_lag_6', 0.10)) / 100
        base_ipca = float(current_input.get('ipca_lag_6', 0.005)) / 100
    else:
        base_selic = float(current_input.get('selic_lag_6', 10.0))
        base_ipca = float(current_input.get('ipca_lag_6', 0.5))
        
    base_dolar = float(current_input.get('dolar_ptax_lag_6', 5.0)) 
    
    current_month = int(current_input.get('mes', datetime.now().month))
    
    for i in range(months):
        # 1. Atualizar Economia
        # A tendência vem do slider (ex: +0.5). 
        # Se for decimal, temos que dividir a tendência também? 
        # R: Sim! Se o modelo opera em 0.10, um aumento de 1% é +0.01
        
        factor = 100 if is_decimal else 1
        
        new_selic = max(0, (base_selic * factor) + (selic_trend * (i+1)))
        new_ipca = max(-1, (base_ipca * factor) + (ipca_trend * (i+1)))
        
        # Devolve para a escala do modelo
        current_input['selic_lag_6'] = new_selic / factor if is_decimal else new_selic
        current_input['ipca_lag_6'] = new_ipca / factor if is_decimal else new_ipca
        
        if 'dolar_ptax_lag_6' in feature_names:
            current_input['dolar_ptax_lag_6'] = max(2.0, base_dolar + (dolar_trend * (i+1)))
        
        # 2. Sazonalidade
        current_month += 1
        if current_month > 12: current_month = 1
        
        if 'mes' in feature_names: current_input['mes'] = current_month
        if 'periodo_safra' in feature_names: 
            current_input['periodo_safra'] = 1 if current_month in [2,3,4,5] else 0
            
        # 3. Prever
        df_input = pd.DataFrame([current_input])
        df_input = df_input.reindex(columns=feature_names, fill_value=0)
        
        scaled = scaler.transform(df_input)
        pred = model.predict(scaled)[0]
        
        # Se o alvo também foi treinado em decimal (ex: inadimplencia 0.03), multiplicamos por 100 para mostrar bonito
        # Mas geralmente inadimplência já está em % no banco de dados. Vamos assumir que sim.
        pred = max(0.0, pred)
        
        predictions.append(pred)
        
    return predictions

# Interface
tabs = st.tabs(["👤 Pessoa Física", "🏢 Pessoa Jurídica", "🚜 Rural PF", "🚜 Rural PJ"])
mapa = {"👤 Pessoa Física": "PF", "🏢 Pessoa Jurídica": "PJ", "🚜 Rural PF": "Rural_PF", "🚜 Rural PJ": "Rural_PJ"}

nomes_limpos = {
    "PF": "Pessoa Física", "PJ": "Pessoa Jurídica",
    "Rural_PF": "Rural Pessoa Física", "Rural_PJ": "Rural Pessoa Jurídica"
}

for tab_name, segmento in mapa.items():
    with tabs[list(mapa.keys()).index(tab_name)]:
        
        model, scaler, cols, last_vals = load_assets(segmento)
        
        if not model:
            st.error(f"Modelo para {segmento} não encontrado.")
            continue
            
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.subheader("Cenário Macro")
            
            # --- DETECTOR DE ESCALA ---
            # Verifica se a Selic salva está em decimal (ex: 0.11) ou % (11.0)
            raw_selic = float(last_vals.get('selic_lag_6', 10.0))
            is_decimal = raw_selic < 1.0 # Se for menor que 1, assumimos que é decimal
            
            # Valores para EXIBIÇÃO (Sempre em %)
            display_selic = raw_selic * 100 if is_decimal else raw_selic
            display_ipca = float(last_vals.get('ipca_lag_6', 0.5))
            if is_decimal and display_ipca < 1: display_ipca *= 100
                
            start_selic = st.number_input("Selic Inicial (%)", value=display_selic, step=0.5, key=f"s_{segmento}")
            
            val_dolar = last_vals.get('dolar_ptax_lag_6', 5.0)
            start_dolar = st.number_input("Dólar Inicial (R$)", value=float(val_dolar), step=0.1, key=f"d_{segmento}")
            
            # Ajuste IPCA input
            start_ipca = st.number_input("IPCA/Mês Inicial (%)", value=display_ipca, step=0.1, key=f"i_{segmento}")

            st.markdown("---")
            st.write("Tendências (Simulação)")
            
            trend_selic = st.slider("Selic (pp/mês)", -0.5, 0.5, 0.0, 0.05, key=f"ts_{segmento}")
            trend_ipca = st.slider("IPCA (pp/mês)", -0.2, 0.2, 0.0, 0.01, key=f"ti_{segmento}")
            trend_dolar = st.slider("Dólar (R$/mês)", -0.50, 0.50, 0.0, 0.05, key=f"td_{segmento}")
            
            # Monta input inicial (Usuário vê %, mas mandamos para a função tratar)
            inputs_iniciais = last_vals.copy()
            inputs_iniciais['selic_lag_6'] = start_selic 
            inputs_iniciais['ipca_lag_6'] = start_ipca
            if 'dolar_ptax_lag_6' in inputs_iniciais.index:
                inputs_iniciais['dolar_ptax_lag_6'] = start_dolar

        with c2:
            try:
                # Passamos o flag 'is_decimal' para a função saber se precisa dividir por 100
                projecao = predict_scenario(model, scaler, cols, inputs_iniciais, trend_selic, trend_ipca, trend_dolar, is_decimal=is_decimal)
                projecao_base = predict_scenario(model, scaler, cols, inputs_iniciais, 0.0, 0.0, 0.0, is_decimal=is_decimal)
                
                # Gráfico
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Cores temáticas
                cor_linha = '#2ca02c' if 'Rural' in segmento else '#1f77b4'
                
                ax.plot(range(1, 19), projecao, marker='o', linewidth=3, color=cor_linha, label="Seu Cenário")
                ax.plot(range(1, 19), projecao_base, linestyle='--', color='gray', alpha=0.5, label="Cenário Estável")
                
                titulo_grafico = nomes_limpos.get(segmento, segmento)
                ax.set_title(f"Projeção: {titulo_grafico}", fontsize=14)
                ax.set_xlabel("Meses à Frente")
                ax.set_ylabel("Inadimplência (%)")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.3)
                
                y_vals = projecao + projecao_base
                # Evita crash se lista vazia
                if len(y_vals) > 0:
                    ax.set_ylim(min(y_vals)*0.95, max(y_vals)*1.05)
                
                st.pyplot(fig)
                
                var_total = projecao[-1] - projecao[0]
                st.info(f"Variação Projetada: {var_total:+.2f} pp")

            except Exception as e:
                st.error(f"Erro: {e}")
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

# --- CORREÇÃO DO ERRO DE "BOMB" (Imagens Gigantes) ---
Image.MAX_IMAGE_PIXELS = None 

# --- Configuração ---
st.set_page_config(page_title="Arena de Modelos - Risco", layout="wide", page_icon="⚔️")

st.title("⚔️ Arena de Modelos: Simulador de Risco")
st.markdown("Compare como diferentes Inteligências Artificiais reagem aos cenários econômicos.")

# --- Seletor de Modelo (Global) ---
st.sidebar.header("🧠 Cérebro da IA")
modelo_escolhido = st.sidebar.selectbox(
    "Escolha o Algoritmo:",
    ["Ridge (Linear/Tendência)", "RandomForest (Conservador)", "XGBoost (Agressivo)"]
)

# Mapear nome bonito para sufixo do arquivo
mapa_algoritmos = {
    "Ridge (Linear/Tendência)": "Ridge",
    "RandomForest (Conservador)": "RandomForest",
    "XGBoost (Agressivo)": "XGBoost"
}
sufixo_modelo = mapa_algoritmos[modelo_escolhido]

# --- Funções ---
@st.cache_resource
def load_assets(segmento, algoritmo_sufixo):
    try:
        # Carrega o modelo específico
        model = joblib.load(f"models/model_{segmento}_{algoritmo_sufixo}.pkl")
        # Scaler é comum ao segmento
        scaler = joblib.load(f"models/scaler_{segmento}.pkl")
        cols = pd.read_csv(f"models/columns_{segmento}.csv").columns.tolist()
        last_vals = pd.read_csv(f"models/last_values_{segmento}.csv", index_col=0).squeeze()
        return model, scaler, cols, last_vals
    except FileNotFoundError:
        return None, None, None, None

def predict_dynamic(model, scaler, base_input, start_inad, selic_trend, months=18):
    """ Simulação Dinâmica Robusta com Travas de Segurança """
    predictions = []
    current_inad = start_inad
    current_month = int(base_input.get('mes', datetime.now().month))
    current_input = base_input.copy()
    feature_names = scaler.feature_names_in_
    
    # Base Selic
    base_selic = current_input.get('selic_lag_6', 10.0)
    
    for i in range(months):
        # 1. Autoregressão
        current_input['target_lag_1'] = current_inad
        
        # 2. Tendência Selic
        step_trend = selic_trend * (i + 1)
        new_selic = base_selic + step_trend
        # Trava a Selic em valores realistas (Brasil nunca passou de 45% nem foi a 0%)
        new_selic = max(2.0, min(45.0, new_selic))
        
        # Atualiza todas as colunas de Selic
        for col in current_input.keys():
            if 'selic' in str(col):
                current_input[col] = new_selic
            
        # 3. Sazonalidade
        current_month += 1
        if current_month > 12: current_month = 1
        if 'mes' in list(current_input.index): current_input['mes'] = current_month
        if 'periodo_safra' in list(current_input.index):
            current_input['periodo_safra'] = 1 if current_month in [2,3,4,5] else 0
            
        # 4. Prever
        df_input = pd.DataFrame([current_input])
        df_input = df_input.reindex(columns=feature_names, fill_value=0)
        scaled = scaler.transform(df_input)
        
        # --- SEGURANÇA CONTRA CRASH DO XGBOOST/RIDGE ---
        raw_pred = model.predict(scaled)
        
        # Garante que é um número simples (float), não um array [4.5]
        if isinstance(raw_pred, (np.ndarray, list)):
            pred_val = float(raw_pred[0])
        else:
            pred_val = float(raw_pred)
            
        # --- TRAVA DE REALIDADE (Evita o erro de Imagem Gigante) ---
        # Nenhuma inadimplência passa de 20% no curto prazo ou fica negativa
        pred_val = max(0.0, min(20.0, pred_val))
        
        predictions.append(pred_val)
        current_inad = pred_val 
        
    return predictions

# --- Interface ---
tabs = st.tabs(["👤 Pessoa Física", "🏢 Pessoa Jurídica", "🚜 Rural PF", "🚜 Rural PJ"])
mapa_seg = {"👤 Pessoa Física": "PF", "🏢 Pessoa Jurídica": "PJ", "🚜 Rural PF": "Rural_PF", "🚜 Rural PJ": "Rural_PJ"}

for tab_name, segmento in mapa_seg.items():
    with tabs[list(mapa_seg.keys()).index(tab_name)]:
        
        model, scaler, cols, last_vals = load_assets(segmento, sufixo_modelo)
        
        if not model:
            st.error(f"Modelo {sufixo_modelo} para {segmento} não encontrado. Verifique a pasta 'models/'.")
            continue
            
        col_cfg, col_chart = st.columns([1, 2])
        
        with col_cfg:
            st.info(f"Modelo Ativo: **{modelo_escolhido}**")
            
            inad_start = st.number_input("Inadimplência Inicial (%)", value=float(last_vals.get('target_lag_1', 3.0)), step=0.1, key=f"s_{segmento}")
            selic_start = st.number_input("Selic Inicial (%)", value=float(last_vals.get('selic_lag_6', 11.0)), step=0.25, key=f"sel_{segmento}")
            
            st.markdown("---")
            trend_selic = st.slider(
                "Tendência da Selic (pp/mês)", 
                min_value=-0.50, max_value=0.50, value=0.0, step=0.05,
                key=f"t_{segmento}"
            )

        with col_chart:
            base_input = pd.Series(last_vals)
            base_input['selic_lag_6'] = selic_start
            
            try:
                projecao = predict_dynamic(model, scaler, base_input, inad_start, trend_selic)
                
                # Gráfico
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(range(1, 19), projecao, marker='o', linewidth=2.5, label=f"{sufixo_modelo}", color="#1f77b4")
                
                ax.set_title(f"Projeção ({sufixo_modelo}) - 18 Meses", fontsize=14)
                ax.set_ylabel("Inadimplência (%)")
                ax.set_xlabel("Meses à frente")
                ax.set_ylim(bottom=0) # Garante que eixo Y começa no 0
                
                # Se os valores forem muito altos, ajusta o topo
                if max(projecao) > 10:
                    ax.set_ylim(top=max(projecao) + 2)
                else:
                    ax.set_ylim(top=10)

                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend()
                
                final = projecao[-1]
                ax.annotate(f"{final:.2f}%", (18, final), xytext=(18, final+0.5), fontweight='bold')
                
                st.pyplot(fig)
                plt.close(fig) # Limpa a memória explicitamente
                
            except Exception as e:
                st.error(f"Erro ao gerar projeção: {e}")
#  Projeto: Simulador de Inadimplência de Operações de Crédito (PF/PJ e Crédito Rural)

## 1. Organização do Modelo (Limpeza e Reprodutibilidade)
projeto_inadimplencia/
├── app.py
├── models/
│   ├── model_PF.pkl      # Apenas os .pkl finais (Ridge/Focados)
│   ├── scaler_PF.pkl
│   ├── columns_PF.csv
│   └── last_values_PF.csv
│   # ... (repita para PJ, Rural, etc. Apague os antigos XGBoost se não estiver usando)
├── data/
│   └── processed/df_modelagem_v3.csv
├── notebooks/
│   ├── 01_coleta_variaveis.ipynb
│   ├── 02_analise_exploratoria.ipynb
│   ├── 03_treinamento_final_focado.ipynb # Renomeie o '08' para ser o oficial
│   └── archive/          # Jogue os notebooks 04, 05, 06, 07 aqui
└── requirements.txt      # CRÍTICO para reprodução

projeto_inadimplencia/
├── api/  
├── dashboard/  
├── data/  
│   ├── processed/  
│   │   ├── df_modelagem_v3.csv  
│   │   ├── df_modelagem.csv  
│   │   └── X_train_sample.csv  
│   │
│   └── raw/  
│       ├── df_economico.csv  
│       ├── df_eventos_politicos.csv  
│       ├── df_ibge.csv  
│       ├── df_inadimplencia.csv  
│       ├── df_inmet.csv  
│       └── INPE.zip   
│
├── models/  
│   ├── model_PF_XGBoost.pkl  
│   ├── model_PF.pkl  
│   ├── model_PJ_RandomForest.pkl  
│   ├── model_PJ_Ridge.pkl  
│   ├── model_PJ_XGBoost.pkl  
│   ├── model_PJ.pkl  
│   ├── model_Rural_PF_RandomForest.pkl  
│   ├── model_Rural_PF_Ridge.pkl  
│   ├── model_Rural_PF_XGBoost.pkl  
│   ├── model_Rural_PF.pkl  
│   ├── model_Rural_PJ_RandomForest.pkl  
│   ├── model_Rural_PJ_Ridge.pkl  
│   ├── model_Rural_PJ_XGBoost.pkl  
│   ├── model_Rural_PJ.pkl  
│   ├── scaler_final.pkl  
│   ├── scaler_PF.pkl  
│   ├── scaler_PJ.pkl  
│   ├── scaler_Rural_PF.pkl  
│   └── scaler_Rural_PJ.pkl  
│
├── notebooks/  
│   ├── 01_coleta_variaveis.ipynb  
│   ├── 02_analise_exploratoria.ipynb  
│   ├── 03_feature_engineering.ipynb  
│   ├── 04_treinamento_modelo.ipynb  
│   ├── 05_treinamento_multimodelo.ipynb  
│   ├── 06_treinamento_comparativo.ipynb  
│   ├── 07_treinamento_comparativo_delta.ipynb  
│   └── 08_treinamento_focado.ipynb   
│
├── reports/  
├── src/  
├── venv/  
├── app_1.py  
├── app_2.py  
├── app_3.py  
├── app_4.py  
├── app_5.py  
├── readme.md    
└── requirements.txt 
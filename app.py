import streamlit as st
import pandas as pd
from visualization import show_visualization_tab
from data_explorer import show_data_explorer_tab
from ml_training import show_ml_training_tab
from dnn_training import show_dnn_training_tab
from ensemble_training import show_ensemble_training_tab
from predictions import show_predictions_tab
from updated_predictions import show_helper_tab


import ee
from google.oauth2 import service_account

service_account_info = dict(st.secrets["google_ee"])  # No need for .to_json()

credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=["https://www.googleapis.com/auth/earthengine"]
)

ee.Initialize(credentials)


def load_data():
    file_path = "Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv"
    return pd.read_csv(file_path)


df = load_data()

df = df[
    df["Country"].isin(
        [
            "Morocco",
            "Tunisia",
            "Mauritania",
            "Iraq",
            "Syrian Arab Republic",
            "Azerbaijan",
            "Afghanistan",
            "Pakistan",
            "Uzbekistan",
            "Tajikistan",
            "Kyrgyzstan",
            "Egypt",
            "Jordan",
            "Lebanon",
        ]
    )
]

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📍 Visualization",
        "📊 Data Explorer",
        "🖥️ ML Training",
        "🧠 DNN Training",
        "📈 Ensemble Training",
        "🔮 Predictions",
    ]
)

# Visualization Tab
with tab1:
    show_visualization_tab(df)

# Data Explorer Tab
with tab2:
    show_data_explorer_tab(df)

# ML Training Tab
with tab3:
    show_ml_training_tab(df)

# DL Training Tab
with tab4:
    show_dnn_training_tab(df)

# Ensemble Training Tab
with tab5:
    show_ensemble_training_tab(df)

# Predictions Tab
with tab6:
    show_helper_tab(df)

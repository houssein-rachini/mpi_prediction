import streamlit as st
import pandas as pd
from pathlib import Path
from visualization import show_visualization_tab
from data_explorer import show_data_explorer_tab
from ml_training import show_ml_training_tab
from dnn_training import show_dnn_training_tab
from ensemble_training import show_ensemble_training_tab
from stacked_ensemble import show_stacking_tab
from updated_predictions import show_helper_tab
from ee_auth import initialize_earth_engine

initialize_earth_engine()


DATASET_OPTIONS = {
    "0m buffer": "merged_all_vars_0m_original_ref_gaul.csv",
    "250m buffer": "merged_all_vars_250m_original_ref_gaul.csv",
    "500m buffer": "Final_Merged_MPI_LST_NTL_NDVI_v4 - original.csv",
    "1000m buffer": "merged_all_vars_1000m_original_ref_gaul.csv",
    "2000m buffer": "merged_all_vars_2000m_original_ref_gaul.csv",
    "3000m buffer": "merged_all_vars_3000m_original_ref_gaul.csv",
}


def get_file_signature(file_path):
    path = Path(file_path)
    stat = path.stat()
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


@st.cache_data(show_spinner=False)
def load_data(file_path, file_signature):
    # file_signature participates in Streamlit's cache key.
    _ = file_signature
    return pd.read_csv(file_path)


selected_dataset_label = st.sidebar.selectbox(
    "Training dataset",
    options=list(DATASET_OPTIONS.keys()),
    index=2,
)
selected_file = DATASET_OPTIONS[selected_dataset_label]
st.session_state["selected_buffer"] = selected_dataset_label
st.session_state["selected_dataset_file"] = selected_file
file_signature = get_file_signature(selected_file)
if st.sidebar.button("Clear data cache"):
    st.cache_data.clear()
df = load_data(selected_file, file_signature)
st.sidebar.caption(f"Loaded: `{selected_file}`")
st.markdown(f"### Active buffer: `{selected_dataset_label}`")

# df = df[
#     df["Country"].isin(
#         [
#             "Morocco",
#             "Tunisia",
#             "Mauritania",
#             "Iraq",
#             "Syrian Arab Republic",
#             "Azerbaijan",
#             "Afghanistan",
#             "Pakistan",
#             "Uzbekistan",
#             "Tajikistan",
#             "Kyrgyzstan",
#             "Egypt",
#             "Jordan",
#             "Turkmenistan",
#         ]
#     )
# ]

st.markdown(
    """
<style>
/* Keep tabs on one line and allow horizontal scrolling for cleaner layout */
div[data-baseweb="tab-list"] {
    flex-wrap: nowrap !important;
    overflow-x: auto;
    overflow-y: hidden;
    scrollbar-width: thin;
    gap: 0.35rem;
}
div[data-baseweb="tab"] {
    white-space: nowrap;
    flex: 0 0 auto;
    min-height: 2.25rem;
    padding-top: 0.25rem;
    padding-bottom: 0.25rem;
}
</style>
""",
    unsafe_allow_html=True,
)


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Visualization",
        "Data Explorer",
        "ML Training",
        "DNN Training",
        "Ensemble Training",
        "Stacked Ensemble",
        "Predictions",
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

# Stacked Ensemble Tab
with tab6:
    show_stacking_tab(df)

# Predictions Tab
with tab7:
    show_helper_tab(df)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import pandas as pd

DEFAULT_LAYERS = [
    {"type": "Dense", "units": 256, "activation": "relu"},
    {"type": "BatchNormalization"},
    {"type": "Dropout", "rate": 0.15},
    {"type": "Dense", "units": 128, "activation": "relu"},
    {"type": "BatchNormalization"},
    {"type": "Dropout", "rate": 0.10},
    {"type": "Dense", "units": 64, "activation": "relu"},
    {"type": "BatchNormalization"},
    {"type": "Dense", "units": 32, "activation": "relu"},
    {"type": "Dense", "units": 1, "activation": "relu"},
]


def create_dnn_model(
    input_dim,
    layers_config,
    initial_learning_rate,
    weight_decay,
    optimizer_choice,
    loss_function_choice,
    huber_delta,
):
    """Builds a DNN model based on user-defined architecture."""
    lr_schedule = CosineDecay(
        initial_learning_rate=initial_learning_rate, decay_steps=10000, alpha=0.0001
    )
    model = Sequential()

    for i, layer in enumerate(layers_config):
        if layer["type"] == "Dense":
            model.add(
                Dense(
                    layer["units"],
                    activation=layer["activation"],
                    input_shape=(input_dim,) if i == 0 else (),
                )
            )
        elif layer["type"] == "BatchNormalization":
            model.add(BatchNormalization())
        elif layer["type"] == "Dropout":
            model.add(Dropout(layer["rate"]))

    # Set optimizer based on user selection
    if optimizer_choice == "AdamW":
        optimizer = AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
    elif optimizer_choice == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif optimizer_choice == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    elif optimizer_choice == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

    # Set loss function based on user selection
    if loss_function_choice == "Huber":
        loss = Huber(delta=huber_delta)
    elif loss_function_choice == "Mean Squared Error":
        loss = "mse"
    elif loss_function_choice == "Mean Absolute Error":
        loss = "mae"

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    # model.compile(
    #     optimizer=AdamW(learning_rate=lr_schedule, weight_decay=1e-5),
    #     loss=Huber(delta=1.0),
    #     metrics=["mae"],
    # )
    return model


def train_ensemble_model(
    X_train,
    X_val,
    y_train,
    y_val,
    epochs,
    initial_learning_rate,
    batch_size,
    early_stopping_patience,
    layers_config,
    weight_decay,
    optimizer_choice,
    loss_function_choice,
    huber_delta,
    alpha,
    base_model,
    base_model_params,
    scaler_choice,
):
    """Trains a DNN model and saves the model and scaler."""
    # Standardize features
    # scaler = StandardScaler()
    # scaler = MinMaxScaler()

    if scaler_choice == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train base model
    if base_model == "XGBoost":
        base_model_instance = xgb.XGBRegressor(**base_model_params)
        base_model_instance.fit(X_train_scaled, y_train)
    elif base_model == "Random Forest":
        base_model_instance = RandomForestRegressor(**base_model_params)
        base_model_instance.fit(X_train_scaled, y_train)

    # Train DNN model
    dnn_model = create_dnn_model(
        X_train_scaled.shape[1],
        st.session_state.layers_config,  # Use stored layers
        initial_learning_rate,
        weight_decay,
        optimizer_choice,
        loss_function_choice,
        huber_delta,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
    )

    history = dnn_model.fit(
        X_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_scaled, y_val),
        verbose=1,
        callbacks=[early_stopping],
    )

    # Ensemble model predictions
    y_pred_dnn = dnn_model.predict(X_val_scaled).flatten()
    y_pred_base = base_model_instance.predict(X_val_scaled)
    y_pred_ensemble = alpha * y_pred_dnn + (1 - alpha) * y_pred_base

    # Compute evaluation metrics
    mae = mean_absolute_error(y_val, y_pred_ensemble)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))
    r2 = r2_score(y_val, y_pred_ensemble)

    # Save models and scaler
    joblib.dump(scaler, "ensemble_scaler.pkl")
    if base_model == "XGBoost":
        base_model_instance.save_model("trained_ensemble_xgb_model.json")
        dnn_model.save("trained_ensemble_xgb_dnn_model.h5")
    elif base_model == "Random Forest":
        joblib.dump(base_model_instance, "trained_ensemble_rf_model.pkl")
        dnn_model.save("trained_ensemble_rf_dnn_model.h5")
    if base_model == "XGBoost":
        st.write("✅ XGBoost model saved as trained_ensemble_xgb_model.json")
    elif base_model == "Random Forest":
        st.write("✅ Random Forest model saved as trained_ensemble_rf_model.pkl")

    st.write("✅ DNN model saved as trained_ensemble_dnn_model.h5")
    st.write("✅ Scaler saved as ensemble_scaler.pkl")
    return y_val, y_pred_ensemble, history.history, mae, rmse, r2


def plot_loss_curve(history):
    """Plots the training vs validation loss curve."""
    fig, ax = plt.subplots()
    ax.plot(history["loss"], label="Training Loss", color="red")
    ax.plot(history["val_loss"], label="Validation Loss", color="green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curve")
    ax.legend()
    st.pyplot(fig)


def plot_results(y_val, y_pred):
    """Plots actual vs predicted results."""
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x=y_val, y=y_pred, alpha=0.7)
    plt.axline((0, 0), slope=1, color="red", linestyle="--")
    plt.xlabel("Actual MPI")
    plt.ylabel("Predicted MPI")
    plt.title("Actual vs Predicted MPI (Ensemble Model)")
    st.pyplot(fig)


def plot_residuals(y_val, y_pred):
    """Plots residuals to check model performance."""
    residuals = y_val - y_pred

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_val, y=residuals, alpha=0.7)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Actual MPI")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot (Error Analysis)")

    st.pyplot(fig)


def show_ensemble_training_tab(df):
    """Displays the UI for training the ensemble learning model."""
    st.title("📈Ensemble Model Training")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols.remove("Year")
    default_cols = [
        "StdDev_NTL",
        "Mean_GPP",
        "StdDev_Pop",
        "StdDev_LST",
        "StdDev_NDVI",
        "Mean_NTL",
        "StdDev_GPP",
        "Mean_Pop",
        "Mean_LST",
        "Mean_NDVI",
    ]
    selected_features = st.multiselect(
        "Select features for training:",
        numeric_cols,
        default=default_cols,
        key="ensemble_features",
    )

    if "MPI" not in selected_features:
        selected_features.append("MPI")
    df_selected = df[selected_features].dropna()
    X = df_selected.drop(columns=["MPI"])
    y = np.maximum(df_selected["MPI"], 0)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    alpha = st.slider("Ensemble Weight (DNN Contribution)", 0.0, 1.0, 0.15)
    epochs = st.slider("Number of Epochs", 10, 600, 300, key="ensemble_epochs")
    # Select optimizer
    optimizer_choice = st.selectbox(
        "Select Optimizer",
        ["AdamW", "Adam", "SGD", "RMSprop"],
        key="ensemble_optimizer",
    )
    scaler_choice = "StandardScaler"
    initial_learning_rate = st.number_input(
        "Initial Learning Rate",
        min_value=1e-7,
        max_value=0.1,
        value=0.0005,
        step=0.0001,
        format="%.6f",
        key="ensemble_lr",
    )
    weight_decay = 0.0
    if optimizer_choice == "AdamW":
        weight_decay = st.number_input(
            "Weight Decay (for AdamW)",
            min_value=0.0,
            max_value=1e-2,
            value=1e-6,
            step=1e-6,
            format="%.6f",
            key="ensemble_wd",
        )

    # Select loss function
    loss_function_choice = st.selectbox(
        "Select Loss Function",
        ["Huber", "Mean Squared Error", "Mean Absolute Error"],
        key="ensemble_loss_function",
    )

    # Specify delta for Huber loss if selected
    huber_delta = None
    if loss_function_choice == "Huber":
        huber_delta = st.number_input(
            "Huber Loss Delta",
            min_value=0.1,
            max_value=10.0,
            value=0.4,
            step=0.1,
            format="%.1f",
            key="ensemble_huber_delta",
        )

    batch_size = st.slider("Batch Size", 8, 1024, 128, key="ensemble_batch_size")
    early_stopping_patience = st.slider(
        "Early Stopping Patience", 5, 50, 20, key="ensemble_patience"
    )

    st.subheader("Neural Network Architecture")
    if "layers_config" not in st.session_state:
        st.session_state.layers_config = DEFAULT_LAYERS.copy()
    layers = []
    num_layers = st.number_input(
        "Number of Layers",
        1,
        20,
        len(st.session_state.layers_config),
        step=1,
        key="ensemble_num_layers",
    )
    if num_layers > len(st.session_state.layers_config):
        st.session_state.layers_config.extend(
            [{"type": "Dense", "units": 64, "activation": "relu"}]
            * (num_layers - len(st.session_state.layers_config))
        )
    elif num_layers < len(st.session_state.layers_config):
        st.session_state.layers_config = st.session_state.layers_config[:num_layers]
    for i in range(num_layers):
        col1, col2, col3 = st.columns([0.4, 0.3, 0.3])

        layer_type = col1.selectbox(
            f"Layer {i+1} Type",
            ["Dense", "BatchNormalization", "Dropout"],
            index=["Dense", "BatchNormalization", "Dropout"].index(
                st.session_state.layers_config[i]["type"]
            ),
            key=f"ensemble_type_{i}",
        )

        if layer_type == "Dense":
            units = col2.slider(
                f"Units {i+1}",
                1,
                512,
                st.session_state.layers_config[i].get("units", 128),
                key=f"ensemble_units_{i}",
            )
            activation = col3.selectbox(
                f"Activation {i+1}",
                ["relu", "tanh", "sigmoid", "linear", "softplus"],
                index=["relu", "tanh", "sigmoid", "linear", "softplus"].index(
                    st.session_state.layers_config[i].get("activation", "relu")
                ),
                key=f"ensemble_activation_{i}",
            )
            layers.append({"type": "Dense", "units": units, "activation": activation})

        elif layer_type == "Dropout":
            rate = col2.slider(
                f"Dropout Rate {i+1}",
                0.0,
                0.5,
                st.session_state.layers_config[i].get("rate", 0.1),
                key=f"ensemble_dropout_{i}",
            )
            layers.append({"type": "Dropout", "rate": rate})

        elif layer_type == "BatchNormalization":
            layers.append({"type": "BatchNormalization"})

    # Store updated layers in session state
    st.session_state.layers_config = layers
    st.subheader("Base Model")
    base_model = st.selectbox("Select Base Model", ["XGBoost", "Random Forest"])

    base_model_params = {}
    if base_model == "XGBoost":
        base_model_params = {
            "learning_rate": st.slider(
                "Learning Rate", 0.01, 0.5, 0.05, key="ensemble_xgb_learning_rate"
            ),
            "max_depth": st.slider("Max Depth", 3, 10, 6, key="ensemble_xgb_max_depth"),
            "n_estimators": st.slider(
                "Number of Trees", 50, 500, 200, key="ensemble_xgb_n_estimators"
            ),
            "min_child_weight": st.slider(
                "Min Child Weight", 1, 10, 1, key="ensemble_xgb_min_child_weight"
            ),
            "random_state": 42,
        }

    elif base_model == "Random Forest":
        base_model_params = {
            "n_estimators": st.slider("Number of Trees", 50, 300, 150),
            "min_samples_split": st.slider("Min Samples Split", 2, 10, 2),
            "min_samples_leaf": st.slider("Min Samples Leaf", 1, 10, 1),
            "random_state": 42,
        }

    if st.button("Train Model", key=f"ensemble_train_button"):
        with st.spinner("Training the model..."):
            y_val, y_pred_ensemble, history, mae, rmse, r2 = train_ensemble_model(
                X_train,
                X_val,
                y_train,
                y_val,
                epochs,
                initial_learning_rate,
                batch_size,
                early_stopping_patience,
                layers,
                weight_decay,
                optimizer_choice,
                loss_function_choice,
                huber_delta,
                alpha,
                base_model,
                base_model_params,
                scaler_choice,
            )
        st.success("Training completed!")
        st.subheader("📊 Model Performance")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")
        # metrics of the last few epochs
        st.write("### Epochs")
        st.write(pd.DataFrame(history))
        st.subheader("Training and Validation Loss Curve")
        plot_loss_curve(history)
        st.subheader("Actual vs Predicted Scatter Plot")
        plot_results(y_val, y_pred_ensemble)
        st.subheader("Residual Plot (Error Analysis)")
        plot_residuals(y_val, y_pred_ensemble)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
import joblib
import pandas as pd
from io import BytesIO


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


def render_plot_download(fig, file_stem):
    plot_counter = st.session_state.get("dnn_plot_download_counter", 0)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        "Download Plot (PNG, 300 DPI)",
        data=buffer.getvalue(),
        file_name=f"{file_stem}.png",
        mime="image/png",
        key=f"dnn_plot_download_{file_stem}_{plot_counter}",
    )
    st.session_state["dnn_plot_download_counter"] = plot_counter + 1


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
        initial_learning_rate=initial_learning_rate, decay_steps=10000, alpha=0.0005
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


def train_dnn_model(
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

    # Train DNN model
    dnn_model = create_dnn_model(
        X_train_scaled.shape[1],
        layers_config,
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

    # Compute evaluation metrics
    y_pred_dnn = dnn_model.predict(X_val_scaled).flatten()
    mae = mean_absolute_error(y_val, y_pred_dnn)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_dnn))
    r2 = r2_score(y_val, y_pred_dnn)

    # Save model and scaler
    joblib.dump(scaler, "dnn_scaler.pkl")
    dnn_model.save("trained_dnn_model.h5")
    st.write("✅ Model and Scaler saved to 'trained_dnn_model.h5' and 'dnn_scaler.pkl'")
    return y_val, y_pred_dnn, history.history, mae, rmse, r2


def plot_loss_curve(history):
    """Plots the training vs validation loss curve."""
    fig, ax = plt.subplots()
    ax.plot(history["loss"], label="Training Loss", color="red")
    ax.plot(history["val_loss"], label="Validation Loss", color="green")
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("Training and Validation Loss Curve", fontsize=15)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11)
    st.pyplot(fig)
    render_plot_download(fig, "dnn_loss_curve")
    plt.close(fig)


def plot_results(y_val, y_pred):
    """Plots actual vs predicted results."""
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x=y_val, y=y_pred, alpha=0.7)
    plt.axline((0, 0), slope=1, color="red", linestyle="--")
    plt.xlabel("Actual MPI")
    plt.ylabel("Predicted MPI")
    plt.title("Actual vs Predicted MPI (DNN Model)")
    st.pyplot(fig)
    render_plot_download(fig, "dnn_actual_vs_predicted_scatter")
    plt.close(fig)


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
    render_plot_download(fig, "dnn_residual_plot")
    plt.close(fig)


def _make_validation_results(ids_df, actual, predicted):
    results_df = pd.DataFrame(
        {
            "Actual_MPI": pd.Series(actual).reset_index(drop=True),
            "Predicted_MPI": pd.Series(predicted).reset_index(drop=True),
        }
    )
    if ids_df is not None and not ids_df.empty:
        results_df = pd.concat([ids_df.reset_index(drop=True), results_df], axis=1)
    return results_df


def _cv_download_buttons(cv_metrics_df, cv_oof_df, key_prefix):
    """Render download buttons for per-fold CV metrics and OOF predictions."""
    if cv_metrics_df is not None and not cv_metrics_df.empty:
        st.download_button(
            "⬇️ Download per-fold CV metrics CSV",
            data=cv_metrics_df.to_csv(index=False).encode("utf-8"),
            file_name="dnn_cv_fold_metrics.csv",
            mime="text/csv",
            key=f"{key_prefix}_cv_metrics",
        )
    if cv_oof_df is not None and not cv_oof_df.empty:
        st.download_button(
            "⬇️ Download CV out-of-fold predictions CSV",
            data=cv_oof_df.to_csv(index=False).encode("utf-8"),
            file_name="dnn_cv_oof_predictions.csv",
            mime="text/csv",
            key=f"{key_prefix}_cv_oof",
        )


def show_dnn_training_tab(df):
    """Displays the UI for training the deep learning model."""
    st.title("🧠Deep Learning Model Training")
    if "dnn_results" in st.session_state:
        st.subheader("📊 Previous Training Results")
        results = st.session_state["dnn_results"]
        st.write(f"**Mean Absolute Error (MAE):** {results['mae']:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {results['rmse']:.4f}")
        st.write(f"**R² Score:** {results['r2']:.4f}")
        st.write("### Epochs")
        st.write(pd.DataFrame(results["history"]))
        if "validation_results" in results:
            st.download_button(
                "Download Actual vs Predicted CSV",
                data=results["validation_results"].to_csv(index=False).encode("utf-8"),
                file_name="dnn_validation_results.csv",
                mime="text/csv",
                key="dnn_download_previous_results",
            )
        _cv_prev = results.get("cv_df")
        if _cv_prev is not None and not _cv_prev.empty:
            st.write(f"**{results.get('n_folds', '')}-Fold GroupKFold CV (by Country)**")
            st.dataframe(_cv_prev.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}))
            _cv_download_buttons(_cv_prev, results.get("cv_oof"), "dnn_prev")
        plot_loss_curve(results["history"])
        plot_results(results["y_val"], results["y_pred"])
        plot_residuals(results["y_val"], results["y_pred"])
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols.remove("Year")
    default_cols = [
        "Mean_NTL",
        "Mean_LST",
        "Median_NTL",
        "Mean_LST_Day",
        "NTL_anom",
        "StdDev_NTL",
        "StdDev_Pop",
        "ndvi_lst_ratio",
        "Mean_Pop",
        "Median_Pop",
        "Mean_GPP",
        "Sum_NTL",
        "NDVI_anom",
        "LSTN_anom",
        "LST_Day_anom",
        "NTL_anom_lag1",
        "Mean_BUILT_S",
        "Median_BUILT_S",
        "StdDev_BUILT_S",
        "StdDev_BUILT_V",
    ]
    selected_features = st.multiselect(
        "Select features for training:", numeric_cols, default=[c for c in default_cols if c in numeric_cols]
    )
    if selected_features:
        n_rows = df.dropna(subset=["MPI"] + selected_features).shape[0]
        st.caption(f"Rows available for training: **{n_rows:,}**")

    if "MPI" not in selected_features:
        selected_features.append("MPI")
    df_selected = df[selected_features].dropna()
    X = df_selected.drop(columns=["MPI"])
    y = np.maximum(df_selected["MPI"], 0)
    wanted_ids = ["Country", "Region", "Year"]
    id_cols_present = [c for c in wanted_ids if c in df.columns]
    df_ids = df.loc[df_selected.index, id_cols_present]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    ids_val = df_ids.loc[X_val.index]

    epochs = st.slider("Number of Epochs", 10, 500, 200, key="dnn_epochs")
    # Select optimizer
    optimizer_choice = st.selectbox(
        "Select Optimizer", ["AdamW", "Adam", "SGD", "RMSprop"], key="optimizer"
    )
    scaler_choice = "StandardScaler"
    initial_learning_rate = st.number_input(
        "Initial Learning Rate",
        min_value=1e-7,
        max_value=0.1,
        value=0.001,
        step=0.0001,
        format="%.6f",
        key="dnn_lr",
    )
    weight_decay = 0.0
    if optimizer_choice == "AdamW":
        weight_decay = st.number_input(
            "Weight Decay (for AdamW)",
            min_value=0.0,
            max_value=1e-2,
            value=1e-5,
            step=1e-6,
            format="%.6f",
            key="dnn_wd",
        )

    # Select loss function
    loss_function_choice = st.selectbox(
        "Select Loss Function",
        ["Huber", "Mean Squared Error", "Mean Absolute Error"],
        key="loss_function",
    )

    # Specify delta for Huber loss if selected
    huber_delta = None
    if loss_function_choice == "Huber":
        huber_delta = st.number_input(
            "Huber Loss Delta",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            format="%.1f",
            key="dnn_huber_delta",
        )

    batch_size = st.slider("Batch Size", 8, 1024, 128, key="dnn_batch_size")
    early_stopping_patience = st.slider(
        "Early Stopping Patience", 5, 50, 10, key="patience"
    )

    st.subheader("Neural Network Architecture")
    if "dnn_layers_config" not in st.session_state:
        st.session_state.dnn_layers_config = DEFAULT_LAYERS.copy()
    layers = []
    num_layers = st.number_input(
        "Number of Layers",
        1,
        20,
        len(st.session_state.dnn_layers_config),
        step=1,
        key="dnn_num_layers",
    )
    if num_layers > len(st.session_state.dnn_layers_config):
        st.session_state.dnn_layers_config.extend(
            [{"type": "Dense", "units": 64, "activation": "relu"}]
            * (num_layers - len(st.session_state.dnn_layers_config))
        )
    elif num_layers < len(st.session_state.dnn_layers_config):
        st.session_state.dnn_layers_config = st.session_state.dnn_layers_config[
            :num_layers
        ]

    for i in range(num_layers):
        col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
        layer_type = col1.selectbox(
            f"Layer {i+1} Type",
            ["Dense", "BatchNormalization", "Dropout"],
            index=["Dense", "BatchNormalization", "Dropout"].index(
                st.session_state.dnn_layers_config[i]["type"]
            ),
            key=f"dnn_type_{i}",
        )
        if layer_type == "Dense":
            units = col2.slider(
                f"Units {i+1}",
                1,
                512,
                st.session_state.dnn_layers_config[i].get("units", 128),
                key=f"dnn_units_{i}",
            )
            activation = col3.selectbox(
                f"Activation {i+1}",
                ["relu", "tanh", "sigmoid", "linear", "softplus"],
                index=["relu", "tanh", "sigmoid", "linear", "softplus"].index(
                    st.session_state.dnn_layers_config[i].get("activation", "relu")
                ),
                key=f"dnn_activation_{i}",
            )
            layers.append({"type": "Dense", "units": units, "activation": activation})
        elif layer_type == "Dropout":
            rate = col2.slider(
                f"Dropout Rate {i+1}",
                0.0,
                0.5,
                st.session_state.dnn_layers_config[i].get("rate", 0.1),
                key=f"dnn_dropout_{i}",
            )
            layers.append({"type": "Dropout", "rate": rate})
        elif layer_type == "BatchNormalization":
            layers.append({"type": "BatchNormalization"})
    st.session_state.dnn_layers_config = layers

    use_cv = st.checkbox("Use GroupKFold cross-validation (by Country)", value=False, key="dnn_use_cv")
    n_folds = 5
    if use_cv:
        n_folds = st.slider("Number of folds", 2, 10, 5, key="dnn_n_folds")
        st.caption("Each fold trains a full DNN — this may take several minutes.")

    if st.button("Train Model", key=f"dnn_train_button"):
        cv_df = pd.DataFrame()
        cv_oof_df = pd.DataFrame()
        if use_cv and "Country" in df.columns:
            cv_fold_rows = []
            cv_oof_parts = []
            gkf = GroupKFold(n_splits=n_folds)
            X_np = X.values.astype(np.float32)
            y_np = y.values.astype(np.float32)
            groups_arr = df.loc[df_selected.index, "Country"].values
            with st.spinner(f"Running {n_folds}-fold GroupKFold CV..."):
                for fold, (tr_idx, te_idx) in enumerate(gkf.split(X_np, y_np, groups_arr)):
                    _, y_fold_pred, _, fold_mae, fold_rmse, fold_r2 = train_dnn_model(
                        X_np[tr_idx], X_np[te_idx],
                        y_np[tr_idx], y_np[te_idx],
                        epochs, initial_learning_rate, batch_size,
                        early_stopping_patience, layers, weight_decay,
                        optimizer_choice, loss_function_choice, huber_delta, scaler_choice,
                    )
                    cv_fold_rows.append({"Fold": fold + 1, "MAE": fold_mae, "RMSE": fold_rmse, "R²": fold_r2})
                    _oof_part = df_ids.iloc[te_idx].reset_index(drop=True)
                    _oof_part.insert(0, "Fold", fold + 1)
                    _oof_part["Actual_MPI"] = np.asarray(y_np[te_idx]).ravel()
                    _oof_part["Predicted_MPI"] = np.asarray(y_fold_pred).ravel()
                    cv_oof_parts.append(_oof_part)
            cv_df = pd.DataFrame(cv_fold_rows)
            cv_oof_df = (
                pd.concat(cv_oof_parts, ignore_index=True)
                if cv_oof_parts
                else pd.DataFrame()
            )
            st.subheader(f"📊 {n_folds}-Fold GroupKFold CV (by Country)")
            st.dataframe(cv_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}))
            summary = cv_df[["MAE", "RMSE", "R²"]].agg(["mean", "std"])
            st.write(
                f"Mean ± Std — MAE: {summary.loc['mean','MAE']:.4f} ± {summary.loc['std','MAE']:.4f} | "
                f"RMSE: {summary.loc['mean','RMSE']:.4f} ± {summary.loc['std','RMSE']:.4f} | "
                f"R²: {summary.loc['mean','R²']:.4f} ± {summary.loc['std','R²']:.4f}"
            )
            _cv_download_buttons(cv_df, cv_oof_df, "dnn_now")

        with st.spinner("Training the model..."):
            y_val, y_pred_dnn, history, mae, rmse, r2 = train_dnn_model(
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
                scaler_choice,
            )
        st.success("Training completed!")
        # Save results in session state
        st.session_state["dnn_results"] = {
            "y_val": y_val,
            "y_pred": y_pred_dnn,
            "validation_results": _make_validation_results(ids_val, y_val, y_pred_dnn),
            "history": history,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "cv_df": cv_df,
            "cv_oof": cv_oof_df,
            "n_folds": n_folds,
        }
        st.subheader("📊 Model Performance")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")
        st.download_button(
            "Download Actual vs Predicted CSV",
            data=st.session_state["dnn_results"]["validation_results"]
            .to_csv(index=False)
            .encode("utf-8"),
            file_name="dnn_validation_results.csv",
            mime="text/csv",
            key="dnn_download_current_results",
        )
        # metrics of the last few epochs
        st.write("### Epochs")
        st.write(pd.DataFrame(history))
        st.subheader("Training and Validation Loss Curve")

        plot_loss_curve(history)
        plot_results(y_val, y_pred_dnn)

        st.subheader("Residual Plot (Error Analysis)")
        plot_residuals(y_val, y_pred_dnn)

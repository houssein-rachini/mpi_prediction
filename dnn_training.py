import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GroupKFold, TimeSeriesSplit
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
from typing import List, Tuple, Optional, Iterable

# =========================
# Default DNN architecture (same as ensemble)
# =========================
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


# =========================
# Helpers for metrics & bins
# =========================
def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _make_bins(y: np.ndarray, quantiles=(0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> np.ndarray:
    """Quantile-based bins for stratification (same idea as ensemble)."""
    qs = np.quantile(y, quantiles)
    return np.clip(np.digitize(y, qs[1:-1], right=True), 0, len(qs) - 2)


# =========================
# Balanced Group K-Fold
# =========================
def _balanced_group_kfold(
    df_idx_len: int,
    groups: Iterable,
    n_splits: int = 5,
    y: Optional[np.ndarray] = None,
    y_bins: Optional[np.ndarray] = None,
    shuffle: bool = True,
    random_state: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[List[str]], np.ndarray]:
    """
    Returns (folds, fold_groups, fold_loads)
      - folds: list of (train_idx, val_idx) where indices are 0..df_idx_len-1
      - fold_groups: groups assigned to each fold
      - fold_loads: row counts per fold
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    idx = np.arange(df_idx_len)
    groups = np.asarray(groups)
    if len(np.unique(groups)) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} unique groups for {n_splits} folds."
        )

    tmp = pd.DataFrame({"idx": idx, "grp": groups})
    if y_bins is not None:
        tmp["bin"] = np.asarray(y_bins)

    grp_indices = tmp.groupby("grp")["idx"].apply(np.array)
    grp_sizes = grp_indices.apply(len)
    ordered_groups = grp_sizes.sort_values(ascending=False).index.to_numpy()

    # tie-breaker jitter for equal sizes
    if shuffle:
        rng = np.random.default_rng(random_state)
        sizes = grp_sizes.loc[ordered_groups].to_numpy()
        jitter = rng.normal(scale=1e-9, size=len(ordered_groups))
        order = np.lexsort((jitter, -sizes))
        ordered_groups = ordered_groups[order]

    if y_bins is not None:
        max_bin = int(tmp["bin"].max())
        grp_bin_hist = tmp.groupby("grp")["bin"].apply(
            lambda s: np.bincount(s.astype(int), minlength=max_bin + 1)
        )
    else:
        grp_bin_hist = None

    fold_loads = np.zeros(n_splits, dtype=int)
    fold_groups: List[List[str]] = [[] for _ in range(n_splits)]
    fold_bin_hist = [
        np.zeros_like(grp_bin_hist.iloc[0]) if grp_bin_hist is not None else None
        for _ in range(n_splits)
    ]

    for g in ordered_groups:
        gsize = int(grp_sizes.loc[g])
        if grp_bin_hist is None:
            k = int(np.argmin(fold_loads))  # lightest fold
        else:
            gh = grp_bin_hist.loc[g]
            costs = []
            for i in range(n_splits):
                load_cost = fold_loads[i] + gsize
                bh = fold_bin_hist[i] + gh
                share = bh / max(bh.sum(), 1)
                bal_cost = float(np.var(share))
                costs.append((load_cost, bal_cost, i))
            costs.sort(key=lambda t: (t[0], t[1]))
            k = costs[0][2]

        fold_groups[k].append(g)
        fold_loads[k] += gsize
        if grp_bin_hist is not None:
            fold_bin_hist[k] += grp_bin_hist.loc[g]

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_splits):
        val_g = set(fold_groups[k])
        val_idx = tmp[tmp["grp"].isin(val_g)]["idx"].to_numpy()
        trn_idx = tmp[~tmp["grp"].isin(val_g)]["idx"].to_numpy()
        folds.append((trn_idx, val_idx))
    return folds, fold_groups, fold_loads


def _summarize_folds(
    groups: Iterable, folds: List[Tuple[np.ndarray, np.ndarray]]
) -> pd.DataFrame:
    """Fold summary table (fold, group, rows, fold_rows)."""
    parts = []
    groups = np.asarray(groups)
    for k, (_, va) in enumerate(folds):
        parts.append(pd.DataFrame({"fold": k, "group": groups[va]}))
    s = pd.concat(parts)
    table = (
        s.groupby(["fold", "group"])
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["fold", "rows"], ascending=[True, False])
    )
    totals = table.groupby("fold")["rows"].sum().rename("fold_rows").reset_index()
    return table.merge(totals, on="fold")


# =========================
# DNN model builder
# =========================
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

    # Optimizer
    if optimizer_choice == "AdamW":
        optimizer = AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)
    elif optimizer_choice == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif optimizer_choice == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    elif optimizer_choice == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
    else:
        optimizer = AdamW(learning_rate=lr_schedule, weight_decay=weight_decay)

    # Loss
    if loss_function_choice == "Huber":
        loss = Huber(delta=huber_delta if huber_delta is not None else 1.0)
    elif loss_function_choice == "Mean Squared Error":
        loss = "mse"
    elif loss_function_choice == "Mean Absolute Error":
        loss = "mae"
    else:
        loss = Huber(delta=1.0)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


# =========================
# Single-split training
# =========================
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
    """Trains a DNN model on one train/val split and saves the model and scaler."""
    # Scaling
    if scaler_choice == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Model
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

    # Metrics
    y_pred_dnn = dnn_model.predict(X_val_scaled).flatten()
    mae = mean_absolute_error(y_val, y_pred_dnn)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_dnn))
    r2 = r2_score(y_val, y_pred_dnn)

    # Save
    joblib.dump(scaler, "dnn_scaler.pkl")
    dnn_model.save("trained_dnn_model.h5")
    st.write("✅ Model and Scaler saved to 'trained_dnn_model.h5' and 'dnn_scaler.pkl'")

    return y_val, y_pred_dnn, history.history, mae, rmse, r2


# =========================
# Cross-validation (4 strategies)
# =========================
def cross_validate_dnn_model(
    X: pd.DataFrame,
    y: pd.Series,
    cv_type: str,
    n_splits: int,
    epochs: int,
    initial_learning_rate: float,
    batch_size: int,
    early_stopping_patience: int,
    layers_config,
    weight_decay: float,
    optimizer_choice: str,
    loss_function_choice: str,
    huber_delta: float,
    scaler_choice: str,
    shuffle: bool = True,
    random_state: int = 42,
    groups=None,
    balance_groups: bool = False,
    stratify_bins: bool = False,
):
    """
    Cross-validation for DNN with the same 4 strategies as ensemble:
      - kfold
      - timeseries
      - groupkfold
      - balanced groupkfold (balance_groups + stratify_bins)
    """
    fold_info = None

    # choose splitter/splits
    if cv_type == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        splits = splitter.split(X)

    elif cv_type == "timeseries":
        splitter = TimeSeriesSplit(n_splits=n_splits)
        splits = splitter.split(X)

    elif cv_type == "groupkfold":
        assert groups is not None, "groups array is required for GroupKFold"

        if len(np.unique(groups)) < n_splits:
            raise ValueError(
                f"GroupKFold needs ≥ {n_splits} unique groups; got {len(np.unique(groups))}."
            )

        if balance_groups:
            y_bins = _make_bins(y.to_numpy()) if stratify_bins else None
            folds, fold_groups, fold_loads = _balanced_group_kfold(
                df_idx_len=len(X),
                groups=groups,
                n_splits=n_splits,
                y=y.to_numpy(),
                y_bins=y_bins,
                shuffle=shuffle,
                random_state=random_state,
            )
            splits = folds
            fold_info = {
                "fold_loads": fold_loads,
                "table": _summarize_folds(groups, folds),
            }
        else:
            splitter = GroupKFold(n_splits=n_splits)
            splits = splitter.split(X, y, groups=groups)

    else:
        raise ValueError("cv_type must be 'kfold', 'timeseries', or 'groupkfold'")

    rows = []

    # CV loop
    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # Scaling per fold
        if scaler_choice == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_choice == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        # New model for each fold
        dnn_model = create_dnn_model(
            X_tr_scaled.shape[1],
            layers_config,
            initial_learning_rate,
            weight_decay,
            optimizer_choice,
            loss_function_choice,
            huber_delta,
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        )

        dnn_model.fit(
            X_tr_scaled,
            y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_va_scaled, y_va),
            verbose=0,
            callbacks=[early_stopping],
        )

        y_pred_va = dnn_model.predict(X_va_scaled).flatten()
        mae = mean_absolute_error(y_va, y_pred_va)
        rmse = _rmse(y_va, y_pred_va)
        r2 = r2_score(y_va, y_pred_va)

        rows.append(
            {
                "Fold": fold,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    metrics_df = pd.DataFrame(rows)
    summary = {
        "MAE_mean": metrics_df["MAE"].mean(),
        "MAE_std": metrics_df["MAE"].std(ddof=1),
        "RMSE_mean": metrics_df["RMSE"].mean(),
        "RMSE_std": metrics_df["RMSE"].std(ddof=1),
        "R2_mean": metrics_df["R2"].mean(),
        "R2_std": metrics_df["R2"].std(ddof=1),
    }

    return metrics_df, summary, fold_info


# =========================
# Plot helpers
# =========================
def plot_loss_curve(history):
    """Plots the training vs validation loss curve."""
    fig, ax = plt.subplots()
    ax.plot(history["loss"], label="Training Loss")
    ax.plot(history["val_loss"], label="Validation Loss")
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
    plt.title("Actual vs Predicted MPI (DNN Model)")
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


# =========================
# Main DNN Training Tab
# =========================
def show_dnn_training_tab(df):
    """Displays the UI for training the deep learning model."""
    st.title("🧠 Deep Learning Model Training")

    # Previous results
    if "dnn_results" in st.session_state:
        st.subheader("📊 Previous Training Results")
        results = st.session_state["dnn_results"]
        st.write(f"**Mean Absolute Error (MAE):** {results['mae']:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {results['rmse']:.4f}")
        st.write(f"**R² Score:** {results['r2']:.4f}")
        st.write("### Epoch History")
        st.write(pd.DataFrame(results["history"]))
        plot_loss_curve(results["history"])
        plot_results(results["y_val"], results["y_pred"])
        plot_residuals(results["y_val"], results["y_pred"])

    # ====== Data & features ======
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "Year" in numeric_cols:
        numeric_cols.remove("Year")

    target_col = "MPI"
    if target_col not in df.columns:
        st.error("MPI column not found in the dataset.")
        return

    # Ensure target not in candidate features
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    default_cols = [
        "Mean_NTL",
        "StdDev_NTL",
        "Sum_NTL",
        "Mean_GPP",
        "StdDev_GPP",
        "Median_Pop",
        "StdDev_Pop",
        "Mean_LST",
        "StdDev_LST",
        "Median_NDVI",
        "StdDev_NDVI",
        "ndvi_lst_ratio",
    ]
    default_cols = [c for c in default_cols if c in numeric_cols]

    selected_features = st.multiselect(
        "Select features for training:", numeric_cols, default=default_cols
    )

    # Build cleaned subset: keep all columns so Year/Region/etc are still available
    df_clean = df.dropna(subset=[target_col] + selected_features)
    X = df_clean[selected_features]
    y = np.maximum(df_clean[target_col], 0)

    # ====== Hyperparameters ======
    epochs = st.slider("Number of Epochs", 10, 500, 200, key="dnn_epochs")

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

    # Loss function
    loss_function_choice = st.selectbox(
        "Select Loss Function",
        ["Huber", "Mean Squared Error", "Mean Absolute Error"],
        key="loss_function",
    )

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
        "Early Stopping Patience", 5, 1000, 20, key="patience"
    )

    # ====== Architecture (same style as ensemble) ======
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

    # Adjust stored config length if user changes number of layers
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

    # ====== Cross-validation (4 strategies) ======
    st.subheader("Validation / Cross-Validation (same strategies as ensemble)")
    use_cv = st.checkbox("Use cross-validation", value=False)

    if use_cv:
        cv_type = st.selectbox(
            "CV type",
            ["kfold", "timeseries", "groupkfold"],
            index=0,
            key="dnn_cv_type",
        )
        n_splits = st.slider("Number of folds", 3, 10, 5, key="dnn_cv_folds")

        groups = None
        balance_groups = False
        stratify_bins = False
        X_cv = X.copy()
        y_cv = y.copy()
        df_ts = None

        if cv_type == "groupkfold":
            group_candidates = [c for c in df_clean.columns if c != target_col]
            group_col = st.selectbox(
                "Grouping column (e.g., Country / Region)",
                options=group_candidates,
                key="dnn_group_col",
            )
            groups = df_clean[group_col].to_numpy()

            balance_groups = st.checkbox(
                "Balance folds by row count (keep groups intact)", value=True
            )
            stratify_bins = st.checkbox(
                "Also stratify target bins (uses quantiles of y)", value=True
            )

        if cv_type == "timeseries":
            time_candidates = [
                c for c in df_clean.columns if c.lower() in ("year", "date", "time")
            ] or list(df_clean.columns)
            time_col = st.selectbox(
                "Time column for ordering", options=time_candidates, key="dnn_time_col"
            )
            df_ts = df_clean.sort_values(time_col)
            X_cv = df_ts[selected_features]
            y_cv = np.maximum(df_ts[target_col], 0)

        if st.button("Run Cross-Validation", key="dnn_cv_button"):
            with st.spinner("Running cross-validation..."):
                metrics_df, summary, fold_info = cross_validate_dnn_model(
                    X=X_cv,
                    y=y_cv,
                    cv_type=cv_type,
                    n_splits=n_splits,
                    epochs=epochs,
                    initial_learning_rate=initial_learning_rate,
                    batch_size=batch_size,
                    early_stopping_patience=early_stopping_patience,
                    layers_config=st.session_state.dnn_layers_config,
                    weight_decay=weight_decay,
                    optimizer_choice=optimizer_choice,
                    loss_function_choice=loss_function_choice,
                    huber_delta=huber_delta,
                    scaler_choice=scaler_choice,
                    shuffle=True,
                    random_state=42,
                    groups=groups if cv_type == "groupkfold" else None,
                    balance_groups=(
                        balance_groups if cv_type == "groupkfold" else False
                    ),
                    stratify_bins=(stratify_bins if cv_type == "groupkfold" else False),
                )

            st.success("Cross-validation complete!")

            if fold_info is not None:
                st.write("### Fold balance (rows per fold)")
                st.write(list(map(int, fold_info["fold_loads"])))
                st.dataframe(fold_info["table"])

            st.write("### Per-fold metrics")
            st.dataframe(
                metrics_df.style.format(
                    {"MAE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}"}
                )
            )

            st.write("### Summary (mean ± std)")
            st.write(
                f"MAE: {summary['MAE_mean']:.4f} ± {summary['MAE_std']:.4f} | "
                f"RMSE: {summary['RMSE_mean']:.4f} ± {summary['RMSE_std']:.4f} | "
                f"R²: {summary['R2_mean']:.4f} ± {summary['R2_std']:.4f}"
            )

    # ====== Final train/val split for model saving ======
    st.subheader("Train Final Model (80/20 split)")
    if st.button("Train Model", key="dnn_train_button"):
        with st.spinner("Training the model..."):
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            y_val_out, y_pred_dnn, history, mae, rmse, r2 = train_dnn_model(
                X_train,
                X_val,
                y_train,
                y_val,
                epochs,
                initial_learning_rate,
                batch_size,
                early_stopping_patience,
                st.session_state.dnn_layers_config,
                weight_decay,
                optimizer_choice,
                loss_function_choice,
                huber_delta,
                scaler_choice,
            )

        st.success("Training completed!")
        st.session_state["dnn_results"] = {
            "y_val": y_val_out,
            "y_pred": y_pred_dnn,
            "history": history,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

        st.subheader("📊 Model Performance (Hold-out Validation)")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")

        st.write("### Epoch History")
        st.write(pd.DataFrame(history))

        st.subheader("Training and Validation Loss Curve")
        plot_loss_curve(history)

        st.subheader("Actual vs Predicted")
        plot_results(y_val_out, y_pred_dnn)

        st.subheader("Residual Plot (Error Analysis)")
        plot_residuals(y_val_out, y_pred_dnn)

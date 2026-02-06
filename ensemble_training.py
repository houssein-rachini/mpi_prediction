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
from tensorflow.keras.losses import Huber
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import joblib
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, TimeSeriesSplit

# NEW: SHAP import
import shap

# -------------- DNN default --------------
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


# ---------------- Utils -----------------
def _rmse(y_true, y_pred, sample_weight=None):
    return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))


def _pick_loss(loss_function_choice, huber_delta):
    if loss_function_choice == "Huber":
        return tf.keras.losses.Huber(delta=huber_delta)
    elif loss_function_choice == "Mean Squared Error":
        return tf.keras.losses.MeanSquaredError()
    elif loss_function_choice == "Mean Absolute Error":
        return tf.keras.losses.MeanAbsoluteError()
    else:
        return tf.keras.losses.Huber(delta=huber_delta)


# -------- DNN builder ----------
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
    loss = _pick_loss(loss_function_choice, huber_delta)

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


# -------- SHAP helpers ----------
def _tree_shap(model, X_sample):
    expl = shap.TreeExplainer(model)
    vals = expl.shap_values(X_sample)  # (n, p)
    return vals


def _dnn_shap(dnn_model, X_background, X_sample):
    """Try DeepExplainer first; fallback to KernelExplainer."""
    try:
        expl = shap.DeepExplainer(dnn_model, X_background)
        vals = expl.shap_values(X_sample)[0]
        return vals
    except Exception:
        # Fallback: slower but robust
        f = lambda data: dnn_model.predict(data, verbose=0).flatten()
        expl = shap.KernelExplainer(f, X_background[:50])
        vals = expl.shap_values(X_sample, nsamples=100)[0]
        return vals


def _knn_shap(knn_model, X_background, X_sample):
    f = lambda data: knn_model.predict(data)
    expl = shap.KernelExplainer(f, X_background[:50])
    vals = expl.shap_values(X_sample, nsamples=100)[0]
    return vals


def compute_ensemble_shap(
    dnn_model,
    base_model_instance,
    base_model_name,
    X_train_scaled,
    X_val_scaled,
    feature_names,
    alpha=0.4,
    random_state=0,
    max_val_points=600,
    bg_size=200,
):
    """
    Returns:
      shap_ens (n, p), shap_dnn (n, p), shap_base (n, p), Xv_sample (DataFrame)
    """
    rng = np.random.RandomState(random_state)

    # Background for DNN/Kernel
    bg_sz = min(bg_size, len(X_train_scaled))
    bg_idx = rng.choice(len(X_train_scaled), size=bg_sz, replace=False)
    X_bg = X_train_scaled[bg_idx]

    # Subsample validation for SHAP speed
    n_val = len(X_val_scaled)
    if n_val == 0:
        return None, None, None, None
    sample_sz = min(max_val_points, n_val)
    val_idx = rng.choice(n_val, size=sample_sz, replace=False)
    Xv = X_val_scaled[val_idx]

    # Compute per-model SHAP
    # DNN
    shap_dnn = _dnn_shap(dnn_model, X_bg, Xv)

    # Base
    if base_model_name == "XGBoost" or isinstance(
        base_model_instance, xgb.XGBRegressor
    ):
        shap_base = _tree_shap(base_model_instance, Xv)
    elif base_model_name == "Random Forest" or isinstance(
        base_model_instance, RandomForestRegressor
    ):
        shap_base = _tree_shap(base_model_instance, Xv)
    elif base_model_name == "KNN Regressor" or isinstance(
        base_model_instance, KNeighborsRegressor
    ):
        shap_base = _knn_shap(base_model_instance, X_bg, Xv)
    else:
        # default to Kernel on base if unknown
        f = lambda data: base_model_instance.predict(data)
        expl = shap.KernelExplainer(f, X_bg[:50])
        shap_base = expl.shap_values(Xv, nsamples=100)[0]

    # Combine
    # --- Ensure equal number of rows before combining ---
    # Defensive guards in case one explainer produced fewer samples
    n_dnn = shap_dnn.shape[0] if shap_dnn is not None else 0
    n_base = shap_base.shape[0] if shap_base is not None else 0
    n_val = Xv.shape[0] if "Xv" in locals() else 0
    min_n = (
        min([n for n in [n_dnn, n_base, n_val] if n > 0])
        if (n_dnn and n_base and n_val)
        else 0
    )

    if min_n == 0:
        # nothing valid computed → return empty placeholders
        shap_ens = shap_dnn = shap_base = np.empty((0, len(feature_names)))
        Xv_df = pd.DataFrame(columns=feature_names)
        return shap_ens, shap_dnn, shap_base, Xv_df

    # Trim everything to equal length
    shap_dnn = shap_dnn[:min_n]
    shap_base = shap_base[:min_n]
    Xv = Xv[:min_n]

    # Combine ensemble SHAP
    shap_ens = alpha * shap_dnn + (1 - alpha) * shap_base

    # Create dataframe for plotting
    Xv_df = pd.DataFrame(Xv, columns=feature_names).reset_index(drop=True)

    # Final sanity check
    assert (
        shap_ens.shape[0] == Xv_df.shape[0]
    ), f"Row mismatch after trimming: SHAP {shap_ens.shape[0]} vs X {Xv_df.shape[0]}"

    return shap_ens, shap_dnn, shap_base, Xv_df


# -------- Plot helpers ----------
def plot_loss_curve(history):
    fig, ax = plt.subplots()
    ax.plot(history["loss"], label="DNN Training Loss")
    ax.plot(history["val_loss"], label="DNN Validation Loss")
    ax.plot(history["ensemble_train_loss"], label="Ensemble Training Loss")
    ax.plot(history["ensemble_val_loss"], label="Ensemble Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss Curve")
    ax.legend()
    st.pyplot(fig)


def plot_results(y_val, y_pred):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(x=y_val, y=y_pred, alpha=0.7, ax=ax)
    ax.axline((0, 0), slope=1, linestyle="--")
    ax.set_xlabel("Actual MPI")
    ax.set_ylabel("Predicted MPI")
    ax.set_title("Actual vs Predicted MPI (Ensemble)")
    st.pyplot(fig)


def plot_residuals(y_val, y_pred):
    residuals = y_val - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_val, y=residuals, alpha=0.7, ax=ax)
    ax.axhline(y=0, linestyle="--")
    ax.set_xlabel("Actual MPI")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Plot")
    st.pyplot(fig)


def plot_shap_global_bar(shap_vals, feature_names, title="Ensemble SHAP (mean |SHAP|)"):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(-mean_abs)[:25]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[order][::-1], mean_abs[order][::-1])
    ax.set_title(title)
    ax.set_xlabel("mean |SHAP value|")
    st.pyplot(fig)


def plot_shap_beeswarm(shap_vals, X_df, title="Ensemble SHAP Beeswarm"):
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_vals, X_df, show=False, max_display=25)
    plt.title(title)
    st.pyplot(fig)


def plot_shap_dependence(shap_vals, X_df, top_k=3):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(-mean_abs)[:top_k]
    for j in order:
        feat = X_df.columns[j]
        # color by the most interacting partner (simple: the next best feature)
        partner = (
            X_df.columns[order[0]]
            if order[0] != j
            else (X_df.columns[order[1]] if len(order) > 1 else None)
        )
        fig = plt.figure(figsize=(7, 5))
        shap.dependence_plot(
            feat,
            shap_vals,
            X_df,
            interaction_index=partner if partner is not None else "auto",
            show=False,
        )
        plt.title(f"Dependence: {feat} (color: {partner})")
        st.pyplot(fig)


# -------- Training with SHAP ----------
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
    save_models=True,
    ids_val=None,
    compute_shap=False,  # NEW
    shap_max_val_points=600,  # NEW
    shap_bg_size=200,  # NEW
    shap_random_state=0,  # NEW
):
    # --- Scaling ---
    if scaler_choice == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # --- Base model ---
    if base_model == "XGBoost":
        base_model_instance = xgb.XGBRegressor(**base_model_params)
    elif base_model == "Random Forest":
        base_model_instance = RandomForestRegressor(**base_model_params)
    elif base_model == "KNN Regressor":
        base_model_instance = KNeighborsRegressor(**base_model_params)
    else:
        raise ValueError("base_model must be XGBoost / Random Forest / KNN Regressor")
    base_model_instance.fit(X_train_scaled, y_train)

    # --- DNN ---
    dnn_model = create_dnn_model(
        X_train_scaled.shape[1],
        layers_config,
        initial_learning_rate,
        weight_decay,
        optimizer_choice,
        loss_function_choice,
        huber_delta,
    )

    history = {
        "loss": [],
        "val_loss": [],
        "ensemble_train_loss": [],
        "ensemble_val_loss": [],
    }
    patience_counter = 0
    best_val_loss = float("inf")
    best_weights = dnn_model.get_weights()

    for epoch in range(epochs):
        hist = dnn_model.fit(
            X_train_scaled,
            y_train,
            epochs=1,
            batch_size=batch_size,
            validation_data=(X_val_scaled, y_val),
            verbose=1,
        )

        loss_fn = _pick_loss(loss_function_choice, huber_delta)

        # Predictions
        y_pred_dnn_train = dnn_model.predict(X_train_scaled, verbose=0).flatten()
        y_pred_dnn_val = dnn_model.predict(X_val_scaled, verbose=0).flatten()
        y_pred_base_train = base_model_instance.predict(X_train_scaled)
        y_pred_base_val = base_model_instance.predict(X_val_scaled)
        y_pred_ens_train = alpha * y_pred_dnn_train + (1 - alpha) * y_pred_base_train
        y_pred_ens_val = alpha * y_pred_dnn_val + (1 - alpha) * y_pred_base_val

        # Losses
        dnn_train_loss = float(loss_fn(y_train, y_pred_dnn_train).numpy())
        dnn_val_loss = float(loss_fn(y_val, y_pred_dnn_val).numpy())
        ens_train_loss = float(loss_fn(y_train, y_pred_ens_train).numpy())
        ens_val_loss = float(loss_fn(y_val, y_pred_ens_val).numpy())

        history["loss"].append(dnn_train_loss)
        history["val_loss"].append(dnn_val_loss)
        history["ensemble_train_loss"].append(ens_train_loss)
        history["ensemble_val_loss"].append(ens_val_loss)

        # Early stopping on ensemble val loss
        if ens_val_loss < best_val_loss:
            best_val_loss = ens_val_loss
            best_weights = dnn_model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    # Restore best DNN weights
    dnn_model.set_weights(best_weights)

    # --- Final VAL predictions ---
    y_pred_ensemble = alpha * dnn_model.predict(X_val_scaled, verbose=0).flatten() + (
        1 - alpha
    ) * base_model_instance.predict(X_val_scaled)

    mae = mean_absolute_error(y_val, y_pred_ensemble)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))
    r2 = r2_score(y_val, y_pred_ensemble)

    # Save models
    if save_models:
        joblib.dump(scaler, "ensemble_scaler.pkl")
        if base_model == "XGBoost":
            base_model_instance.save_model("trained_ensemble_xgb_model.json")
            dnn_model.save("trained_ensemble_xgb_dnn_model.h5")
        elif base_model == "Random Forest":
            joblib.dump(base_model_instance, "trained_ensemble_rf_model.pkl")
            dnn_model.save("trained_ensemble_rf_dnn_model.h5")
        elif base_model == "KNN Regressor":
            joblib.dump(base_model_instance, "trained_ensemble_knn_model.pkl")
            dnn_model.save("trained_ensemble_knn_dnn_model.h5")

    # Cache results
    st.session_state["ensemble_results"] = {
        "y_val": y_val,
        "y_pred": y_pred_ensemble,
        "history": history,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }

    # Save validation results
    val_results = pd.DataFrame(
        {
            "Actual_MPI": pd.Series(y_val).reset_index(drop=True),
            "Predicted_MPI": pd.Series(y_pred_ensemble).reset_index(drop=True),
        }
    )
    if ids_val is not None and not ids_val.empty:
        val_results = pd.concat([ids_val.reset_index(drop=True), val_results], axis=1)
    val_results.to_csv("ensemble_validation_results.csv", index=False)

    # ---- SHAP: optional insight mode ----
    shap_payload = None
    if compute_shap:
        feature_names = list(X_train.columns)
        shap_ens, shap_dnn, shap_base, Xv_df = compute_ensemble_shap(
            dnn_model=dnn_model,
            base_model_instance=base_model_instance,
            base_model_name=base_model,
            X_train_scaled=X_train_scaled,
            X_val_scaled=X_val_scaled,
            feature_names=feature_names,
            alpha=alpha,
            random_state=shap_random_state,
            max_val_points=shap_max_val_points,
            bg_size=shap_bg_size,
        )
        if shap_ens is not None:
            shap_payload = {
                "shap_ensemble": shap_ens,
                "shap_dnn": shap_dnn,
                "shap_base": shap_base,
                "Xv_df": Xv_df,
                "feature_names": feature_names,
            }

    return y_val, y_pred_ensemble, history, mae, rmse, r2, shap_payload


# -------- Cross-Validation ----------
def cross_validate_ensemble(
    X,
    y,
    cv_type="kfold",
    n_splits=5,
    shuffle=True,
    random_state=42,
    groups=None,
    sample_weights=None,
    # training params
    epochs=200,
    initial_learning_rate=5e-4,
    batch_size=128,
    early_stopping_patience=20,
    layers_config=None,
    weight_decay=0.0,
    optimizer_choice="AdamW",
    loss_function_choice="Huber",
    huber_delta=1.0,
    alpha=0.4,
    base_model="XGBoost",
    base_model_params=None,
    scaler_choice="StandardScaler",
    # NEW: SHAP options
    compute_shap_last_fold=False,
    shap_max_val_points=400,
    shap_bg_size=150,
    shap_random_state=0,
):
    if layers_config is None:
        layers_config = DEFAULT_LAYERS
    if base_model_params is None:
        base_model_params = {}

    # splitter
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

        # 🔹 NEW: randomize group order (so folds differ) using the function's shuffle/random_state
        unique_groups = np.unique(groups)
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(unique_groups)
        # map original group labels to shuffled indices
        group_to_shuffled = {g: i for i, g in enumerate(unique_groups)}
        shuffled_group_indices = np.array([group_to_shuffled[g] for g in groups])

        splitter = GroupKFold(n_splits=n_splits)
        # use shuffled_group_indices instead of raw groups
        splits = splitter.split(X, y, groups=shuffled_group_indices)

    else:
        raise ValueError("cv_type must be 'kfold', 'timeseries', or 'groupkfold'")

    rows, histories, preds, val_indices = [], [], [], []
    shap_last_payload = None

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        # only compute SHAP on the last fold if requested (keeps CV fast)
        compute_shap_flag = compute_shap_last_fold and (fold == n_splits)

        _, y_pred_va, hist, mae, rmse, r2, shap_payload = train_ensemble_model(
            X_tr,
            X_va,
            y_tr,
            y_va,
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
            save_models=False,
            ids_val=None,
            compute_shap=compute_shap_flag,
            shap_max_val_points=shap_max_val_points,
            shap_bg_size=shap_bg_size,
            shap_random_state=shap_random_state,
        )

        if sample_weights is not None:
            w_va = (
                sample_weights.iloc[va_idx]
                if hasattr(sample_weights, "iloc")
                else sample_weights[va_idx]
            )
            w_mae = mean_absolute_error(y_va, y_pred_va, sample_weight=w_va)
            w_rmse = _rmse(y_va, y_pred_va, sample_weight=w_va)
        else:
            w_mae, w_rmse = np.nan, np.nan

        rows.append(
            {
                "Fold": fold,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "W_MAE": None if np.isnan(w_mae) else w_mae,
                "W_RMSE": None if np.isnan(w_rmse) else w_rmse,
            }
        )
        histories.append(hist)
        preds.append(y_pred_va)
        val_indices.append(va_idx)

        if compute_shap_flag and shap_payload is not None:
            shap_last_payload = shap_payload

    metrics_df = pd.DataFrame(rows)
    summary = {
        "MAE_mean": metrics_df["MAE"].mean(),
        "MAE_std": metrics_df["MAE"].std(ddof=1),
        "RMSE_mean": metrics_df["RMSE"].mean(),
        "RMSE_std": metrics_df["RMSE"].std(ddof=1),
        "R2_mean": metrics_df["R2"].mean(),
        "R2_std": metrics_df["R2"].std(ddof=1),
    }

    if "W_MAE" in metrics_df:
        w_mae_series = metrics_df["W_MAE"].dropna()
        w_rmse_series = metrics_df["W_RMSE"].dropna()
        summary["W_MAE_mean"] = w_mae_series.mean() if not w_mae_series.empty else None
        summary["W_MAE_std"] = (
            w_mae_series.std(ddof=1) if len(w_mae_series) > 1 else None
        )
        summary["W_RMSE_mean"] = (
            w_rmse_series.mean() if not w_rmse_series.empty else None
        )
        summary["W_RMSE_std"] = (
            w_rmse_series.std(ddof=1) if len(w_rmse_series) > 1 else None
        )

    return metrics_df, summary, histories, preds, val_indices, shap_last_payload


# -------- UI Tab ----------
def show_ensemble_training_tab(df):
    st.title("📈 Ensemble Model Training + SHAP Insights")

    # Re-render saved results
    if "ensemble_results" in st.session_state:
        st.subheader("📊 Previous Training Results")
        results = st.session_state["ensemble_results"]
        st.write(f"**MAE:** {results['mae']:.4f}")
        st.write(f"**RMSE:** {results['rmse']:.4f}")
        st.write(f"**R²:** {results['r2']:.4f}")
        st.write("### Epochs")
        st.write(pd.DataFrame(results["history"]))
        plot_loss_curve(results["history"])
        plot_results(results["y_val"], results["y_pred"])
        plot_residuals(results["y_val"], results["y_pred"])

    # Data & split
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if "Year" in numeric_cols:
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
        "ndvi_lst_ratio",
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

    wanted_ids = ["Country", "Region", "Year"]
    id_cols_present = [c for c in wanted_ids if c in df.columns]
    df_ids = df.loc[df_selected.index, id_cols_present]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    ids_val = df_ids.loc[X_val.index]

    # Global hyperparams
    alpha = st.slider("Ensemble Weight α (DNN contribution)", 0.0, 1.0, 0.4)
    epochs = st.slider("Number of Epochs", 10, 600, 300, key="ensemble_epochs")
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
            "Weight Decay (AdamW)",
            min_value=0.0,
            max_value=1e-2,
            value=1e-6,
            step=1e-6,
            format="%.6f",
            key="ensemble_wd",
        )
    loss_function_choice = st.selectbox(
        "Select Loss Function",
        ["Huber", "Mean Squared Error", "Mean Absolute Error"],
        key="ensemble_loss_function",
    )
    huber_delta = 1.0
    if loss_function_choice == "Huber":
        huber_delta = st.number_input(
            "Huber Loss Delta",
            min_value=0.001,
            max_value=10.0,
            value=1.0,
            step=0.001,
            format="%.3f",
            key="ensemble_huber_delta",
        )
    batch_size = st.slider("Batch Size", 8, 1024, 128, key="ensemble_batch_size")
    early_stopping_patience = st.slider(
        "Early Stopping Patience", 5, 50, 20, key="ensemble_patience"
    )

    # DNN architecture
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
    st.session_state.layers_config = layers

    # Base Model
    st.subheader("Base Model")
    base_model = st.selectbox(
        "Select Base Model", ["XGBoost", "Random Forest", "KNN Regressor"]
    )
    base_model_params = {}
    if base_model == "XGBoost":
        base_model_params = {
            "learning_rate": st.slider(
                "XGB Learning Rate", 0.01, 0.5, 0.05, key="ensemble_xgb_learning_rate"
            ),
            "max_depth": st.slider(
                "XGB Max Depth", 3, 10, 6, key="ensemble_xgb_max_depth"
            ),
            "n_estimators": st.slider(
                "XGB Trees", 50, 500, 200, key="ensemble_xgb_n_estimators"
            ),
            "min_child_weight": st.slider(
                "XGB Min Child Weight", 1, 10, 2, key="ensemble_xgb_min_child_weight"
            ),
            "random_state": 42,
        }
    elif base_model == "Random Forest":
        base_model_params = {
            "n_estimators": st.slider("RF Trees", 50, 300, 150),
            "min_samples_split": st.slider("RF Min Samples Split", 2, 10, 2),
            "min_samples_leaf": st.slider("RF Min Samples Leaf", 1, 10, 1),
            "random_state": 42,
        }
    elif base_model == "KNN Regressor":
        base_model_params = {
            "n_neighbors": st.slider(
                "KNN Neighbors", 1, 20, 4, key="ensemble_knn_neighbors"
            ),
            "metric": st.selectbox(
                "KNN Distance Metric",
                ["manhattan", "euclidean", "minkowski"],
                key="ensemble_knn_metric",
            ),
        }

    # Validation / Training controls
    st.subheader("Validation Strategy")
    use_cv = st.checkbox(
        "Use cross-validation (recommended for robustness)", value=False
    )

    # Global SHAP toggles
    with st.expander("🔍 SHAP Insight Options"):
        do_shap_train = st.checkbox("Compute SHAP on train/val split", value=True)
        shap_max_pts = st.slider(
            "Max validation points for SHAP", 100, 5000, 600, step=100
        )
        shap_bg_sz = st.slider("Background size (DNN/Kernel)", 20, 2000, 200, step=20)
        shap_seed = st.number_input("SHAP random_state", value=0, step=1)

    if use_cv:
        cv_type = st.selectbox(
            "CV type", ["kfold", "timeseries", "groupkfold"], index=0
        )
        n_splits = st.slider("Number of folds", 3, 10, 5)

        groups = None
        if cv_type == "groupkfold":
            group_col = st.selectbox(
                "Grouping column (e.g., Country / Region)",
                options=[c for c in df.columns if c not in ["MPI"]],
            )
            groups = df.loc[df_selected.index, group_col]

        if cv_type == "timeseries":
            time_candidates = [
                c for c in df.columns if c.lower() in ("year", "date", "time")
            ] or list(df.columns)
            time_col = st.selectbox("Time column for ordering", options=time_candidates)
            df_sorted = df_selected.join(df[[time_col]]).sort_values(time_col)
            X = df_sorted.drop(columns=["MPI", time_col])
            y = np.maximum(df_sorted["MPI"], 0)

        numeric_cols_full = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        weight_col = st.selectbox(
            "Weight column (optional, e.g., Total_Pop)",
            options=["<None>"] + numeric_cols_full,
            index=(
                (["<None>"] + numeric_cols_full).index("Total_Pop")
                if "Total_Pop" in numeric_cols_full
                else 0
            ),
        )
        weights = None
        if weight_col != "<None>":
            weights = (
                df.loc[df_sorted.index, weight_col]
                if cv_type == "timeseries"
                else df.loc[df_selected.index, weight_col]
            )

        compute_shap_last_fold = st.checkbox(
            "During CV, compute SHAP on the last fold only", value=True
        )

        if st.button("Run Cross-Validation", key="ensemble_cv_button"):
            with st.spinner("Running cross-validation..."):
                metrics_df, summary, histories, preds, val_idx, shap_payload = (
                    cross_validate_ensemble(
                        X,
                        y,
                        cv_type=cv_type,
                        n_splits=n_splits,
                        groups=groups if cv_type == "groupkfold" else None,
                        shuffle=True,
                        random_state=42,
                        epochs=epochs,
                        initial_learning_rate=initial_learning_rate,
                        batch_size=batch_size,
                        early_stopping_patience=early_stopping_patience,
                        layers_config=st.session_state.layers_config,
                        weight_decay=weight_decay,
                        optimizer_choice=optimizer_choice,
                        loss_function_choice=loss_function_choice,
                        huber_delta=huber_delta,
                        alpha=alpha,
                        base_model=base_model,
                        base_model_params=base_model_params,
                        scaler_choice=scaler_choice,
                        sample_weights=weights,
                        compute_shap_last_fold=compute_shap_last_fold,
                        shap_max_val_points=shap_max_pts,
                        shap_bg_size=shap_bg_sz,
                        shap_random_state=shap_seed,
                    )
                )

            st.success("Cross-validation complete!")
            st.write("### Per-fold metrics")
            st.dataframe(
                metrics_df.style.format(
                    {
                        "MAE": "{:.4f}",
                        "RMSE": "{:.4f}",
                        "R2": "{:.4f}",
                        "W_MAE": "{:.4f}",
                        "W_RMSE": "{:.4f}",
                    }
                )
            )

            st.write("### Summary (mean ± std)")
            base_line = (
                f"MAE: {summary['MAE_mean']:.4f} ± {summary['MAE_std']:.4f} | "
                f"RMSE: {summary['RMSE_mean']:.4f} ± {summary['RMSE_std']:.4f} | "
                f"R²: {summary['R2_mean']:.4f} ± {summary['R2_std']:.4f}"
            )
            if "W_MAE_mean" in summary and summary["W_MAE_mean"] is not None:
                w_line = (
                    f" | W-MAE: {summary['W_MAE_mean']:.4f}"
                    f"{'' if summary['W_MAE_std'] is None else f' ± {summary['W_MAE_std']:.4f}'}"
                    f" | W-RMSE: {summary['W_RMSE_mean']:.4f}"
                    f"{'' if summary['W_RMSE_std'] is None else f' ± {summary['W_RMSE_std']:.4f}'}"
                )
                st.write(base_line + w_line)
            else:
                st.write(base_line)

            # SHAP plots for last fold (if computed)
            if shap_payload is not None:
                st.subheader("🔍 SHAP (last fold validation)")
                shap_ens = shap_payload["shap_ensemble"]
                Xv_df = shap_payload["Xv_df"]
                plot_shap_global_bar(
                    shap_ens, Xv_df.columns, title="Ensemble SHAP (CV last fold)"
                )
                plot_shap_beeswarm(
                    shap_ens, Xv_df, title="Ensemble SHAP Beeswarm (CV last fold)"
                )
                plot_shap_dependence(shap_ens, Xv_df, top_k=3)

    else:
        if st.button("Train Model", key="ensemble_train_button"):
            with st.spinner("Training the model..."):
                y_val_out, y_pred_ensemble, history, mae, rmse, r2, shap_payload = (
                    train_ensemble_model(
                        X_train,
                        X_val,
                        y_train,
                        y_val,
                        epochs,
                        initial_learning_rate,
                        batch_size,
                        early_stopping_patience,
                        st.session_state.layers_config,
                        weight_decay,
                        optimizer_choice,
                        loss_function_choice,
                        huber_delta,
                        alpha,
                        base_model,
                        base_model_params,
                        scaler_choice,
                        save_models=True,
                        ids_val=ids_val,
                        compute_shap=do_shap_train,
                        shap_max_val_points=shap_max_pts,
                        shap_bg_size=shap_bg_sz,
                        shap_random_state=shap_seed,
                    )
                )
            st.success("Training completed!")
            st.subheader("📊 Model Performance")
            st.write(f"**MAE:** {mae:.4f}")
            st.write(f"**RMSE:** {rmse:.4f}")
            st.write(f"**R²:** {r2:.4f}")
            st.write("### Epochs")
            st.write(pd.DataFrame(history))
            st.subheader("Training and Validation Loss Curve")
            plot_loss_curve(history)
            st.subheader("Actual vs Predicted Scatter Plot")
            plot_results(y_val_out, y_pred_ensemble)
            st.subheader("Residual Plot (Error Analysis)")
            plot_residuals(y_val_out, y_pred_ensemble)

            # SHAP plots
            if shap_payload is not None:
                st.subheader("🔍 SHAP Insights (validation subset)")
                shap_ens = shap_payload["shap_ensemble"]
                Xv_df = shap_payload["Xv_df"]
                plot_shap_global_bar(
                    shap_ens, Xv_df.columns, title="Ensemble SHAP (mean |SHAP|)"
                )
                plot_shap_beeswarm(shap_ens, Xv_df, title="Ensemble SHAP Beeswarm")
                plot_shap_dependence(shap_ens, Xv_df, top_k=3)

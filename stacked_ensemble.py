import json
import os
from copy import deepcopy

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from tensorflow.keras.callbacks import EarlyStopping

from dnn_training import create_dnn_model
from ensemble_training import DEFAULT_LAYERS as ENSEMBLE_DEFAULT_LAYERS


DEFAULT_FEATURES = [
    "Mean_GPP",
    "StdDev_GPP",
    "Median_Pop",
    "StdDev_Pop",
    "Mean_LST",
    "StdDev_LST",
    "Mean_NTL",
    "StdDev_NTL",
    "Sum_NTL",
    "Median_NDVI",
    "StdDev_NDVI",
    "ndvi_lst_ratio",
]

# Keep default DNN architecture aligned exactly with ensemble training tab
DEFAULT_STACKED_LAYERS = deepcopy(ENSEMBLE_DEFAULT_LAYERS)

BASE_MODEL_OPTIONS = [
    "XGBoost",
    "Random Forest",
    "Support Vector Regression",
    "KNN Regressor",
    "DNN+XGBoost Ensemble",
    "DNN+Random Forest Ensemble",
    "DNN+KNN Ensemble",
]

TRAINABLE_BASE_MODELS = [
    "XGBoost",
    "Random Forest",
    "Support Vector Regression",
    "KNN Regressor",
]


def _build_scaler(scaler_choice):
    if scaler_choice == "MinMaxScaler":
        return MinMaxScaler()
    return StandardScaler()


def _filter_by_correlation(X, threshold):
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]
    return X.drop(columns=to_drop, errors="ignore"), to_drop


def _metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def _plot_pred_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, ax=ax)
    minv = min(float(np.min(y_true)), float(np.min(y_pred)))
    maxv = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([minv, maxv], [minv, maxv], "--r")
    ax.set_xlabel("Actual MPI")
    ax.set_ylabel("Predicted MPI")
    ax.set_title("Stacked Ensemble: Predicted vs Actual (OOF)")
    st.pyplot(fig)


def _plot_dnn_learning_curves(histories):
    if not histories:
        return
    max_len = max(len(h.get("loss", [])) for h in histories)
    if max_len == 0:
        return
    train_arr = np.full((len(histories), max_len), np.nan)
    val_arr = np.full((len(histories), max_len), np.nan)
    for i, h in enumerate(histories):
        loss = np.array(h.get("loss", []), dtype=float)
        val_loss = np.array(h.get("val_loss", []), dtype=float)
        train_arr[i, : len(loss)] = loss
        val_arr[i, : len(val_loss)] = val_loss

    train_mean = np.nanmean(train_arr, axis=0)
    val_mean = np.nanmean(val_arr, axis=0)
    epochs = np.arange(1, max_len + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_mean, label="Train Loss (mean across folds)")
    ax.plot(epochs, val_mean, label="Val Loss (mean across folds)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("DNN Learning Curves (GroupKFold)")
    ax.legend()
    st.pyplot(fig)


def _plot_meta_weights(meta_weights_df):
    if meta_weights_df is None or meta_weights_df.empty:
        return
    plot_df = meta_weights_df.copy().sort_values("weight", key=np.abs, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=plot_df, x="weight", y="meta_feature", orient="h", ax=ax)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Coefficient")
    ax.set_ylabel("Meta Feature")
    ax.set_title("Ridge Meta-Learner Weights")
    st.pyplot(fig)


def _init_base_model(model_name, base_model_params):
    params = base_model_params.get(model_name, {})
    if model_name == "XGBoost":
        return xgb.XGBRegressor(**params)
    if model_name == "Random Forest":
        return RandomForestRegressor(**params)
    if model_name == "Support Vector Regression":
        return SVR(kernel="rbf", C=params["C"], gamma=params["gamma"])
    if model_name == "KNN Regressor":
        return KNeighborsRegressor(
            n_neighbors=params["n_neighbors"], metric=params["metric"]
        )
    raise ValueError(f"Unsupported base model: {model_name}")


def _ensure_single_output_layer(layers_config):
    layers = deepcopy(layers_config)
    if not layers:
        return [{"type": "Dense", "units": 1, "activation": "relu"}]

    last = layers[-1]
    if not (last.get("type") == "Dense" and int(last.get("units", -1)) == 1):
        layers.append({"type": "Dense", "units": 1, "activation": "relu"})
    return layers


def _build_meta_features(
    oof_map, meta_source_order, use_abs_diff, ensemble_feature_alphas=None
):
    if ensemble_feature_alphas is None:
        ensemble_feature_alphas = {}
    cols = []
    names = []

    for model_name in meta_source_order:
        if model_name == "DNN":
            if "DNN" not in oof_map:
                raise ValueError(
                    "DNN meta source selected but DNN predictions are unavailable."
                )
            cols.append(oof_map["DNN"])
            names.append("oof_dnn")
        elif model_name in TRAINABLE_BASE_MODELS:
            cols.append(oof_map[model_name])
            names.append(f"oof_{model_name.lower().replace(' ', '_')}")
        elif model_name == "DNN+XGBoost Ensemble":
            alpha = float(ensemble_feature_alphas.get(model_name, 0.4))
            cols.append(alpha * oof_map["DNN"] + (1.0 - alpha) * oof_map["XGBoost"])
            names.append("oof_dnn_xgboost_ensemble")
        elif model_name == "DNN+Random Forest Ensemble":
            alpha = float(ensemble_feature_alphas.get(model_name, 0.4))
            cols.append(
                alpha * oof_map["DNN"] + (1.0 - alpha) * oof_map["Random Forest"]
            )
            names.append("oof_dnn_random_forest_ensemble")
        elif model_name == "DNN+KNN Ensemble":
            alpha = float(ensemble_feature_alphas.get(model_name, 0.4))
            cols.append(
                alpha * oof_map["DNN"] + (1.0 - alpha) * oof_map["KNN Regressor"]
            )
            names.append("oof_dnn_knn_ensemble")
        else:
            raise ValueError(f"Unsupported meta source: {model_name}")

    if use_abs_diff and "DNN" in oof_map and "XGBoost" in oof_map:
        cols.append(np.abs(oof_map["DNN"] - oof_map["XGBoost"]))
        names.append("abs_diff_dnn_xgboost")

    if not cols:
        raise ValueError("No meta-features available. Select at least one base model.")

    return np.column_stack(cols), names


def predict_stacked(
    X,
    dnn_model,
    base_models,
    meta_model,
    scaler,
    base_model_order,
    meta_scaler=None,
    include_dnn=True,
    use_abs_diff=False,
    meta_source_order=None,
    ensemble_feature_alphas=None,
):
    X_scaled = scaler.transform(X)
    pred_map = {}
    if include_dnn:
        if dnn_model is None:
            raise ValueError("DNN model is missing while include_dnn=True.")
        dnn_pred = dnn_model.predict(X_scaled, verbose=0)
        if dnn_pred.ndim == 2 and dnn_pred.shape[1] != 1:
            raise ValueError("DNN output must be one unit for stacking inference.")
        pred_map["DNN"] = dnn_pred.reshape(-1)

    for model_name in base_model_order:
        pred_map[model_name] = base_models[model_name].predict(X_scaled)

    if meta_source_order is None:
        meta_source_order = base_model_order
    meta_input, _ = _build_meta_features(
        pred_map,
        meta_source_order,
        use_abs_diff,
        ensemble_feature_alphas=ensemble_feature_alphas,
    )
    if meta_scaler is not None:
        meta_input = meta_scaler.transform(meta_input)
    return meta_model.predict(meta_input)


def _train_stacked_models(
    X,
    y,
    groups,
    n_splits,
    corr_threshold,
    scaler_choice,
    dnn_params,
    base_model_order,
    meta_source_order,
    ensemble_feature_alphas,
    base_model_params,
    meta_model_choice,
    ridge_alpha,
    include_dnn,
    use_abs_diff,
    debug_log_fn=None,
):
    gkf = GroupKFold(n_splits=n_splits)
    n = len(X)
    oof_map = {}
    if include_dnn:
        oof_map["DNN"] = np.zeros(n, dtype=float)
    for model_name in base_model_order:
        oof_map[model_name] = np.zeros(n, dtype=float)

    seen = np.zeros(n, dtype=bool)
    fold_indices = []
    fold_sizes = []
    fold_rows = []
    fold_debug_rows = []
    dnn_histories = []

    dnn_layers = _ensure_single_output_layer(dnn_params["layers_config"])

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr_raw, X_va_raw = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        if debug_log_fn is not None:
            debug_log_fn(
                f"Fold {fold}/{n_splits}: train={len(tr_idx)} rows, val={len(va_idx)} rows"
            )

        # Strict CV: fit correlation filter on training fold only, then apply to validation fold.
        X_tr, fold_dropped = _filter_by_correlation(X_tr_raw, corr_threshold)
        X_va = X_va_raw.drop(columns=fold_dropped, errors="ignore")
        if X_tr.shape[1] == 0:
            raise ValueError(
                f"Fold {fold}: all features removed by correlation filter (threshold={corr_threshold})."
            )
        if debug_log_fn is not None and fold_dropped:
            debug_log_fn(
                f"Fold {fold}: dropped {len(fold_dropped)} correlated feature(s)"
            )

        scaler = _build_scaler(scaler_choice)
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        if include_dnn:
            dnn_model = create_dnn_model(
                input_dim=X_tr_scaled.shape[1],
                layers_config=dnn_layers,
                initial_learning_rate=dnn_params["initial_learning_rate"],
                weight_decay=dnn_params["weight_decay"],
                optimizer_choice=dnn_params["optimizer_choice"],
                loss_function_choice=dnn_params["loss_function_choice"],
                huber_delta=dnn_params["huber_delta"],
            )
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=dnn_params["early_stopping_patience"],
                restore_best_weights=True,
            )
            history = dnn_model.fit(
                X_tr_scaled,
                y_tr,
                epochs=dnn_params["epochs"],
                batch_size=dnn_params["batch_size"],
                validation_data=(X_va_scaled, y_va),
                callbacks=[early_stopping],
                verbose=0,
            )
            dnn_histories.append(history.history)
            fold_epochs_ran = len(history.history.get("loss", []))
            best_val_loss = (
                float(np.nanmin(history.history["val_loss"]))
                if history.history.get("val_loss")
                else np.nan
            )
            if debug_log_fn is not None:
                debug_log_fn(
                    f"Fold {fold}: DNN epochs_ran={fold_epochs_ran}, best_val_loss={best_val_loss:.6f}"
                )

            dnn_pred = dnn_model.predict(X_va_scaled, verbose=0)
            if dnn_pred.ndim == 2 and dnn_pred.shape[1] != 1:
                raise ValueError(
                    "DNN output dimension must be 1 for stacking. Ensure final Dense(1)."
                )
            oof_map["DNN"][va_idx] = dnn_pred.reshape(-1)

        for model_name in base_model_order:
            model = _init_base_model(model_name, base_model_params)
            model.fit(X_tr_scaled, y_tr)
            oof_map[model_name][va_idx] = model.predict(X_va_scaled)
            if debug_log_fn is not None:
                debug_log_fn(f"Fold {fold}: trained {model_name}")

        seen[va_idx] = True
        fold_indices.append(va_idx)
        fold_sizes.append((len(tr_idx), len(va_idx)))

    if not seen.all():
        raise ValueError("OOF predictions are incomplete; some rows were not scored.")

    meta_X, meta_feature_names = _build_meta_features(
        oof_map,
        meta_source_order,
        use_abs_diff,
        ensemble_feature_alphas=ensemble_feature_alphas,
    )
    meta_y = y.to_numpy()

    # Final meta model (trained on all meta features) for saving/inference.
    use_meta_scaling = meta_model_choice == "Ridge"
    meta_scaler = None
    meta_X_final = meta_X
    if use_meta_scaling:
        meta_scaler = StandardScaler()
        meta_X_final = meta_scaler.fit_transform(meta_X)

    if meta_model_choice == "LinearRegression":
        meta_model = LinearRegression()
    else:
        meta_model = Ridge(alpha=ridge_alpha, random_state=42)
    meta_model.fit(meta_X_final, meta_y)

    # Proper meta-level OOF evaluation: fit meta per fold, predict only that fold.
    meta_oof_pred = np.zeros_like(meta_y, dtype=float)
    for tr_idx, va_idx in gkf.split(X, y, groups=groups):
        meta_X_tr = meta_X[tr_idx]
        meta_X_va = meta_X[va_idx]
        if use_meta_scaling:
            fold_meta_scaler = StandardScaler()
            meta_X_tr = fold_meta_scaler.fit_transform(meta_X_tr)
            meta_X_va = fold_meta_scaler.transform(meta_X_va)

        if meta_model_choice == "LinearRegression":
            meta_fold_model = LinearRegression()
        else:
            meta_fold_model = Ridge(alpha=ridge_alpha, random_state=42)
        meta_fold_model.fit(meta_X_tr, meta_y[tr_idx])
        meta_oof_pred[va_idx] = meta_fold_model.predict(meta_X_va)

    overall = _metrics(meta_y, meta_oof_pred)

    for i, va_idx in enumerate(fold_indices, start=1):
        tr_size, va_size = fold_sizes[i - 1]
        m = _metrics(meta_y[va_idx], meta_oof_pred[va_idx])
        fold_rows.append({"fold": i, "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"]})
        fold_debug_row = {
            "fold": i,
            "train_rows": int(tr_size),
            "val_rows": int(va_size),
        }
        if include_dnn and i - 1 < len(dnn_histories):
            hist = dnn_histories[i - 1]
            fold_debug_row["dnn_epochs_ran"] = int(len(hist.get("loss", [])))
            fold_debug_row["dnn_best_val_loss"] = (
                float(np.nanmin(hist["val_loss"])) if hist.get("val_loss") else np.nan
            )
        fold_debug_rows.append(fold_debug_row)
    fold_df = pd.DataFrame(fold_rows)
    fold_debug_df = pd.DataFrame(fold_debug_rows)
    fold_avg = {
        "mae": float(fold_df["mae"].mean()),
        "rmse": float(fold_df["rmse"].mean()),
        "r2": float(fold_df["r2"].mean()),
    }

    # Final deployment feature set/scaler are fit on full data after CV.
    X_full_filtered, dropped_features = _filter_by_correlation(X, corr_threshold)
    if X_full_filtered.shape[1] == 0:
        raise ValueError(
            "All selected features were removed by correlation filtering on full data."
        )
    full_scaler = _build_scaler(scaler_choice)
    X_full_scaled = full_scaler.fit_transform(X_full_filtered)

    full_dnn = None
    if include_dnn:
        full_dnn = create_dnn_model(
            input_dim=X_full_scaled.shape[1],
            layers_config=dnn_layers,
            initial_learning_rate=dnn_params["initial_learning_rate"],
            weight_decay=dnn_params["weight_decay"],
            optimizer_choice=dnn_params["optimizer_choice"],
            loss_function_choice=dnn_params["loss_function_choice"],
            huber_delta=dnn_params["huber_delta"],
        )
        full_early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=dnn_params["early_stopping_patience"],
            restore_best_weights=True,
        )
        full_dnn.fit(
            X_full_scaled,
            y,
            epochs=dnn_params["epochs"],
            batch_size=dnn_params["batch_size"],
            validation_split=0.1,
            callbacks=[full_early_stopping],
            verbose=0,
        )

    full_base_models = {}
    for model_name in base_model_order:
        model = _init_base_model(model_name, base_model_params)
        model.fit(X_full_scaled, y)
        full_base_models[model_name] = model

    meta_weights_df = pd.DataFrame()
    if meta_model_choice == "Ridge" and hasattr(meta_model, "coef_"):
        meta_weights_df = pd.DataFrame(
            {
                "meta_feature": meta_feature_names,
                "weight": np.asarray(meta_model.coef_, dtype=float),
            }
        )

    return {
        "oof_true": meta_y,
        "oof_pred": meta_oof_pred,
        "overall": overall,
        "fold_df": fold_df,
        "fold_avg": fold_avg,
        "histories": dnn_histories,
        "dnn_model": full_dnn,
        "base_models": full_base_models,
        "meta_model": meta_model,
        "meta_scaler": meta_scaler,
        "scaler": full_scaler,
        "features_used": list(X_full_filtered.columns),
        "dropped_features_correlation": dropped_features,
        "meta_feature_names": meta_feature_names,
        "meta_weights_df": meta_weights_df,
        "fold_debug_df": fold_debug_df,
    }


def _save_stacked_artifacts(
    dnn_model,
    base_models,
    meta_model,
    meta_scaler,
    scaler,
    metadata,
):
    out_dir = os.path.join("models", "stacked")
    os.makedirs(out_dir, exist_ok=True)

    if dnn_model is not None:
        dnn_model.save(os.path.join(out_dir, "dnn_model.h5"))

    saved_model_files = {}
    if "XGBoost" in base_models:
        joblib.dump(base_models["XGBoost"], os.path.join(out_dir, "xgb_model.pkl"))
        saved_model_files["XGBoost"] = "xgb_model.pkl"
    if "Random Forest" in base_models:
        joblib.dump(base_models["Random Forest"], os.path.join(out_dir, "rf_model.pkl"))
        saved_model_files["Random Forest"] = "rf_model.pkl"
    if "Support Vector Regression" in base_models:
        joblib.dump(
            base_models["Support Vector Regression"],
            os.path.join(out_dir, "svr_model.pkl"),
        )
        saved_model_files["Support Vector Regression"] = "svr_model.pkl"
    if "KNN Regressor" in base_models:
        joblib.dump(
            base_models["KNN Regressor"], os.path.join(out_dir, "knn_model.pkl")
        )
        saved_model_files["KNN Regressor"] = "knn_model.pkl"

    joblib.dump(meta_model, os.path.join(out_dir, "meta_model.pkl"))
    if meta_scaler is not None:
        joblib.dump(meta_scaler, os.path.join(out_dir, "meta_scaler.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

    metadata = dict(metadata)
    metadata["saved_base_model_files"] = saved_model_files
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def show_stacking_tab(df):
    st.title("Stacked Ensemble")

    if "Country" not in df.columns:
        st.error("Country column is required for GroupKFold stacking.")
        return
    if "MPI" not in df.columns:
        st.error("MPI column is required as the target.")
        return

    numeric_cols = [
        c for c in df.select_dtypes(include=["number"]).columns if c != "MPI"
    ]
    default_features = [c for c in DEFAULT_FEATURES if c in numeric_cols]
    selected_features = st.multiselect(
        "Select features for training:",
        numeric_cols,
        default=(
            default_features
            if default_features
            else numeric_cols[: min(10, len(numeric_cols))]
        ),
        key="stacked_features",
    )
    if not selected_features:
        st.warning("Select at least one feature to continue.")
        return

    corr_threshold = st.slider(
        "Correlation threshold for feature filtering",
        min_value=0.70,
        max_value=0.99,
        value=0.95,
        step=0.01,
        key="stacked_corr_threshold",
    )

    st.subheader("Base Models")
    selected_meta_sources = st.multiselect(
        "Select base/meta sources:",
        BASE_MODEL_OPTIONS,
        default=BASE_MODEL_OPTIONS,
        key="stacked_base_models",
    )
    if not selected_meta_sources:
        st.warning("Select at least one base model.")
        return

    st.subheader("XGBoost Parameters")
    xgb_params = {
        "n_estimators": st.slider(
            "XGB Trees", 50, 800, 300, key="stacked_xgb_n_estimators"
        ),
        "learning_rate": st.slider(
            "XGB Learning Rate", 0.01, 0.5, 0.05, key="stacked_xgb_learning_rate"
        ),
        "max_depth": st.slider("XGB Max Depth", 2, 12, 6, key="stacked_xgb_max_depth"),
        "min_child_weight": st.slider(
            "XGB Min Child Weight", 1, 20, 2, key="stacked_xgb_min_child_weight"
        ),
        "subsample": st.slider(
            "XGB Subsample", 0.5, 1.0, 0.9, key="stacked_xgb_subsample"
        ),
        "colsample_bytree": st.slider(
            "XGB Colsample Bytree", 0.5, 1.0, 0.9, key="stacked_xgb_colsample_bytree"
        ),
        "random_state": 42,
    }

    st.subheader("Random Forest Parameters")
    rf_params = {
        "n_estimators": st.slider(
            "RF Trees", 50, 500, 200, key="stacked_rf_n_estimators"
        ),
        "min_samples_split": st.slider(
            "RF Min Samples Split", 2, 20, 2, key="stacked_rf_min_samples_split"
        ),
        "min_samples_leaf": st.slider(
            "RF Min Samples Leaf", 1, 20, 1, key="stacked_rf_min_samples_leaf"
        ),
        "random_state": 42,
    }

    st.subheader("SVR Parameters")
    svr_params = {
        "C": st.slider("SVR C", 1, 500, 100, key="stacked_svr_c"),
        "gamma": st.slider("SVR Gamma", 0.001, 1.0, 0.1, key="stacked_svr_gamma"),
    }

    st.subheader("KNN Parameters")
    knn_params = {
        "n_neighbors": st.slider(
            "KNN Neighbors", 1, 30, 5, key="stacked_knn_n_neighbors"
        ),
        "metric": st.selectbox(
            "KNN Distance Metric",
            ["manhattan", "euclidean", "minkowski"],
            key="stacked_knn_metric",
        ),
    }

    include_dnn = st.checkbox(
        "Include DNN base model",
        value=True,
        key="stacked_include_dnn",
    )
    ensemble_feature_alphas = {}
    if "DNN+XGBoost Ensemble" in selected_meta_sources:
        ensemble_feature_alphas["DNN+XGBoost Ensemble"] = st.slider(
            "DNN+XGBoost Ensemble Alpha (DNN contribution)",
            0.0,
            1.0,
            0.4,
            0.01,
            key="stacked_ens_alpha_xgb",
        )
    if "DNN+Random Forest Ensemble" in selected_meta_sources:
        ensemble_feature_alphas["DNN+Random Forest Ensemble"] = st.slider(
            "DNN+Random Forest Ensemble Alpha (DNN contribution)",
            0.0,
            1.0,
            0.4,
            0.01,
            key="stacked_ens_alpha_rf",
        )
    if "DNN+KNN Ensemble" in selected_meta_sources:
        ensemble_feature_alphas["DNN+KNN Ensemble"] = st.slider(
            "DNN+KNN Ensemble Alpha (DNN contribution)",
            0.0,
            1.0,
            0.4,
            0.01,
            key="stacked_ens_alpha_knn",
        )
    layers = deepcopy(DEFAULT_STACKED_LAYERS)
    epochs = 200
    batch_size = 128
    early_stopping_patience = 20
    optimizer_choice = "AdamW"
    initial_learning_rate = 0.001
    weight_decay = 1e-5
    loss_function_choice = "Huber"
    huber_delta = 0.1

    if include_dnn:
        st.subheader("DNN Parameters")
        epochs = st.slider("Number of Epochs", 10, 500, 200, key="stacked_dnn_epochs")
        batch_size = st.slider("Batch Size", 8, 1024, 128, key="stacked_dnn_batch_size")
        early_stopping_patience = st.slider(
            "Early Stopping Patience", 5, 60, 20, key="stacked_dnn_patience"
        )
        optimizer_choice = st.selectbox(
            "Optimizer",
            ["AdamW", "Adam", "SGD", "RMSprop"],
            key="stacked_dnn_optimizer",
        )
        initial_learning_rate = st.number_input(
            "Initial Learning Rate",
            min_value=1e-7,
            max_value=0.1,
            value=0.001,
            step=0.0001,
            format="%.6f",
            key="stacked_dnn_lr",
        )
        weight_decay = 0.0
        if optimizer_choice == "AdamW":
            weight_decay = st.number_input(
                "Weight Decay (AdamW)",
                min_value=0.0,
                max_value=1e-2,
                value=1e-5,
                step=1e-6,
                format="%.6f",
                key="stacked_dnn_wd",
            )
        loss_function_choice = st.selectbox(
            "Loss Function",
            ["Huber", "Mean Squared Error", "Mean Absolute Error"],
            key="stacked_dnn_loss",
        )
        huber_delta = None
        if loss_function_choice == "Huber":
            huber_delta = st.number_input(
                "Huber Delta",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
                format="%.2f",
                key="stacked_dnn_huber_delta",
            )

        st.subheader("DNN Architecture")
        if "stacked_layers_config" not in st.session_state:
            st.session_state["stacked_layers_config"] = deepcopy(DEFAULT_STACKED_LAYERS)
        else:
            st.session_state["stacked_layers_config"] = _ensure_single_output_layer(
                st.session_state["stacked_layers_config"]
            )

        layers = []
        num_layers = st.number_input(
            "Number of Layers",
            min_value=1,
            max_value=20,
            value=len(st.session_state["stacked_layers_config"]),
            step=1,
            key="stacked_num_layers",
        )
        if num_layers > len(st.session_state["stacked_layers_config"]):
            st.session_state["stacked_layers_config"].extend(
                [{"type": "Dense", "units": 64, "activation": "relu"}]
                * (num_layers - len(st.session_state["stacked_layers_config"]))
            )
        elif num_layers < len(st.session_state["stacked_layers_config"]):
            st.session_state["stacked_layers_config"] = st.session_state[
                "stacked_layers_config"
            ][:num_layers]

        for i in range(num_layers):
            conf = st.session_state["stacked_layers_config"][i]
            col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
            layer_type = col1.selectbox(
                f"Layer {i + 1} Type",
                ["Dense", "BatchNormalization", "Dropout"],
                index=["Dense", "BatchNormalization", "Dropout"].index(conf["type"]),
                key=f"stacked_layer_type_{i}",
            )
            if layer_type == "Dense":
                units = col2.slider(
                    f"Units {i + 1}",
                    min_value=1,
                    max_value=512,
                    value=conf.get("units", 64),
                    key=f"stacked_layer_units_{i}",
                )
                activation = col3.selectbox(
                    f"Activation {i + 1}",
                    ["relu", "tanh", "sigmoid", "linear", "softplus"],
                    index=["relu", "tanh", "sigmoid", "linear", "softplus"].index(
                        conf.get("activation", "relu")
                    ),
                    key=f"stacked_layer_activation_{i}",
                )
                layers.append(
                    {"type": "Dense", "units": units, "activation": activation}
                )
            elif layer_type == "Dropout":
                rate = col2.slider(
                    f"Dropout Rate {i + 1}",
                    min_value=0.0,
                    max_value=0.6,
                    value=float(conf.get("rate", 0.1)),
                    step=0.05,
                    key=f"stacked_layer_dropout_{i}",
                )
                layers.append({"type": "Dropout", "rate": rate})
            else:
                layers.append({"type": "BatchNormalization"})
        layers = _ensure_single_output_layer(layers)
        st.session_state["stacked_layers_config"] = layers

    st.subheader("Meta-Learner")
    meta_model_choice = st.selectbox(
        "Meta Model",
        ["Ridge", "LinearRegression"],
        index=0,
        key="stacked_meta_model_choice",
    )
    ridge_alpha = st.number_input(
        "Ridge Alpha",
        min_value=1e-6,
        max_value=100.0,
        value=1.0,
        step=0.1,
        format="%.4f",
        key="stacked_ridge_alpha",
    )
    n_folds = st.slider(
        "Number of folds", min_value=3, max_value=10, value=5, key="stacked_folds"
    )
    enable_debug_logs = st.checkbox(
        "Show training debug logs",
        value=False,
        key="stacked_enable_debug_logs",
    )
    can_use_abs = include_dnn and (
        "XGBoost" in selected_meta_sources
        or "DNN+XGBoost Ensemble" in selected_meta_sources
    )
    add_abs_diff = st.checkbox(
        "Use abs(DNN - XGB) as extra meta-feature",
        value=False,
        disabled=not can_use_abs,
        key="stacked_use_abs_diff",
    )
    if not can_use_abs:
        add_abs_diff = False

    scaler_choice = "StandardScaler"

    if st.button("Train Stacked Ensemble", key="stacked_train_button"):
        with st.spinner("Training stacked ensemble with GroupKFold OOF..."):
            debug_messages = []
            debug_placeholder = st.empty()

            def _debug_log(msg):
                debug_messages.append(msg)
                if enable_debug_logs:
                    debug_placeholder.text("\n".join(debug_messages[-20:]))

            required_cols = selected_features + ["MPI", "Country"]
            df_clean = df[required_cols].dropna()
            if df_clean.empty:
                st.error("No rows available after dropping missing values.")
                return

            X_raw = df_clean[selected_features]

            groups = df_clean["Country"].astype(str)
            unique_groups = groups.nunique()
            if unique_groups < n_folds:
                st.error(
                    f"GroupKFold requires at least {n_folds} unique countries; found {unique_groups}."
                )
                return

            # Keep target as pandas Series for index-aligned .iloc in CV loops.
            y = df_clean["MPI"].clip(lower=0)
            dnn_params = {
                "epochs": epochs,
                "batch_size": batch_size,
                "early_stopping_patience": early_stopping_patience,
                "layers_config": layers,
                "initial_learning_rate": initial_learning_rate,
                "weight_decay": weight_decay,
                "optimizer_choice": optimizer_choice,
                "loss_function_choice": loss_function_choice,
                "huber_delta": huber_delta if huber_delta is not None else 0.1,
            }

            base_model_params = {
                "XGBoost": xgb_params,
                "Random Forest": rf_params,
                "Support Vector Regression": svr_params,
                "KNN Regressor": knn_params,
            }

            training_base_models = [
                m for m in selected_meta_sources if m in TRAINABLE_BASE_MODELS
            ]
            if "DNN+XGBoost Ensemble" in selected_meta_sources:
                if not include_dnn:
                    st.error("DNN+XGBoost Ensemble requires DNN to be enabled.")
                    return
                if "XGBoost" not in training_base_models:
                    training_base_models.append("XGBoost")
            if "DNN+Random Forest Ensemble" in selected_meta_sources:
                if not include_dnn:
                    st.error("DNN+Random Forest Ensemble requires DNN to be enabled.")
                    return
                if "Random Forest" not in training_base_models:
                    training_base_models.append("Random Forest")
            if "DNN+KNN Ensemble" in selected_meta_sources:
                if not include_dnn:
                    st.error("DNN+KNN Ensemble requires DNN to be enabled.")
                    return
                if "KNN Regressor" not in training_base_models:
                    training_base_models.append("KNN Regressor")

            results = _train_stacked_models(
                X=X_raw,
                y=y,
                groups=groups,
                n_splits=n_folds,
                corr_threshold=corr_threshold,
                scaler_choice=scaler_choice,
                dnn_params=dnn_params,
                base_model_order=training_base_models,
                meta_source_order=selected_meta_sources,
                ensemble_feature_alphas=ensemble_feature_alphas,
                base_model_params=base_model_params,
                meta_model_choice=meta_model_choice,
                ridge_alpha=ridge_alpha,
                include_dnn=include_dnn,
                use_abs_diff=add_abs_diff,
                debug_log_fn=_debug_log if enable_debug_logs else None,
            )

            metadata = {
                "features_used": results["features_used"],
                "dropped_features_correlation": results["dropped_features_correlation"],
                "correlation_threshold": corr_threshold,
                "folds": int(n_folds),
                "grouping": "Country",
                "meta_model": meta_model_choice,
                "meta_scaled": bool(meta_model_choice == "Ridge"),
                "ridge_alpha": float(ridge_alpha),
                "include_dnn": bool(include_dnn),
                "base_model_order": training_base_models,
                "meta_source_order": selected_meta_sources,
                "ensemble_feature_alphas": ensemble_feature_alphas,
                "use_abs_diff": bool(add_abs_diff),
                "meta_feature_order": results["meta_feature_names"],
            }
            _save_stacked_artifacts(
                dnn_model=results["dnn_model"],
                base_models=results["base_models"],
                meta_model=results["meta_model"],
                meta_scaler=results["meta_scaler"],
                scaler=results["scaler"],
                metadata=metadata,
            )

            st.session_state["stacked_results"] = {
                "overall": results["overall"],
                "fold_avg": results["fold_avg"],
                "fold_df": results["fold_df"],
                "oof_true": results["oof_true"],
                "oof_pred": results["oof_pred"],
                "histories": results["histories"],
                "dropped_features": results["dropped_features_correlation"],
                "features_used": results["features_used"],
                "base_models": selected_meta_sources,
                "meta_features": results["meta_feature_names"],
                "meta_weights_df": results["meta_weights_df"],
                "fold_debug_df": results["fold_debug_df"],
                "debug_messages": debug_messages,
            }

        st.success(
            "Stacked ensemble training completed and artifacts saved to models/stacked/"
        )

    if "stacked_results" in st.session_state:
        r = st.session_state["stacked_results"]
        # Backward compatibility for results cached before debug fields existed.
        if "fold_debug_df" not in r:
            r["fold_debug_df"] = pd.DataFrame()
        if "debug_messages" not in r:
            r["debug_messages"] = []
        if "meta_weights_df" not in r:
            r["meta_weights_df"] = pd.DataFrame()
        st.subheader("Model Performance")
        st.write(f"**MAE:** {r['overall']['mae']:.5f}")
        st.write(f"**RMSE:** {r['overall']['rmse']:.5f}")
        st.write(f"**R²:** {r['overall']['r2']:.5f}")

        st.subheader("K-Fold Averaged Metrics")
        st.write(f"**MAE (avg):** {r['fold_avg']['mae']:.5f}")
        st.write(f"**RMSE (avg):** {r['fold_avg']['rmse']:.5f}")
        st.write(f"**R² (avg):** {r['fold_avg']['r2']:.5f}")
        st.dataframe(r["fold_df"], use_container_width=True)

        st.caption("Base models used: " + ", ".join(r["base_models"]))
        st.caption("Meta features: " + ", ".join(r["meta_features"]))

        if not r["meta_weights_df"].empty:
            st.subheader("Ridge Meta-Feature Weights")
            st.dataframe(
                r["meta_weights_df"].sort_values("weight", key=np.abs, ascending=False),
                use_container_width=True,
            )
            _plot_meta_weights(r["meta_weights_df"])

        st.subheader("Training Debug")
        if not r["fold_debug_df"].empty:
            st.dataframe(r["fold_debug_df"], use_container_width=True)
        else:
            st.caption("No fold debug table available for this run.")
        if r.get("debug_messages"):
            with st.expander("Training debug logs"):
                st.text("\n".join(r["debug_messages"]))

        if r["dropped_features"]:
            st.info(
                "Dropped due to correlation threshold: "
                + ", ".join(r["dropped_features"])
            )

        st.subheader("Predicted vs Actual")
        _plot_pred_vs_actual(r["oof_true"], r["oof_pred"])

        st.subheader("DNN Learning Curves")
        if r["histories"]:
            _plot_dnn_learning_curves(r["histories"])
        else:
            st.caption("DNN was disabled for this run.")

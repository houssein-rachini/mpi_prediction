import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    KFold,
    GroupKFold,
    TimeSeriesSplit,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Iterable


# ---------- Metrics helpers ----------
def _rmse(y_true, y_pred, sample_weight=None):
    return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))


def _make_bins(y: np.ndarray, quantiles=(0, 0.2, 0.4, 0.6, 0.8, 1.0)) -> np.ndarray:
    """Same binning helper as in ensemble: quantile-based bins for stratification."""
    qs = np.quantile(y, quantiles)
    return np.clip(np.digitize(y, qs[1:-1], right=True), 0, len(qs) - 2)


# =========================
# Balanced Group K-Fold (same logic style as ensemble)
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
    """Same fold summary table style as ensemble (fold, country, rows, total rows)."""
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


# -------- Cross-Validation for base models (analog of cross_validate_ensemble) --------
def cross_validate_base_model(
    X: pd.DataFrame,
    y: pd.Series,
    base_model,
    cv_type: str = "kfold",
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    groups=None,
    sample_weights=None,
    balance_groups: bool = False,
    stratify_bins: bool = False,
):
    """
    Run CV on a *pure base model* using the same CV strategies as the ensemble:
      - kfold
      - timeseries
      - groupkfold
      - balanced groupkfold (via balance_groups + stratify_bins)

    Returns:
      metrics_df, summary, fold_info
    """
    fold_info = None

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

        # per-fold scaling (same good practice as ensemble)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)

        model = clone(base_model)
        model.fit(X_tr_scaled, y_tr)

        y_pred_va = model.predict(X_va_scaled)

        mae = mean_absolute_error(y_va, y_pred_va)
        rmse = _rmse(y_va, y_pred_va)
        r2 = r2_score(y_va, y_pred_va)

        # optional weighted metrics
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

    return metrics_df, summary, fold_info


# ---------- Plot helpers ----------
def display_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")


def plot_predictions(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
    plt.xlabel("Actual MPI")
    plt.ylabel("Predicted MPI")
    plt.title("Predicted vs Actual MPI")
    st.pyplot(fig)


def plot_learning_curve(model, X, y, cv, title="Learning Curve"):
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10),
    )
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_scores_mean, label="Training Loss", marker="o")
    ax.plot(train_sizes, test_scores_mean, label="Validation Loss", marker="s")
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)


def plot_residuals(y_val, y_pred):
    residuals = y_val - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_val, y=residuals, alpha=0.7)
    plt.axhline(y=0, color="red", linestyle="--")
    plt.xlabel("Actual MPI")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot (Error Analysis)")
    st.pyplot(fig)


# ---------- Main ML Training Tab ----------
def show_ml_training_tab(df: pd.DataFrame):
    st.title("🖥️ Machine Learning Training")

    # Show previous *trained* model results (hold-out test), like before
    if "ml_results" in st.session_state:
        st.subheader("📊 Previous Training Results (Hold-out Test)")
        results = st.session_state["ml_results"]
        model_name = results["model"]
        st.write(f"**Model:** {model_name}")
        display_metrics(results["y_test"], results["y_pred"])
        plot_predictions(results["y_test"], results["y_pred"])
        plot_residuals(results["y_test"], results["y_pred"])

    # ====== Feature selection ======
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    target_col = "MPI"
    # Make sure target cannot be in features
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

    if target_col not in df.columns:
        st.error("MPI column not found in the dataset.")
        return

    # Just in case: strip MPI from selected_features if user imported old state
    selected_features = [f for f in selected_features if f != target_col]

    # Drop rows with missing target or selected features
    df_clean = df.dropna(subset=[target_col] + selected_features)

    # Base X, y for CV (full data)
    X_full = df_clean[selected_features]
    y_full = df_clean[target_col]

    # ====== Model selection & hyperparameters ======
    model_options = [
        "XGBoost",
        "Random Forest",
        "Support Vector Regression",
        "KNN Regressor",
    ]
    selected_model = st.selectbox("Select an ML model:", model_options)

    base_model = None
    params = {}

    if selected_model == "XGBoost":
        params["n_estimators"] = st.slider(
            "Number of Trees (n_estimators)", 50, 500, 200
        )
        params["learning_rate"] = st.slider("Learning Rate", 0.01, 0.5, 0.05)
        params["max_depth"] = st.slider("Max Depth", 3, 10, 5)
        params["min_child_weight"] = st.slider(
            "Min Child Weight", 1, 10, 1, key="xgb_min_child_weight"
        )
        base_model = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            random_state=42,
        )

    elif selected_model == "Random Forest":
        params["n_estimators"] = st.slider(
            "Number of Trees (n_estimators)", 50, 300, 150
        )
        params["min_samples_split"] = st.slider("Min Samples Split", 2, 10, 2)
        params["min_samples_leaf"] = st.slider("Min Samples Leaf", 1, 10, 1)
        base_model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42,
        )

    elif selected_model == "Support Vector Regression":
        params["C"] = st.slider("Regularization Parameter (C)", 1, 500, 100)
        params["gamma"] = st.slider("Kernel Coefficient (gamma)", 0.001, 1.0, 0.1)
        base_model = SVR(kernel="rbf", C=params["C"], gamma=params["gamma"])

    elif selected_model == "KNN Regressor":
        params["n_neighbors"] = st.slider("Number of Neighbors (n_neighbors)", 1, 20, 5)
        params["metric"] = st.selectbox(
            "Distance Metric", ["manhattan", "euclidean", "minkowski"]
        )
        base_model = KNeighborsRegressor(
            n_neighbors=params["n_neighbors"], metric=params["metric"]
        )

    # ====== Validation / CV controls (same style as ensemble) ======
    st.subheader("Validation / Cross-Validation Strategy")
    use_cv = st.checkbox(
        "Use cross-validation (same strategies as ensemble)", value=False
    )

    if use_cv:
        cv_type = st.selectbox(
            "CV type",
            ["kfold", "timeseries", "groupkfold"],
            index=0,
        )
        n_splits = st.slider("Number of folds", 3, 10, 5)

        # Group handling
        groups = None
        balance_groups = False
        stratify_bins = False

        if cv_type == "groupkfold":
            group_candidates = [c for c in df_clean.columns if c != target_col]
            group_col = st.selectbox(
                "Grouping column (e.g., Country / Region)",
                options=group_candidates,
            )
            groups = df_clean[group_col].to_numpy()

            balance_groups = st.checkbox(
                "Balance folds by row count (keep groups intact)", value=True
            )
            stratify_bins = st.checkbox(
                "Also stratify target bins (uses quantiles of y)", value=True
            )

        # TimeSeriesSplit uses a time column
        X_cv = X_full.copy()
        y_cv = y_full.copy()
        df_ts = None
        if cv_type == "timeseries":
            time_candidates = [
                c for c in df_clean.columns if c.lower() in ("year", "date", "time")
            ] or list(df_clean.columns)
            time_col = st.selectbox("Time column for ordering", options=time_candidates)
            # sort by time column to respect temporal order
            df_ts = df_clean.sort_values(time_col)
            X_cv = df_ts[selected_features]
            y_cv = df_ts[target_col]

        # Weight column (like ensemble, e.g. Total_Pop)
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
        sample_weights = None
        if weight_col != "<None>":
            if cv_type == "timeseries":
                sample_weights = df_ts[weight_col]
            else:
                sample_weights = df_clean[weight_col]

        if st.button("Run Cross-Validation", key="ml_cv_button"):
            with st.spinner("Running cross-validation..."):
                metrics_df, summary, fold_info = cross_validate_base_model(
                    X=X_cv,
                    y=y_cv,
                    base_model=base_model,
                    cv_type=cv_type,
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=42,
                    groups=groups if cv_type == "groupkfold" else None,
                    sample_weights=sample_weights,
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
                w_line = f" | W-MAE: {summary['W_MAE_mean']:.4f}"
                if summary["W_MAE_std"] is not None:
                    w_line += f" ± {summary['W_MAE_std']:.4f}"
                w_line += f" | W-RMSE: {summary['W_RMSE_mean']:.4f}"
                if summary["W_RMSE_std"] is not None:
                    w_line += f" ± {summary['W_RMSE_std']:.4f}"
                st.write(base_line + w_line)
            else:
                st.write(base_line)

    else:
        # ====== Simple train/test (no CV), same idea as ensemble's "Train Model" path ======
        st.markdown("**Simple train/test split (no cross-validation).**")
        if st.button("Train ML Model", key="ml_train_button"):
            with st.spinner("Training model..."):
                X_train, X_test, y_train, y_test = train_test_split(
                    X_full, y_full, test_size=0.2, random_state=42
                )

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Fit final model
                base_model.fit(X_train_scaled, y_train)
                y_pred = base_model.predict(X_test_scaled)

                # Save model + scaler for later prediction use
                joblib.dump(base_model, "trained_ml_model.pkl")
                joblib.dump(scaler, "ml_scaler.pkl")
                st.write(
                    "✅ Model and Scaler saved successfully to "
                    "'trained_ml_model.pkl' and 'ml_scaler.pkl'"
                )

            # Store results in session_state
            st.session_state["ml_results"] = {
                "y_test": y_test,
                "y_pred": y_pred,
                "model": selected_model,
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2": r2_score(y_test, y_pred),
            }

            st.subheader("📊 Model Performance on Hold-out Test Set")
            display_metrics(y_test, y_pred)

            st.subheader("📈 Predictions vs Actual Values")
            plot_predictions(y_test, y_pred)

            st.subheader("Residual Plot (Error Analysis)")
            plot_residuals(y_test, y_pred)

            # Learning curve using a simple KFold on the training set
            st.subheader("📉 Learning Curve")
            cv_lc = KFold(n_splits=5, shuffle=True, random_state=42)
            plot_learning_curve(
                base_model,
                X_train_scaled,
                y_train,
                cv=cv_lc,
                title=f"Learning Curve ({selected_model})",
            )

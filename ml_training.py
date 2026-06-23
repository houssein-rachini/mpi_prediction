import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import (
    train_test_split,
    learning_curve,
    GroupKFold,
)
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO


def train_xgb_quantile_models(X_train_scaled, y_train, params):
    common_params = {
        "objective": "reg:quantileerror",
        "tree_method": "hist",
        "n_estimators": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "min_child_weight": params["min_child_weight"],
        "random_state": 42,
    }
    q05_model = xgb.XGBRegressor(quantile_alpha=0.05, **common_params)
    q50_model = xgb.XGBRegressor(quantile_alpha=0.50, **common_params)
    q95_model = xgb.XGBRegressor(quantile_alpha=0.95, **common_params)
    q05_model.fit(X_train_scaled, y_train)
    q50_model.fit(X_train_scaled, y_train)
    q95_model.fit(X_train_scaled, y_train)
    return q05_model, q50_model, q95_model


def render_plot_download(fig, file_stem):
    plot_counter = st.session_state.get("ml_plot_download_counter", 0)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    st.download_button(
        "Download Plot (PNG, 300 DPI)",
        data=buffer.getvalue(),
        file_name=f"{file_stem}.png",
        mime="image/png",
        key=f"ml_plot_download_{file_stem}_{plot_counter}",
    )
    st.session_state["ml_plot_download_counter"] = plot_counter + 1


# Function to display model evaluation metrics
def display_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")


# Function to visualize predictions vs actual values
def plot_predictions(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r")
    plt.xlabel("Actual MPI")
    plt.ylabel("Predicted MPI")
    plt.title("Predicted vs Actual MPI")
    st.pyplot(fig)
    render_plot_download(fig, "ml_predicted_vs_actual")
    plt.close(fig)


def plot_oof_scatter(cv_oof_df, n_splits):
    oa = cv_oof_df["Actual_MPI"].to_numpy()
    op = cv_oof_df["Predicted_MPI"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(oa, op, alpha=0.35, s=10, color="steelblue")
    lim = [min(oa.min(), op.min()) - 0.02, max(oa.max(), op.max()) + 0.02]
    ax.plot(lim, lim, "--r", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual MPI")
    ax.set_ylabel("Predicted MPI")
    ax.set_title(f"OOF Predicted vs Actual ({n_splits}-fold, n={len(oa):,})")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    render_plot_download(fig, "ml_oof_scatter")
    plt.close(fig)


# Function to plot learning curves
def plot_learning_curve(model, X, y, n_splits, title="Learning Curve"):
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=n_splits,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10),
    )
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, train_scores_mean, label="Training Loss", marker="o")
    ax.plot(train_sizes, test_scores_mean, label="Validation Loss", marker="s")
    ax.set_xlabel("Training Examples", fontsize=13)
    ax.set_ylabel("Loss (MSE)", fontsize=13)
    ax.set_title(title, fontsize=15)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(fontsize=11)
    st.pyplot(fig)
    render_plot_download(fig, "ml_learning_curve")
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
    render_plot_download(fig, "ml_residual_plot")
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
            file_name="ml_cv_fold_metrics.csv",
            mime="text/csv",
            key=f"{key_prefix}_cv_metrics",
        )
    if cv_oof_df is not None and not cv_oof_df.empty:
        st.download_button(
            "⬇️ Download CV out-of-fold predictions CSV",
            data=cv_oof_df.to_csv(index=False).encode("utf-8"),
            file_name="ml_cv_oof_predictions.csv",
            mime="text/csv",
            key=f"{key_prefix}_cv_oof",
        )


# Main function for ML training
def show_ml_training_tab(df):
    st.title("🖥️ Machine Learning Training")
    if "ml_results" in st.session_state:
        st.subheader("📊 Previous Training Results")
        results = st.session_state["ml_results"]
        model = results["model"]
        st.write(f"**Model:** {model}")
        display_metrics(results["y_test"], results["y_pred"])
        if "cv_df" in results:
            st.write(f"**{results['n_splits']}-Fold GroupKFold CV (by Country)**")
            st.dataframe(results["cv_df"].style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}))
            _cv_download_buttons(results.get("cv_df"), results.get("cv_oof"), "ml_prev")
        _prev_oof = results.get("cv_oof")
        if _prev_oof is not None and not _prev_oof.empty:
            plot_oof_scatter(_prev_oof, results.get("n_splits", "?"))
        else:
            plot_predictions(results["y_test"], results["y_pred"])
        plot_residuals(results["y_test"], results["y_pred"])
    # Select features
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    default_cols = [
        "Mean_NTL",
        "log_Mean_NTL",
        "NTL_per_capita",
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
    ]
    selected_features = st.multiselect(
        "Select features for training:", numeric_cols, default=[c for c in default_cols if c in numeric_cols]
    )
    if selected_features:
        n_rows = df.dropna(subset=["MPI"] + selected_features).shape[0]
        st.caption(f"Rows available for training: **{n_rows:,}**")

    # Target variable (MPI)
    target_col = "MPI"
    if target_col not in df.columns:
        st.error("MPI column not found in the dataset.")
        return

    # Drop missing values
    df_clean = df.dropna(subset=[target_col] + selected_features)

    # Split data
    X = df_clean[selected_features]
    y = df_clean[target_col]
    wanted_ids = ["Country", "Region", "Year"]
    id_cols_present = [c for c in wanted_ids if c in df.columns]
    df_ids = df.loc[df_clean.index, id_cols_present]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    ids_test = df_ids.loc[X_test.index]

    # ML Models with parameter customization
    model_options = [
        "XGBoost",
        "XGBoost Quantile",
        "Random Forest",
        "Support Vector Regression",
        "KNN Regressor",
    ]
    selected_model = st.selectbox("Select an ML model:", model_options)

    scaler_choice = "StandardScaler"

    # Standardize features
    scaler = StandardScaler() if scaler_choice == "StandardScaler" else MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = None
    params = {}

    if selected_model in ["XGBoost", "XGBoost Quantile"]:
        params["n_estimators"] = st.slider(
            "Number of Trees (n_estimators)", 50, 500, 200
        )
        params["learning_rate"] = st.slider("Learning Rate", 0.01, 0.5, 0.05)
        params["max_depth"] = st.slider("Max Depth", 3, 10, 5)
        params["min_child_weight"] = st.slider(
            "Min Child Weight", 1, 10, 1, key="xgb_min_child_weight"
        )
        if selected_model == "XGBoost":
            model = xgb.XGBRegressor(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                max_depth=params["max_depth"],
                min_child_weight=params["min_child_weight"],
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            )
        else:
            model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=0.50,
                tree_method="hist",
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
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            n_jobs=-1,
            random_state=42,
        )

    elif selected_model == "Support Vector Regression":
        params["C"] = st.slider("Regularization Parameter (C)", 1, 500, 100)
        params["gamma"] = st.slider("Kernel Coefficient (gamma)", 0.001, 1.0, 0.1)
        model = SVR(kernel="rbf", C=params["C"], gamma=params["gamma"])

    elif selected_model == "KNN Regressor":
        params["n_neighbors"] = st.slider("Number of Neighbors (n_neighbors)", 1, 20, 5)
        params["metric"] = st.selectbox(
            "Distance Metric", ["manhattan", "euclidean", "minkowski"]
        )
        model = KNeighborsRegressor(
            n_neighbors=params["n_neighbors"], metric=params["metric"]
        )

    n_splits = st.slider(
        "Number of folds for Cross-Validation (GroupKFold by Country)", 2, 10, 5
    )

    # Train ML Model with Cross-Validation
    if st.button("Train ML Model"):
        with st.spinner("Training in progress..."):
            # GroupKFold CV on the full dataset (by Country)
            cv_fold_rows = []
            cv_oof_parts = []
            if "Country" in df_clean.columns and selected_model != "XGBoost Quantile":
                gkf = GroupKFold(n_splits=n_splits)
                groups_arr = df_clean["Country"].values
                for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups_arr)):
                    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
                    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
                    _sc = scaler.__class__()
                    X_tr_sc = _sc.fit_transform(X_tr)
                    X_te_sc = _sc.transform(X_te)
                    _m = clone(model)
                    _m.fit(X_tr_sc, y_tr)
                    _pred = _m.predict(X_te_sc)
                    cv_fold_rows.append({
                        "Fold": fold + 1,
                        "MAE": mean_absolute_error(y_te, _pred),
                        "RMSE": np.sqrt(mean_squared_error(y_te, _pred)),
                        "R²": r2_score(y_te, _pred),
                    })
                    _oof_part = df_ids.iloc[te_idx].reset_index(drop=True)
                    _oof_part.insert(0, "Fold", fold + 1)
                    _oof_part["Actual_MPI"] = np.asarray(y_te).ravel()
                    _oof_part["Predicted_MPI"] = np.asarray(_pred).ravel()
                    cv_oof_parts.append(_oof_part)
            cv_df = pd.DataFrame(cv_fold_rows) if cv_fold_rows else pd.DataFrame()
            cv_oof_df = (
                pd.concat(cv_oof_parts, ignore_index=True)
                if cv_oof_parts
                else pd.DataFrame()
            )

            # Train final model on full training set
            coverage = None
            interval_width = None
            if selected_model == "XGBoost Quantile":
                q05_model, q50_model, q95_model = train_xgb_quantile_models(
                    X_train_scaled, y_train, params
                )
                y_pred = q50_model.predict(X_test_scaled)
                lower_raw = q05_model.predict(X_test_scaled)
                upper_raw = q95_model.predict(X_test_scaled)
                lower = np.minimum(lower_raw, upper_raw)
                upper = np.maximum(lower_raw, upper_raw)
                interval_width = upper - lower
                coverage = np.mean((y_test >= lower) & (y_test <= upper))

                q05_model.save_model("trained_xgb_quantile_q05.json")
                q50_model.save_model("trained_xgb_quantile_q50.json")
                q95_model.save_model("trained_xgb_quantile_q95.json")
                model = q50_model
                joblib.dump(scaler, "quantile_scaler.pkl")
                st.write(
                    "Saved quantile artifacts to 'trained_xgb_quantile_q05.json', 'trained_xgb_quantile_q50.json', 'trained_xgb_quantile_q95.json', and 'quantile_scaler.pkl'"
                )
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

            # Save trained model and scaler
            joblib.dump(model, "trained_ml_model.pkl")
            joblib.dump(scaler, "ml_scaler.pkl")
            st.write(
                "✅ Model and Scaler saved successfully to 'trained_ml_model.pkl' and 'ml_scaler.pkl'"
            )
        # Display metrics
        st.subheader("📊 Model Performance")
        # Store training results in session state
        validation_results = _make_validation_results(ids_test, y_test, y_pred)
        st.session_state["ml_results"] = {
            "y_test": y_test,
            "y_pred": y_pred,
            "validation_results": validation_results,
            "cv_df": cv_df,
            "cv_oof": cv_oof_df,
            "n_splits": n_splits,
            "model": selected_model,
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
        }
        display_metrics(y_test, y_pred)
        st.download_button(
            "Download Actual vs Predicted CSV",
            data=validation_results.to_csv(index=False).encode("utf-8"),
            file_name="ml_validation_results.csv",
            mime="text/csv",
            key="ml_download_current_results",
        )

        if selected_model == "XGBoost Quantile":
            st.write(f"**90% Interval Coverage:** {coverage:.4f}")
            st.write(f"**Mean Interval Width:** {interval_width.mean():.4f}")

        if not cv_df.empty:
            st.subheader(f"📊 {n_splits}-Fold GroupKFold CV (by Country)")
            st.dataframe(cv_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}))
            summary_row = cv_df[["MAE", "RMSE", "R²"]].agg(["mean", "std"])
            st.write(
                f"Mean ± Std — MAE: {summary_row.loc['mean','MAE']:.4f} ± {summary_row.loc['std','MAE']:.4f} | "
                f"RMSE: {summary_row.loc['mean','RMSE']:.4f} ± {summary_row.loc['std','RMSE']:.4f} | "
                f"R²: {summary_row.loc['mean','R²']:.4f} ± {summary_row.loc['std','R²']:.4f}"
            )
            _cv_download_buttons(cv_df, cv_oof_df, "ml_now")

        # Visualization
        st.subheader("📈 OOF Predictions vs Actual Values")
        if not cv_oof_df.empty:
            plot_oof_scatter(cv_oof_df, n_splits)
        else:
            plot_predictions(y_test, y_pred)

        # Plot Learning Curves
        st.subheader("📉 Learning Curve")
        plot_learning_curve(
            model,
            X_train_scaled,
            y_train,
            n_splits,
            title=f"Learning Curve ({selected_model})",
        )

        st.subheader("Residual Plot (Error Analysis)")
        plot_residuals(y_test, y_pred)

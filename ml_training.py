import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve,
    KFold,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


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
    ax.set_xlabel("Training Examples")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title(title)
    ax.legend()
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


# Main function for ML training
def show_ml_training_tab(df):
    st.title("🖥️ Machine Learning Training")
    if "ml_results" in st.session_state:
        st.subheader("📊 Previous Training Results")
        results = st.session_state["ml_results"]
        model = results["model"]
        st.write(f"**Model:** {model}")
        display_metrics(results["y_test"], results["y_pred"])
        st.write(
            f"**{results["n_splits"]}-Fold CV Mean Squared Error:** {abs(results['cv_scores'].mean()):.4f}"
        )
        plot_predictions(results["y_test"], results["y_pred"])
        plot_residuals(results["y_test"], results["y_pred"])
    # Select features
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    default_cols = ["StdDev_NTL", "Mean_GPP", "StdDev_Pop", "StdDev_LST", "StdDev_NDVI"]
    selected_features = st.multiselect(
        "Select features for training:", numeric_cols, default=default_cols
    )

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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ML Models with parameter customization
    model_options = [
        "XGBoost",
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

    if selected_model == "XGBoost":
        params["n_estimators"] = st.slider(
            "Number of Trees (n_estimators)", 50, 500, 100
        )
        params["learning_rate"] = st.slider("Learning Rate", 0.01, 0.5, 0.05)
        params["max_depth"] = st.slider("Max Depth", 3, 10, 5)
        params["min_child_weight"] = st.slider(
            "Min Child Weight", 1, 10, 1, key="xgb_min_child_weight"
        )
        model = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
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

    # Select number of K-Folds
    n_splits = st.slider(
        "Select number of folds for Cross Validation (K-Fold)", 2, 10, 5
    )

    # Train ML Model with Cross-Validation
    if st.button("Train ML Model"):
        with st.spinner("Training in progress..."):
            # K-Fold Cross Validation
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(
                model,
                X_train_scaled,
                y_train,
                cv=kfold,
                scoring="neg_mean_squared_error",
            )

            # Train final model on full training set
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
        st.session_state["ml_results"] = {
            "y_test": y_test,
            "y_pred": y_pred,
            "cv_scores": scores,
            "n_splits": n_splits,
            "model": selected_model,
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
        }
        display_metrics(y_test, y_pred)

        # Display Cross-Validation Results
        st.write(f"**{n_splits}-Fold CV Mean Squared Error:** {abs(scores.mean()):.4f}")

        # Visualization
        st.subheader("📈 Predictions vs Actual Values")
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

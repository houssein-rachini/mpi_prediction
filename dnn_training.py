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
import joblib
import pandas as pd


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


def show_dnn_training_tab(df):
    """Displays the UI for training the deep learning model."""
    st.title("🧠Deep Learning Model Training")
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
        "Select features for training:", numeric_cols, default=default_cols
    )

    if "MPI" not in selected_features:
        selected_features.append("MPI")
    df_selected = df[selected_features].dropna()
    X = df_selected.drop(columns=["MPI"])
    y = np.maximum(df_selected["MPI"], 0)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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
    layers = []
    num_layers = st.number_input(
        "Number of Layers", 1, 20, 3, step=1, key="dnn_num_layers"
    )
    for i in range(num_layers):
        col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
        layer_type = col1.selectbox(
            f"Layer {i+1} Type",
            ["Dense", "BatchNormalization", "Dropout"],
            key=f"dnn_type_{i}",
        )
        if layer_type == "Dense":
            units = col2.slider(f"Units {i+1}", 1, 512, 128, key=f"dnn_units_{i}")
            activation = col3.selectbox(
                f"Activation {i+1}",
                ["relu", "tanh", "sigmoid", "linear"],
                key=f"dnn_activation_{i}",
            )
            layers.append({"type": "Dense", "units": units, "activation": activation})
        elif layer_type == "Dropout":
            rate = col2.slider(
                f"Dropout Rate {i+1}", 0.0, 0.5, 0.1, key=f"dnn_dropout_{i}"
            )
            layers.append({"type": "Dropout", "rate": rate})
        elif layer_type == "BatchNormalization":
            layers.append({"type": "BatchNormalization"})

    if st.button("Train Model", key=f"dnn_train_button"):
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
        st.subheader("📊 Model Performance")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")
        # metrics of the last few epochs
        st.write("### Epochs")
        st.write(pd.DataFrame(history))
        st.subheader("Training and Validation Loss Curve")

        plot_loss_curve(history)
        plot_results(y_val, y_pred_dnn)

        st.subheader("Residual Plot (Error Analysis)")
        plot_residuals(y_val, y_pred_dnn)

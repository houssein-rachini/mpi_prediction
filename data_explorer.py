import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def show_data_explorer_tab(df):
    """Displays the dataset in an interactive table and provides filtering options."""
    st.title("📊 Data Explorer")

    # Allow users to filter by country
    country_options = ["All"] + list(df["Country"].unique())
    selected_country = st.selectbox("Filter by Country", country_options)

    # Allow users to filter by Region after selecting a country
    if selected_country != "All":
        region_options = ["All"] + list(
            df[df["Country"] == selected_country]["Region"].unique()
        )
        selected_region = st.selectbox("Filter by Region", region_options)
        if selected_region != "All":
            df_filtered = df[
                (df["Country"] == selected_country) & (df["Region"] == selected_region)
            ]
        else:
            df_filtered = df[df["Country"] == selected_country]
    else:
        df_filtered = df

    # Allow users to filter by year
    year_options = ["All"] + list(df_filtered["Year"].unique())
    selected_year = st.selectbox("Filter by Year", year_options)
    if selected_year != "All":
        df_filtered = df_filtered[df_filtered["Year"] == selected_year]

    # Show dataset
    st.write("### Dataset Preview")
    st.dataframe(df_filtered)

    # Show basic statistics
    st.write("### Summary Statistics")
    st.write(df_filtered.describe())

    # Correlation Matrix Section (Computed on the Full Dataset)
    st.write("### Correlation Matrix (Computed on Full Dataset)")

    # Allow user to select numeric variables
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Default columns are those that start with "Mean" or "StdDev" and "MPI"
    default_cols = [
        col
        for col in numeric_cols
        if col.startswith("Mean") or col.startswith("StdDev")
    ]
    default_cols.append("MPI")

    selected_vars = st.multiselect(
        "Select variables for correlation matrix (full dataset):",
        numeric_cols,
        default=default_cols,
    )

    if len(selected_vars) > 1:
        corr_matrix = df[selected_vars].corr()

        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax
        )
        st.pyplot(fig)
    else:
        st.warning(
            "Please select at least two numerical variables to generate the correlation matrix."
        )

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def show_data_explorer_tab(df):
    """Displays the dataset in an interactive table and provides filtering options."""
    st.title("📊 Data Explorer")
    st.markdown(
        """
        Explore the underlying dataset used for MPI prediction.
        You can filter the data by country, governorate, and year, view summary statistics,
        and examine relationships between variables using a correlation matrix.
        """
    )
    # Allow users to filter by country
    country_options = ["All"] + list(df["Country"].unique())

    selected_country = st.selectbox("Filter by Country", country_options)

    # Allow users to filter by Region after selecting a country
    if selected_country != "All":
        region_options = ["All"] + list(
            df[df["Country"] == selected_country]["Region"].unique()
        )
        selected_region = st.selectbox("Filter by Governorate", region_options)
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
    # change the column name from Region to Governorate
    df_filtered = df_filtered.rename(columns={"Region": "Governorate"})
    st.dataframe(df_filtered)

    # Show basic statistics
    st.write("### Summary Statistics")
    # remove the Year column from the statistics
    df_filtered_no_year = df_filtered.drop(columns=["Year"])
    st.write(df_filtered_no_year.describe())

    # Correlation Matrix Section (Computed on the Full Dataset)
    st.write("### Correlation Matrix")
    st.info(
        "This correlation matrix is computed on the full dataset, regardless of any filtering applied above."
    )

    # Allow user to select numeric variables
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Default columns = the 20 production training features (+ MPI), grouped by
    # source so same-source variables sit next to each other in the matrix.
    preferred_default_cols = [
        # Nighttime lights (NTL)
        "Mean_NTL", "Median_NTL", "Sum_NTL", "StdDev_NTL", "NTL_per_capita",
        "NTL_anom", "NTL_anom_lag1",
        # Land surface temperature — night
        "Mean_LST", "LSTN_anom",
        # Land surface temperature — day
        "Mean_LST_Day", "LST_Day_anom",
        # Vegetation (NDVI)
        "ndvi_lst_ratio", "NDVI_anom",
        # Population
        "Mean_Pop", "Median_Pop", "StdDev_Pop", "CV_Pop",
        # Gross primary productivity
        "Mean_GPP",
        # Built-up (GHSL)
        "Mean_BUILT_S", "Median_BUILT_S", "StdDev_BUILT_S", "StdDev_BUILT_V",
        # Target
        "MPI",
    ]
    default_cols = [col for col in preferred_default_cols if col in numeric_cols]

    selected_vars = st.multiselect(
        "Select variables for correlation matrix:",
        numeric_cols,
        default=default_cols,
    )

    if len(selected_vars) > 1:
        corr_matrix = df[selected_vars].corr()

        # Scale the figure with the number of variables. NOTE: st.pyplot downscales
        # the figure to the container width by default, which makes a 20+ var matrix
        # tiny/unreadable — so we render at native size (use_container_width=False)
        # at a fixed per-cell size, and the page scrolls if it's wider than the column.
        n = len(selected_vars)
        cell = 0.85  # inches per cell
        side = max(8, n * cell)
        fig, ax = plt.subplots(figsize=(side, side), dpi=100)
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            ax=ax,
            annot_kws={"size": 14, "weight": "bold"},
            cbar_kws={"shrink": 0.5},
            square=True,
            vmin=-1,
            vmax=1,
        )
        ax.tick_params(axis="x", labelrotation=90, labelsize=13)
        ax.tick_params(axis="y", labelrotation=0, labelsize=13)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
        # Bigger, bold colorbar (legend) tick numbers
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
        for t in cbar.ax.get_yticklabels():
            t.set_fontweight("bold")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
    else:
        st.warning(
            "Please select at least two numerical variables to generate the correlation matrix."
        )

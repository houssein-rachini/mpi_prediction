import streamlit as st
import folium
import ee
from streamlit_folium import folium_static
from google.oauth2 import service_account
import json
import ee
import altair as alt
import branca.colormap as cm


from google.oauth2 import service_account

service_account_info = dict(st.secrets["google_ee"])  # No need for .to_json()

credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=["https://www.googleapis.com/auth/earthengine"]
)

ee.Initialize(credentials)
# Load FAO GAUL dataset
fao_gaul = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")


@st.cache_resource
def get_governorate_geometry(country, governorate):
    filtered = fao_gaul.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", country), ee.Filter.eq("ADM1_NAME", governorate)
        )
    )
    return filtered.geometry().getInfo()


@st.cache_resource
def get_country_center(country):
    filtered = fao_gaul.filter(ee.Filter.eq("ADM0_NAME", country))
    coords = filtered.geometry().centroid().coordinates().getInfo()
    return coords if coords else [0, 0]


@st.cache_resource
def get_governorate_center(country, governorate):
    filtered = fao_gaul.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", country), ee.Filter.eq("ADM1_NAME", governorate)
        )
    )
    coords = filtered.geometry().centroid().coordinates().getInfo()
    return coords if coords else [0, 0]


@st.cache_resource
def generate_map_multiple_governorates(
    governorate_df, center_coords, use_satellite=True, fill_opacity=0.6
):
    tiles = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        if use_satellite
        else "OpenStreetMap"
    )
    attr = "Esri World Imagery" if use_satellite else "OpenStreetMap"

    m = folium.Map(
        location=[center_coords[1], center_coords[0]],
        zoom_start=6,
        tiles=tiles,
        attr=attr,
    )

    min_mpi = governorate_df["MPI"].min()
    max_mpi = governorate_df["MPI"].max()
    colormap = cm.linear.YlOrRd_09.scale(min_mpi, max_mpi)

    colormap.caption = "MPI Value"
    colormap.add_to(m)

    # Build GeoJSON FeatureCollection
    features = []

    for _, row in governorate_df.iterrows():
        gov_name = row["Governorate"]
        mpi = row["MPI"]
        year = row["Year"]

        try:
            geom = get_governorate_geometry(row["Country"], gov_name)

            # Convert GeometryCollection to MultiPolygon if needed
            if geom["type"] == "GeometryCollection":
                polygons = [
                    g
                    for g in geom["geometries"]
                    if g["type"] in ["Polygon", "MultiPolygon"]
                ]
                if not polygons:
                    continue
                if len(polygons) > 1:
                    geom = {
                        "type": "MultiPolygon",
                        "coordinates": [p["coordinates"] for p in polygons],
                    }
                else:
                    geom = polygons[0]

            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"Governorate": gov_name, "MPI": mpi, "Year": year},
                }
            )

        except Exception as e:
            print(f"Skipping {gov_name}: {e}")

    geojson_data = {"type": "FeatureCollection", "features": features}

    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            "fillColor": colormap(feature["properties"]["MPI"]),
            "color": "black",
            "weight": 1,
            "fillOpacity": fill_opacity,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["Governorate", "Year", "MPI"],
            aliases=["Governorate", "Year", "MPI"],
            localize=True,
            sticky=False,
        ),
    ).add_to(m)

    return m


def show_visualization_tab(df):
    st.title("📊 MPI Visualization")

    st.markdown(
        """
    ### Explore Multidimensional Poverty Index (MPI)
    Use the options below to explore poverty trends across countries and their governorates.
    """
    )

    df = df.rename(columns={"Region": "Governorate"})
    df["Year"] = df["Year"].astype(int)

    selected_country = st.selectbox("🌍 Select a Country", df["Country"].unique())
    filtered_df = df[df["Country"] == selected_country]

    viz_option = st.selectbox(
        "Visualization Type",
        [
            "Choose an Option for MPI Visualization",
            "Single Governorate",
            "Time Series",
            "Yearly Countrywide Average",
            "Yearly by Governorate",
        ],
        index=0,
    )
    if viz_option == "Choose an Option for MPI Visualization":
        st.markdown("### 🗺️ Map View (Default)")

        country_df = df[df["Country"] == selected_country]

        if not country_df.empty:
            # Pick first governorate in that country to get center
            first_gov = country_df["Governorate"].iloc[0]
            center = get_country_center(selected_country)

            use_satellite = st.toggle("🛰️ Show Satellite Imagery", value=True)

            tiles = (
                "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                if use_satellite
                else "OpenStreetMap"
            )
            attr = "Esri World Imagery" if use_satellite else "OpenStreetMap"

            m = folium.Map(
                location=[center[1], center[0]],
                zoom_start=6,
                tiles=tiles,
                attr=attr,
            )

            if use_satellite:
                # Esri Boundaries & Labels overlay
                folium.TileLayer(
                    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                    attr="Esri Boundaries & Labels",
                    name="Labels & Boundaries",
                    overlay=True,
                    control=False,
                ).add_to(m)

            folium_static(m, width=750, height=500)
        else:
            st.warning("No data found for this country.")

    if viz_option != "Choose an Option":
        if viz_option == "Single Governorate":
            selected_governorate = st.selectbox(
                "🏙️ Select a Governorate", filtered_df["Governorate"].unique()
            )
            selected_year = st.selectbox(
                "🗓️ Select a Year", sorted(filtered_df["Year"].unique())
            )

            filtered = filtered_df[
                (filtered_df["Governorate"] == selected_governorate)
                & (filtered_df["Year"] == selected_year)
            ]

            if not filtered.empty:
                mpi_value = filtered.iloc[0]["MPI"]
                center_coords = get_governorate_center(
                    selected_country, selected_governorate
                )

                use_satellite = st.toggle("🛰️ Show Satellite Imagery", value=True)

                m = folium.Map(
                    location=[center_coords[1], center_coords[0]],
                    zoom_start=6,
                    tiles=(
                        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        if use_satellite
                        else "OpenStreetMap"
                    ),
                    attr="Esri World Imagery" if use_satellite else "OpenStreetMap",
                )

                if use_satellite:
                    # Add Esri Boundaries & Labels overlay
                    folium.TileLayer(
                        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                        attr="Esri Boundaries & Labels",
                        name="Labels & Boundaries",
                        overlay=True,
                        control=False,
                    ).add_to(m)

                geom = get_governorate_geometry(selected_country, selected_governorate)
                if geom:
                    folium.GeoJson(
                        geom,
                        style_function=lambda feature: {
                            "fillColor": "blue",
                            "color": "black",
                            "weight": 2,
                            "fillOpacity": 0.4,
                        },
                        tooltip=f"{selected_governorate} ({selected_year}): MPI = {mpi_value:.5f}",
                    ).add_to(m)

                folium_static(m, width=750, height=500)
            else:
                st.warning("No data available for this governorate and year.")

        elif viz_option == "Time Series":
            selected_governorate = st.selectbox(
                "🏙️ Select a Governorate", filtered_df["Governorate"].unique()
            )

            # Governorate time series
            ts_gov_df = filtered_df[
                filtered_df["Governorate"] == selected_governorate
            ].copy()
            ts_gov_df["Year"] = ts_gov_df["Year"].astype(int)
            ts_gov_df = ts_gov_df.sort_values("Year")

            chart_gov = (
                alt.Chart(ts_gov_df)
                .mark_line(point=True, color="steelblue")
                .encode(
                    x=alt.X("Year:O", axis=alt.Axis(labelAngle=0), title="Year"),
                    y=alt.Y("MPI", title="Governorate MPI"),
                    tooltip=["Year", "MPI"],
                )
                .properties(
                    width=340,
                    height=400,
                    title=f"{selected_governorate} MPI Over Time",
                )
            )

            # Country-level average MPI per year
            country_avg_df = (
                filtered_df.groupby("Year")["MPI"]
                .mean()
                .reset_index()
                .sort_values("Year")
            )
            chart_country = (
                alt.Chart(country_avg_df)
                .mark_line(point=True, color="green")
                .encode(
                    x=alt.X("Year:O", axis=alt.Axis(labelAngle=0), title="Year"),
                    y=alt.Y("MPI", title="Country Avg MPI"),
                    tooltip=["Year", "MPI"],
                )
                .properties(
                    width=340,
                    height=400,
                    title=f"{selected_country} Avg MPI Over Time",
                )
            )

            # show the charts under each other
            st.altair_chart(chart_gov, use_container_width=True)
            st.altair_chart(chart_country, use_container_width=True)

        elif viz_option == "Yearly Countrywide Average":
            selected_year = st.selectbox(
                "🗓️ Select a Year", sorted(filtered_df["Year"].unique())
            )
            avg_mpi = filtered_df[filtered_df["Year"] == selected_year]["MPI"].mean()
            st.metric(
                label=f"{selected_country} MPI Average in {selected_year}",
                value=f"{avg_mpi:.5f}",
            )

        elif viz_option == "Yearly by Governorate":
            st.markdown(
                "#### Displaying all governorates MPI for selected country (selected year)"
            )

            year_options = sorted(filtered_df["Year"].unique())
            selected_year = st.selectbox("🗓️ Select Year", year_options)

            year_df = filtered_df[filtered_df["Year"] == selected_year]

            if not year_df.empty:
                fallback_center = get_governorate_center(
                    year_df.iloc[0]["Country"], year_df.iloc[0]["Governorate"]
                )

                use_satellite = st.toggle("🛰️ Show Satellite Imagery", value=True)
                fill_opacity = st.slider(
                    "🔆 Adjust MPI Layer Transparency", 0.0, 1.0, 0.6, step=0.05
                )

                # Choose base map tiles
                tiles = (
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                    if use_satellite
                    else "OpenStreetMap"
                )
                attr = "Esri World Imagery" if use_satellite else "OpenStreetMap"

                # Generate map
                m = generate_map_multiple_governorates(
                    year_df, fallback_center, use_satellite, fill_opacity
                )

                # Add labels & boundaries overlay if satellite
                if use_satellite:
                    folium.TileLayer(
                        tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
                        attr="Esri Boundaries & Labels",
                        name="Labels & Boundaries",
                        overlay=True,
                        control=False,
                    ).add_to(m)

                folium_static(m, width=750, height=550)

                st.bar_chart(year_df.set_index("Governorate")["MPI"])
                st.info(f"MPI values for all governorates in {selected_year}.")
            else:
                st.warning("No data available to show for this year.")

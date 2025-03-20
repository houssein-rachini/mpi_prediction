import streamlit as st
import folium
import ee
from streamlit_folium import folium_static
from google.oauth2 import service_account
import json
import ee


from google.oauth2 import service_account

service_account_info = dict(st.secrets["google_ee"])  # No need for .to_json()

credentials = service_account.Credentials.from_service_account_info(
    service_account_info, scopes=["https://www.googleapis.com/auth/earthengine"]
)

ee.Initialize(credentials)

# Load FAO GAUL dataset
fao_gaul = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level1")


@st.cache_resource
def get_region_geometry(country, region):
    """Retrieve the region's boundary from FAO GAUL."""
    filtered_region = fao_gaul.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", country), ee.Filter.eq("ADM1_NAME", region)
        )
    )
    return filtered_region.geometry().getInfo()


@st.cache_resource
def get_region_center(country, region):
    """Retrieve the centroid of a region from FAO GAUL."""
    filtered_region = fao_gaul.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", country), ee.Filter.eq("ADM1_NAME", region)
        )
    )
    coords = filtered_region.geometry().centroid().coordinates().getInfo()
    return coords if coords else [0, 0]


@st.cache_resource
def generate_map(country, region, year, mpi_value, center_coords):
    """Generate Folium Map without default markers."""
    m = folium.Map(
        location=[center_coords[1], center_coords[0]],
        zoom_start=6,
        control_scale=False,
        prefer_canvas=True,
    )

    region_geom = get_region_geometry(country, region)
    if region_geom:
        folium.GeoJson(
            region_geom,
            style_function=lambda feature: {
                "fillColor": "blue",
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.4,
            },
            tooltip=f"{region} ({year}): MPI = {mpi_value:.5f}",
        ).add_to(m)

    return m


def show_visualization_tab(df):
    """Displays the MPI Visualization Tab in Streamlit."""
    st.title("MPI Visualization")

    selected_country = st.selectbox("Select a Country", df["Country"].unique())
    filtered_df = df[df["Country"] == selected_country]

    selected_region = st.selectbox("Select a Region", filtered_df["Region"].unique())
    filtered_df = filtered_df[filtered_df["Region"] == selected_region]

    selected_year = st.selectbox("Select a Year", filtered_df["Year"].unique())
    filtered_df = filtered_df[filtered_df["Year"] == selected_year]

    center_coords = get_region_center(selected_country, selected_region)
    mpi_value = filtered_df.iloc[0]["MPI"] if not filtered_df.empty else None

    if mpi_value is not None:
        m = generate_map(
            selected_country, selected_region, selected_year, mpi_value, center_coords
        )
        st.components.v1.html(m.get_root().render(), height=500, width=700)
    else:
        st.warning("No data available for the selected region and year.")

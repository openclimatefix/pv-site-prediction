# Stremalit Dashboard to explore NWP data and PV data for sites.
# Version 1.0

# ------------ Import libraries ------------
import streamlit as st
import plotly.express as px
import pandas as pd
import xarray as xr
import netCDF4 as nc
import zarr
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
import base64
import datetime as dt
import os

import ocf_blosc2

from streamlit_extras.colored_header import colored_header
from streamlit_option_menu import option_menu

# THESE LIBRARIES CAN BE USED IF LOADING IN THE PV MODEL
# from psp.serialization import load_model
# from psp.typings import X
# from psp.models.regressors.decision_trees import SklearnRegressor
# from psp.exp_configs.island_nwp import ExpConfig

# from psp.gis import CoordinateTransformer


# ----------- PAGE SET UP -----------
st.set_page_config(page_title="PV Dashboard", layout="wide")


# TO DO: reconfigure logo


def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


logo_path = "/home/zak/pv-site-prediction/dashboards/images/OCF_full_mark_wht.png"
logo_base64 = img_to_base64(logo_path)

st.sidebar.markdown(
    f"""
    <style>
        .center {{
            display: block;
            margin-left: auto;
            margin-right: auto;
            margin-top: -60px;
            margin-bottom: 20px;
            width: 200px;  # Adjust this value based on your desired width
        }}
    </style>
    <img class="center" src="data:image/png;base64,{logo_base64}" width="200" alt="OCF logo" />
    """,
    unsafe_allow_html=True,
)


# NEED DIFFERENT SIDE BARS BASED OFF WHAT DATA IS AVAILABLE

# for SME keep to the most simple case of just pv analysis
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Inspect PV"],  # , "NWP Data", "Normalised Plots"],
        icons=["house"],  # ,"book","envelope"],
        menu_icon="cast",
        default_index=0,
    )

st.title("PV Data Dashboard")

colored_header(
    label="Site PV Exploration Dashboard",  # UK Met Office NWP Data +
    description="This dashboard has been designed to explore the PV data provided.",  # NWP data provided by the Met Office and
    color_name="blue-70",
)


# ------------ Import data ------------ [keep clients names out]
# These files could be specified in a separate config file
pv1_file = "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/PV/sme/zarr_format/sme_t5.zarr"
pv_data = pd.read_csv(
    "/mnt/storage_b/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/PV/sme/pv_data_dict.csv"
)


# Function to open netCDF file and convert it to pandas DataFrame
def netcdf_to_dataframe(file_path):
    ds = xr.open_dataset(file_path)
    df = ds.to_dataframe().reset_index()
    return df


# Function to open zarr file and convert it to pandas DataFrame
def zarr_to_dataframe(file_path):
    ds = xr.open_zarr(file_path)
    df = ds.to_dataframe().reset_index()
    return ds, df


pv1_df = netcdf_to_dataframe(pv1_file)
# pv_pred_df = pd.read_csv(pv_pred)


if selected == "Inspect PV":
    st.write("Select Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = pd.to_datetime(
            st.date_input("Start Date", value=pd.to_datetime("2019-01-01").date())
        )
    with col2:
        end_date = pd.to_datetime(
            st.date_input("End Date", value=pd.to_datetime("2021-01-01").date())
        )

    # pv_ids = pv1_df['pv_id'].values

    pv_ids = sorted(pv_data["pv_id"].values)

    # pv_ids = pv_ids.sortby

    # PV ID selection and filter the dataset for the selected PV ID
    selected_pv_id = st.selectbox("Select PV ID", pv_ids)
    selected_data = pv1_df.loc[pv1_df["pv_id"] == int(selected_pv_id)]

    selected_data = selected_data[
        (selected_data["ts"] >= start_date) & (selected_data["ts"] <= end_date)
    ]

    st.title("PV Power Plot")

    # Extract the power values and time index
    power_values = selected_data["power"].values
    capacity_values = selected_data["capacity"].values
    time_index = selected_data["ts"].values

    # Convert power_values to a pandas Series with the appropriate index
    power_series = pd.Series(power_values, index=selected_data["ts"])

    # Calculate the yearly rolling average
    yearly_rolling_avg = power_series.rolling("365D").mean()

    # Discard the first 364 days
    yearly_rolling_avg = yearly_rolling_avg.loc[yearly_rolling_avg.index[364:]]

    # Convert time index to pandas DateTimeIndex for Plotly and create a graph
    time_index = pd.to_datetime(time_index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_index, y=power_values, mode="lines", name="Power"))
    fig.update_layout(
        title=f"PV Power for PV ID: {selected_pv_id}", xaxis_title="Time", yaxis_title="Power"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time_index, y=capacity_values, mode="lines", name="Power"))
    fig2.add_trace(
        go.Scatter(x=time_index, y=yearly_rolling_avg, mode="lines", name="Yearly Rolling Average")
    )
    fig2.update_layout(
        title=f"Capacity for PV ID: {selected_pv_id}", xaxis_title="Time", yaxis_title="Capacity"
    )
    st.plotly_chart(fig2, use_container_width=True)


# elif selected == "NWP Data":

# nwp_ds, nwp_df = zarr_to_dataframe(nwp_file)

# latitudes = nwp_ds["x"].values.tolist()
# longitudes = nwp_ds["y"].values.tolist()

# st.write("Select Date Range")
# col1, col2 = st.columns(2)
# with col1:
#     start_date = pd.to_datetime(st.date_input("Start Date", value=pd.to_datetime("2022-04-05").date()))
# with col2:
#     end_date = pd.to_datetime(st.date_input("End Date", value=pd.to_datetime("2022-04-15").date()))


# # Filter the dataset for the selected PV ID
# pv_dict = {}

# for index, row in pv_data.iterrows():
#     pv_id = row['pv_id']
#     latitude = row['easting']
#     longitude = row['northing']
#     pv_dict[pv_id] = (latitude, longitude)

# # Get the list of PV IDs
# pv_ids = pv1_df['pv_id'].values

# selected_pv_id = st.selectbox("Select PV ID", pv_ids)

# # Get the latitude and longitude for the selected PV site ID
# selected_latitude, selected_longitude = pv_dict.get(selected_pv_id, (None, None))


# # st.write("Select Latitude and Longitude")
# # col3, col4 = st.columns(2)
# # with col3:
# #     selected_latitude = st.selectbox("Select Latitude", latitudes)
# # with col4:
# #     selected_longitude = st.selectbox("Select Longitude", longitudes)

# st.write("Select Steps to Plot")

# step_values = np.unique(nwp_ds.step)

# # step_years = (step_values.astype(np.int64) / 1e9) / (60*60)

# # Set the step variable to the new values
# # nwp_ds["step"] = step_values

# selected_step = st.selectbox('Select step', step_values)
# nwp_ds_subset = nwp_ds.sel(x=selected_latitude, y=selected_longitude, step=selected_step)

# nwp_df = nwp_ds_subset.to_dataframe().reset_index()
# nwp_data_sel = nwp_df[(nwp_df["init_time"] >= start_date) & (nwp_df["init_time"] <= end_date)]


# # Filter data by date range
# # pv1_df = pv1_df[(pv1_df["datetimeUTC"] >= start_date) & (pv1_df["datetimeUTC"] <= end_date)]
# # pv2_df = pv2_df[(pv2_df["datetimeUTC"] >= start_date) & (pv2_df["datetimeUTC"] <= end_date)]
# # nwp_df = nwp_df[(nwp_df["time"] >= start_date) & (nwp_df["time"] <= end_date)]

# # Create Plotly charts
# # fig1 = px.line(pv1_df, x="datetimeUTC", y="15-Minute Output MW", title="PV 15-Min")
# # fig2 = px.line(pv2_df, x="datetimeUTC", y="Hourly PV Generated Units (MW)", title="PV Hourly")

# # st.plotly_chart(fig1, use_container_width=True)
# # st.plotly_chart(fig2, use_container_width=True)
# # Loop over all variables

# variables = ["mcc","lcc","hcc"]

# for variable in variables:
#     # Select the rows where variable is the current variable
#     data = nwp_data_sel[nwp_data_sel["variable"] == variable]
#     # Plot the data using plotly.express
#     fig = px.line(data, x="init_time", y="UKV", title=f"NWP Forecast - {variable}",height=300)
#     st.plotly_chart(fig, use_container_width=True)

# if st.checkbox("Show PV Data"):
#     pv1_df = pv1_df[(pv1_df["ts"] >= start_date) & (pv1_df["ts"] <= end_date)]

#     # Streamlit app
#     st.title("PV Power Plot")

#     # Filter the dataset for the selected PV ID

#     selected_data = pv1_df.loc[pv1_df['pv_id'] == selected_pv_id]


#     # Extract the power values and time index
#     power_values = selected_data['power'].values
#     time_index = selected_data['ts'].values

#     # Convert time index to pandas DateTimeIndex for Plotly
#     time_index = pd.to_datetime(time_index)

#     # Create the Plotly figure
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=time_index, y=power_values, mode='lines', name='Power'))

#     # Set plot layout
#     fig.update_layout(title=f"PV Power for PV ID: {selected_pv_id}",
#                     xaxis_title="Time",
#                     yaxis_title="Power")

#     # Plot the figure
#     st.plotly_chart(fig,use_container_width=True)


# elif selected == "Normalised Plots":
#     st.write("Normalised Plots")


# for step in selected_steps:
#     for variable in nwp_df["variable"].unique():
#         # Select the rows where variable is the current variable
#         data = nwp_df[nwp_df["variable"] == variable]
#         # Plot the data using plotly.express
#         fig = px.line(data, x="time", y="value", title=f"NWP Forecast - {variable}",height=300)
#         st.plotly_chart(fig, use_container_width=True)

# # Loop over the data variables and create a chart for each one
# for data_variable in data_variables:
#     # Filter the data for the selected steps and current data variable
#     data = nwp_data[(nwp_data["step"].isin(selected_steps)) & (nwp_data["variable"] == data_variable)]

#     # Create a chart based on the current chart type
#     if chart_type == "Line Chart":
#         chart = alt.Chart(data).mark_line().encode(
#             x="time",
#             y="value",
#             color="step"
#         ).properties(title=f"{chart_type} - {data_variable}")

#     # Display the chart using Streamlit
#     st.altair_chart(chart, use_container_width=True)

import streamlit as st
import plotly.express as px
import pandas as pd
import xarray as xr
import netCDF4 as nc
import zarr
import numpy as np
import plotly.graph_objs as go
import matplotlib.pyplot as plt

import ocf_blosc2

from streamlit_extras.colored_header import colored_header
from streamlit_option_menu import option_menu

import datetime as dt
from psp.serialization import load_model
from psp.typings import X
from psp.models.regressors.decision_trees import SklearnRegressor
from model.ocf.config import ExpConfig

import plotly.express as px

from PIL import Image
import base64


st.set_page_config(page_title="PV Dashboard", layout="wide")




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


#TO DO: Reconfigure logo path
"""
def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_path = "/Users/zakwatts/Coding/OCF/resources/images/OCF Logo/OCF_full_mark_wht.png"
logo_base64 = img_to_base64(logo_path)

# st.sidebar.markdown(
#     f"""
#     <style>
#         .center {{
#             display: block;
#             margin-left: auto;
#             margin-right: auto;
#             margin-top: -60px;
#             margin-bottom: 20px;
#             width: 200px;  # Adjust this value based on your desired width
#         }}
#     </style>
#     <img class="center" src="data:image/png;base64,{logo_base64}" width="200" alt="OCF logo" />
#     """,
#     unsafe_allow_html=True,
# )


with st.sidebar:
    selected = option_menu(
        menu_title='Main Menu',
        options=["PV Plots","PV vs Forecast (Test)", "PV vs Forecast for horizon"],
        icons=["bar-chart","graph-up-arrow","clipboard-data"],
        menu_icon="list",
        default_index=0
    )

# logo_path = "/Users/zakwatts/Coding/OCF/images/OCF Logo/OCF_full_mark_wht.png"
# st.sidebar.image(logo_path,width=200)

st.title("PV Forecasting Data Dashboard")

colored_header(
    label="Client Data Exploration Dashboard",
    description="This dashboard has been designed to explore the forecasted PV, actual PV and NWP data variables used in the this pilot project",
    color_name="blue-70",
)


pv1_file = "...nc"
pv2_file = "...nc"

pv3_file_client = "...csv"

nwp_file ="...folder path"
pv_pred = "test_errors sample ...csv"

pv4_ocf_full = "full ocf model for island specific use case  ...csv"

pv1_df = netcdf_to_dataframe(pv1_file)
pv2_df = netcdf_to_dataframe(pv2_file)

pv3_df = pd.read_csv(pv3_file_client)
pv3_df.reset_index(inplace=True)
pv3_df["ts"] = pd.to_datetime(pv3_df["ts"])

pv4_df = pd.read_csv(pv4_ocf_full)
pv4_df["ts"] = pd.to_datetime(pv4_df["ts"])

pv_pred_df = pd.read_csv(pv_pred)

# Load NWP forecasting data
nwp_ds, nwp_df = zarr_to_dataframe(nwp_file)



if selected == "PV Plots":

    st.header("Date Range")

    # Create two columns for start date and end date
    col1, col2 = st.columns(2)

    # Date inputs for start date and end date
    start_date = pd.to_datetime(col1.date_input("Start Date", value=pd.to_datetime("2022-01-01").date()))
    end_date = pd.to_datetime(col2.date_input("End Date", value=pd.to_datetime("2023-01-01").date()))


    def compare_pv(pv1, pv2):
        pv1['datetimeUTC'] = pd.to_datetime(pv1['datetimeUTC'])
        pv2['datetimeUTC'] = pd.to_datetime(pv2['datetimeUTC'])

        pv2 = pv2.drop("datetimeUTC", axis=1)  # Remove the "datetimeUTC" column
        # pv2 = pv2.rename(columns={"date": "datetimeUTC"}) 

        # Convert the datetime column to datetime format and set it as the index
        
        pv1.set_index('datetimeUTC', inplace=True)

        # Resample the data to hourly intervals and compute the sum of values in each hour
        df_hourly = (pv1["15-Minute Output MW"].resample('H').sum())/4

        # Reset the index if needed
        # df_hourly.reset_index(inplace=True)
            
        """
        merged_df = pd.merge(df_hourly, pv2, on='datetimeUTC')

        merged_df["error"] = merged_df["Hourly PV Generated Units (MW)"] - (merged_df["15-Minute Output MW"])
        
        merged_df["abs_error"]= abs(merged_df["error"])
        """

        # # get statistics about the error column
        # abs_error_stats = merged_df['abs_error'].describe()
        # st.write("Absolute Error Statistics")
        # st.write(abs_error_stats)
        
        # error_stats = merged_df['error'].describe()
        # st.write("Error Statistics")
        # st.write(error_stats)

        """
        
        # Create a scatter plot with a trendline (rolling average)
        fig = px.scatter(merged_df, x='datetimeUTC', y='error', trendline='lowess', labels={"error": "Error (MW)","datetimeUTC": "Date (UTC)"}, title ="Error Between Hourly and 15-minutely PV Output ")

        # Calculate rolling average
        window_size =st.slider("Select a rolling window",10,1000,600,10)   # Set the window size for the rolling average
        merged_df['rolling_avg'] = merged_df['error'].rolling(window=window_size).mean()

        # Add the line trace (rolling average) to the existing scatter plot
        fig.add_trace(go.Scatter(x=merged_df['datetimeUTC'], y=merged_df['rolling_avg'], mode='lines', name='Rolling Average'))
        fig.update_traces(marker_size=1)

        fig.add_hline(y=0)


        # Show the plot
        return df_hourly, fig
        """

    pv1_df = pv1_df[(pv1_df["datetimeUTC"] >= start_date) & (pv1_df["datetimeUTC"] <= end_date)]
    pv2_df = pv2_df[(pv2_df["datetimeUTC"] >= start_date) & (pv2_df["datetimeUTC"] <= end_date)]

    start_date_format = pd.to_datetime(start_date).tz_localize('UTC')
    end_date_format = pd.to_datetime(end_date).tz_localize('UTC')


    pv3_df = pv3_df[(pv3_df["ts"] >= start_date_format) & (pv3_df["ts"] <= end_date_format)]

    # Convert the ts column to a datetime column
    pv_pred_df['ts'] = pd.to_datetime(pv_pred_df['ts'])
    pv_pred_df = pv_pred_df.sort_values("ts")
    pv_pred_df = pv_pred_df[(pv_pred_df["ts"] >= start_date) & (pv_pred_df["ts"] <= end_date)]

    p4_df = pv4_df[(pv4_df["ts"] >= start_date) & (pv4_df["ts"] <= end_date)]

    horizon = 60

    # Filter the dataframe by the chosen horizon
    pv_pred_horizon = pv_pred_df[pv_pred_df['horizon']==horizon]

    # Create traces for the PV data
    trace1 = go.Scatter(x=pv1_df['datetimeUTC'], y=pv1_df['15-Minute Output MW'], name='15-Minute Output')
    trace2 = go.Scatter(x=pv2_df['datetimeUTC'], y=pv2_df['Hourly PV Generated Units (MW)'], name='Hourly Output')
    trace4 = go.Scatter(x=pv3_df['ts'], y=pv3_df['pv_data'], name='Client Output')
    trace5 = go.Scatter(x=pv_pred_horizon['ts'], y=pv_pred_horizon['pred'], name='PV Test Forecast')
    trace6 = go.Scatter(x=p4_df['ts'], y=p4_df['power'], name='Client forecast *')
    trace7 = go.Scatter(x=p4_df['ts'], y=p4_df['ocf'], name='Ocf forecast *')
    trace8 = go.Scatter(x=p4_df['ts'], y=p4_df['truth'], name='Truth forecast *')


    """
    df_hourly, fig2 = compare_pv(pv1_df,pv2_df)
    """

    # trace3 = go.Scatter(x=df_hourly.index, y=df_hourly.values, name='15-Min Resampled to Hour')

    # Add the PV traces to the list of traces
    traces = [trace1,trace2,trace4,trace5,trace6,trace7,trace8]

    layout = go.Layout(title='PV Sources Comparison', xaxis_title='Date (UTC)', yaxis_title='Power (MW)')

    fig = go.Figure(data=traces, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

    """
    st.plotly_chart(fig2,use_container_width=True)
    """


elif selected == "PV vs Forecast (Test)":

    st.header("Date Range")

    # Create two columns for start date and end date
    col1, col2 = st.columns(2)

    # Date inputs for start date and end date
    start_date = pd.to_datetime(col1.date_input("Start Date", value=pd.to_datetime("2022-04-05").date()))
    end_date = pd.to_datetime(col2.date_input("End Date", value=pd.to_datetime("2022-08-15").date()))

    start_date_format = pd.to_datetime(start_date).tz_localize('UTC')
    end_date_format = pd.to_datetime(end_date).tz_localize('UTC')

    pv3_df = pv3_df[(pv3_df["ts"] >= start_date_format) & (pv3_df["ts"] <= end_date_format)]

    pv1_df = pv1_df[(pv1_df["datetimeUTC"] >= start_date) & (pv1_df["datetimeUTC"] <= end_date)]
    pv2_df = pv2_df[(pv2_df["datetimeUTC"] >= start_date) & (pv2_df["datetimeUTC"] <= end_date)]

    # Convert the ts column to a datetime column
    pv_pred_df['ts'] = pd.to_datetime(pv_pred_df['ts'])
    pv_pred_df = pv_pred_df[(pv_pred_df["ts"] >= start_date) & (pv_pred_df["ts"] <= end_date)]

        # Sort the DataFrame by the "time" column
    pv_pred_df = pv_pred_df.sort_values("ts")

    pv_pred_df['date'] = pv_pred_df['ts'].dt.date

    # Group data by date and sum error for each date
    error_by_date = pv_pred_df.groupby('date')['error'].sum().reset_index()



    # selected_step = st.sidebar.selectbox('Select step', step_values)

            # Choose the horizon for which you want to create the graph
    horizon = 60

    # Filter the dataframe by the chosen horizon
    df_horizon = pv_pred_df[pv_pred_df['horizon']==horizon]

    # Create a line plot with plotly express
    fig = px.line(df_horizon, x='ts', y=['y', 'pred'], title=f'Horizon: {horizon}')
    fig.update_layout(xaxis_title='Time Step', yaxis_title='Power (kW)')

    # Create a bar chart of the error
    fig2 = px.bar(df_horizon, x='ts', y='error', title='Error')
    fig2.update_layout(xaxis_title='Time Step', yaxis_title='Error')

    
    

    # Add the error bar chart to the existing figure as a new subplot
    fig.add_trace(fig2.data[0])

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

elif selected == "PV vs Forecast for horizon":
    config = ExpConfig()

    model = load_model('model/ocf/model_7.pkl')

    model.set_data_sources(**config.get_data_source_kwargs())
    # start_time = dt.datetime(2022,7, 1)

    col1, col2, col3 = st.columns(3)

        # Add content to the first column
    with col1:
        start_time =  pd.to_datetime(st.date_input("Select a start date", dt.datetime(2022,7, 1)))

    # Add content to the second column
    with col2:
        min_value = 1
        max_value = 5
        step = 1
        days_view = st.slider("Number of days to view", min_value, max_value, 1, step, format="%d")

    with col3:
        min_value = 3
        max_value = 48
        step = 3
        horizon_select = st.slider("Horizon to view", min_value, max_value, 3, step, format="%d")

    # start_time = .to_datetime(st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-04-05").date()))

    end_time = start_time + dt.timedelta(days=days_view)

    pv1_df = pv1_df[(pv1_df["datetimeUTC"] >= start_time) & (pv1_df["datetimeUTC"] < end_time)]
    pv2_df = pv2_df[(pv2_df["datetimeUTC"] >= start_time) & (pv2_df["datetimeUTC"] < end_time)]

   
    horizon_index = horizon_select
    horizon_minutes = horizon_index * 60

    power_data: dict[dt.datetime, float] = {}

    ts = start_time
    while ts < end_time:
        x = X(pv_id='0', ts=ts )#- dt.timedelta(minutes=horizon_minutes))
        y = model.predict(x)
        powers = y.powers
        power_at_index = powers[horizon_index]
        offset = dt.timedelta(minutes=horizon_minutes)
        time = ts + offset
        power_data[time] = power_at_index
    
        ts += dt.timedelta(minutes=60)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Convert the power_data dictionary to Matplotlib dates
    times = list(power_data.keys())
    power_outputs = list(power_data.values())


    power_df = pd.DataFrame({"datetimeUTC": list(power_data.keys()), "power_output": list(power_data.values())})

    # Merge the two dataframes based on the "datetimeUTC" column
    merged_df = pd.merge(pv2_df, power_df, on="datetimeUTC")

    # Calculate the error between the predicted and actual power output
    merged_df["error"] = merged_df["Hourly PV Generated Units (MW)"] - merged_df["power_output"]

    # Create the plot

    # Convert the power_data dictionary to a Pandas DataFrame
    df2 = pd.DataFrame(list(power_data.items()))

    # Print the column names of the DataFrame
    # st.write(df2.columns)

    # Filter the pv1_df DataFrame to only include rows with matching timestamps
    # pv1_df_filt = pv1_df[pv1_df["datetimeUTC"].isin(times)]

    # Calculate the error between the predicted and actual power output
    # error = pv1_df_filt["15-Minute Output MW"] - power_outputs     

    trace1 = go.Scatter(x=pv2_df['datetimeUTC'], y=pv2_df['Hourly PV Generated Units (MW)'], name='Actual Hourly')
    trace2 = go.Scatter(x=times, y=power_outputs, name='Predicted')
    # trace3 = go.Scatter(x=pv1_df['datetimeUTC'], y=pv1_df["15-Minute Output MW"], name='Actual 15-Minute')

    # Create a Bar trace for the error bars
    trace4 = go.Bar(x=merged_df['datetimeUTC'], y=merged_df['error'], name='Error')

    # Create a data list with all three traces
    data = [trace1, trace2,trace4]

    # Create a Layout object with the plot title and axis labels
    layout = go.Layout(title="Predicted Power Output", xaxis=dict(title="Datetime (UTC)"), yaxis=dict(title="Power Output (kW)"))

    # Create a Figure object with the Scatter and Layout objects
    fig = go.Figure(data=data, layout=layout)

    # Display the plot in the Streamlit app
    st.plotly_chart(fig, use_container_width=True)




st.sidebar.info('This app is a data explorer for the CLIENT PV Forecasting Pilot.\n'
                '\nCreated by Open Climate Fix\n'
                )
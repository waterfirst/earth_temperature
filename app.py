#%%
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
from numpy.fft import fft


# Load world geometry data using geopandas
url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)


# NASA GISTEMP data download and processing
@st.cache_data
def download_and_process_gistemp_data():
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    df = pd.read_csv(url, skiprows=1, na_values="***")
    df.columns = ["Year"] + df.columns[1:].tolist()
    df_long = df.melt(id_vars=["Year"], var_name="Month", value_name="온도변화")
    df_long = df_long.dropna()
    df_long['Year'] = df_long['Year'].astype(int)
    df_long['Month'] = pd.Categorical(df_long['Month'], categories=list(df.columns[1:]), ordered=True)
    return df_long

# Data for zonal NASA GISTEMP data
@st.cache_data
def download_and_process_gistemp_zonal_data():
    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/ZonAnn.Ts+dSST.csv"
    columns = ["Year", "Glob", "NHem", "SHem", "24N-90N", "24S-24N", "90S-24S", 
               "64N-90N", "44N-64N", "24N-44N", "EQU-24N", "24S-EQU", "44S-24S", "64S-44S", "90S-64S"]
    df = pd.read_csv(url, skiprows=1, na_values="***", names=columns)
    df_long = df.melt(id_vars=["Year"], var_name="Zone", value_name="온도변화")
    df_long = df_long.dropna()
    df_long['Year'] = df_long['Year'].astype(int)
    return df_long

# Function to perform FFT analysis
def perform_fft_analysis(data):
    if 'Annual온도변화' not in data.columns:
        st.error("Error: 'Annual온도변화' column not found in the data.")
        return None
    
    fft_result = fft(data['Annual온도변화'].dropna())
    n = len(fft_result)
    freq = np.fft.fftfreq(n)
    power = np.abs(fft_result) ** 2 / n

    fft_df = pd.DataFrame({
        '주기': 1 / freq[1:n//2],
        '파워': power[1:n//2]
    }).dropna()

    top_periods = fft_df.nlargest(5, '파워')
    return fft_df, top_periods

# UI
st.title("지구 온도 변화 분석")

st.sidebar.subheader("연도 범위 설정")
year_range = st.sidebar.slider("연도 범위:", 1880, 2023, (1880, 2023))

graph_type = st.sidebar.selectbox("그래프 유형 선택", ["연간 추세", "월별 분포", "지역별 변화", "지도 시각화"])

# Load data
with st.spinner('데이터 다운로드 중...'):
    temp_data = download_and_process_gistemp_data()
    zonal_data = download_and_process_gistemp_zonal_data()

# Graphs based on user selection
if graph_type == "연간 추세":
    annual_data = temp_data.groupby("Year").agg(Annual온도변화=("온도변화", "mean")).reset_index()
    filtered_data = annual_data[(annual_data["Year"] >= year_range[0]) & (annual_data["Year"] <= year_range[1])]
    
    fig = px.line(filtered_data, x="Year", y="Annual온도변화", title="연간 평균 온도 편차")
    fig.update_layout(xaxis_title="연도", yaxis_title="온도 편차 (°C)")
    st.plotly_chart(fig)

elif graph_type == "월별 분포":
    filtered_data = temp_data[(temp_data["Year"] >= year_range[0]) & (temp_data["Year"] <= year_range[1])]
    fig = px.scatter(filtered_data, x="Month", y="온도변화", color="Year", title="월별 온도 편차 분포")
    fig.update_layout(xaxis_title="월", yaxis_title="온도 편차 (°C)")
    st.plotly_chart(fig)

elif graph_type == "지역별 변화":
    filtered_zonal_data = zonal_data[(zonal_data["Year"] >= year_range[0]) & (zonal_data["Year"] <= year_range[1])]
    fig = px.line(filtered_zonal_data, x="Year", y="온도변화", color="Zone", title="지역별 온도 변화")
    fig.update_layout(xaxis_title="연도", yaxis_title="온도 편차 (°C)")
    st.plotly_chart(fig)

elif graph_type == "지도 시각화":
    # Load world geometry data using geopandas
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)
    
    # Process recent year zonal temperature anomaly data
    recent_year = zonal_data["Year"].max()  # Latest year available
    decade_data = zonal_data[(zonal_data["Year"] >= recent_year - 10) & (zonal_data["Year"] <= recent_year)]

# Calculate 10-year average temperature anomaly by zone
    average_temp_change = decade_data.groupby("Zone").agg({"온도변화": "mean"}).reset_index()
    
    # Check the temperature data to ensure it's being calculated
    #st.write("Average Temperature Change by Zone:", average_temp_change)
    
    world['centroid_lat'] = world.geometry.centroid.y

# Reassign latitude bins to match the temperature data's zones
    world['Zone'] = pd.cut(
    world['centroid_lat'], 
    bins=[-90, -64, -24, 0, 24, 44, 64, 90], 
    labels=['90S-64S', '64S-24S', '24S-EQU', 'EQU-24N', '24N-44N', '44N-64N', '64N-90N']
    )


    #st.write("World Data with Zones:", world[['NAME', 'Zone']])

# Merge the temperature anomaly data with world map data
    merged_data = world.merge(average_temp_change, how="left", on="Zone")
    
    #st.write("Merged Data:", merged_data[['NAME', 'Zone', '온도변화']])
    
# Merge the temperature anomaly data with world map data
    merged_data['온도변화'] = merged_data['온도변화'].fillna(0)  # You can adjust this to other methods, such as leaving NaN
    
# Use Plotly to create a choropleth map
    fig = px.choropleth(merged_data,
                    geojson=merged_data.geometry,
                    locations=merged_data.index,
                    color="온도변화",  # Column showing temperature anomaly
                    hover_name="NAME",  # Country names
                    title=f"{recent_year-10}-{recent_year}년 지역별 평균 온도 편차",
                    projection="natural earth",
                    color_continuous_scale="RdBu_r",  # Use a valid colorscale
                    labels={'온도변화': '온도 편차 (°C)'})  # Legend label

# Update layout
    fig.update_geos(showcountries=True, showcoastlines=True, showland=True, fitbounds="locations")

# Show the map in Streamlit
    st.plotly_chart(fig)


# FFT Analysis Section
st.subheader("FFT 분석 결과")

annual_data = temp_data.groupby("Year").agg(Annual온도변화=("온도변화", "mean")).reset_index()
fft_results = perform_fft_analysis(annual_data)

if fft_results:
    fft_df, top_periods = fft_results
    st.write(f"주요 주기성 (상위 5개): {top_periods}")
    
    fig_fft = px.line(fft_df, x="주기", y="파워", log_x=True, title="FFT 분석 결과")
    st.plotly_chart(fig_fft)


    


# %%

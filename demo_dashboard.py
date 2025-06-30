import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import random
from datetime import datetime
import json

# Load static filter data
def load_filters():
    with open('data/dashboard_static_data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# Load static chart templates
def load_chart_templates():
    with open('data/dashboard_chart_data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

st.set_page_config(layout="wide", page_title="Dashboard (Demo with Random Data)")
col1, col2 = st.columns([5, 1])
with col1:
    title_placeholder = st.empty()
with col2:
    st.image('logo.png', width=120)

# Sidebar filters
filters = load_filters()
SPECIALITY_OPTIONS = filters.get('specialties', [])
CITY_OPTIONS       = filters.get('cities', [])
PINCODE_OPTIONS    = filters.get('pincodes', [])
CLIENT_OPTIONS     = filters.get('clients', [])
PROJECT_OPTIONS    = filters.get('projects', [])
STATE_OPTIONS      = filters.get('states', [])

title_placeholder.title("Dashboard: Demo with Randomized Data")

# Capture sidebar selections
selected_states   = st.sidebar.multiselect("Select State", STATE_OPTIONS)
selected_cities   = st.sidebar.multiselect("Select City", CITY_OPTIONS)
selected_pincodes = st.sidebar.multiselect("Select Pincode", PINCODE_OPTIONS)
selected_specs    = st.sidebar.multiselect("Select Speciality", SPECIALITY_OPTIONS)
selected_clients  = st.sidebar.multiselect("Select Client(s)", CLIENT_OPTIONS)
selected_projects = st.sidebar.multiselect("Select Project(s)", PROJECT_OPTIONS)

# Display active filters
st.sidebar.header("Active Filters")
st.sidebar.write(f"States: {len(selected_states)} selected")
st.sidebar.write(f"Cities: {len(selected_cities)} selected")
st.sidebar.write(f"Pincodes: {len(selected_pincodes)} selected")
st.sidebar.write(f"Specialities: {len(selected_specs)} selected")
st.sidebar.write(f"Clients: {len(selected_clients)} selected")
st.sidebar.write(f"Projects: {len(selected_projects)} selected")

# Determine scale: more data if no filter applied
def get_scale():
    any_filter = any([selected_states, selected_cities, selected_pincodes,
                      selected_specs, selected_clients, selected_projects])
    return 1.0 if any_filter else 2.0
scale = get_scale()

# Function to randomize base values according to scale
def randomize_data(base_list, scale_factor):
    randomized = []
    for item in base_list:
        base_value = item.get('value', 0)
        # apply scale and random noise
        value = max(0, int((base_value * scale_factor) * random.uniform(0.8, 1.2)))
        randomized.append({ 'label': item['label'], 'value': value })
    return randomized

# Load chart templates
chart_templates = load_chart_templates()

# Tabs setup

# Fixed tab names and corresponding chart keys (must align)
tab_names = [
    "🏷️ Manufacturer Analysis", "📂 Data Types within Rx", "📍 Geographical Distribution",
    "📊 Demographic Distribution", "💊 Medicines", "🏭 Pharma Analytics", "🩺 Observations",
    "🧪 Diagnostics", "🔍 Manufacturer Comparison", "💰 Value-Based Comparison",
    "🏭 Market Share by primary use", "🩸 Vitals"
]
chart_keys = [
    'manufacturer_analysis', 'data_types_within_rx', 'geographical_distribution',
    'demographic_distribution', 'medicines', 'pharma_analytics', 'observations',
    'diagnostics', 'manufacturer_comparison', 'value_comparison',
    'market_share_primary_use', 'vitals'
]
# Ensure keys match loaded template keys count
tabs = st.tabs(tab_names)


# Chart rendering helper
# Chart rendering helper
def render_chart(data_list, chart_type='pie', top_n=None, key=None):
    # Convert to DataFrame; handle empty or invalid data
    if not data_list or not isinstance(data_list, list):
        st.info("No data available for this view.")
        return
    df = pd.DataFrame(data_list)
    if df.empty or 'label' not in df.columns or 'value' not in df.columns:
        st.info("No valid data to display.")
        return
    if top_n:
        df = df.head(top_n)
    # Generate chart
    try:
        if chart_type == 'pie':
            fig = px.pie(df, names='label', values='value')
        else:
            fig = px.bar(df, x='value', y='label', orientation='h', text='value')
    except Exception as e:
        st.error(f"Chart error: {e}")
        st.dataframe(df)
        return
    st.plotly_chart(fig, use_container_width=True, key=key)
    st.dataframe(df, use_container_width=True)

# Render tabs with randomized data
def main():
    for idx, tab in enumerate(tabs):
        with tab:
            st.subheader(tab_names[idx])
            base = chart_templates.get(chart_keys[idx], []) or []
            randomized = randomize_data(base, scale)
            chart_type = 'pie' if idx % 2 == 0 else 'bar'
            top_n = 8 if chart_type=='pie' else 7
            render_chart(randomized, chart_type=chart_type, top_n=top_n, key=f"chart_{idx}")

def main():
    for idx, tab in enumerate(tabs):
        with tab:
            st.subheader(tab_names[idx])
            base = chart_templates.get(chart_keys[idx], [])
            randomized = randomize_data(base, scale)
            chart_type = 'pie' if idx % 2 == 0 else 'bar'
            top_n = 8 if chart_type=='pie' else 7
            render_chart(randomized, chart_type=chart_type, top_n=top_n, key=f"chart_{idx}")

if __name__ == '__main__':
    main()

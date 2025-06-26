import logging
import sys
import streamlit as st

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
)
import json
import os
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
from streamlit import session_state as state


def log_time(func):
    """Decorator to log start, end, and duration of a function call."""

    def wrapper(*args, **kwargs):
        logger.info(f"⏱️  {func.__name__} started")
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        logger.info(f"⏱️  {func.__name__} finished in {duration:.2f}s")
        return result

    return wrapper


@log_time
@st.cache_data
def load_data(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    elif file_extension == '.json':
        return json_to_dataframe(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV, Excel, or JSON file.")
    # The file_path parameter can be a URL. No additional code needed.


def json_to_dataframe(file_path):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    df = pd.json_normalize(json_data)
    return df


@log_time
@st.cache_data
def clean_medical_data(data):
    data['average_mrp'] = data[['min_mrp', 'max_mrp']].mean(axis=1).round(2)
    data['gender'] = data['gender'].replace("", "Unknown")
    data['value'] = data['value'].str.lower().apply(lambda x: "pain in abdomen" if "pain in abd" in str(x) else x)
    replacements = {
        'cbc': 'cbc',
        'urine': 'urine',
        'hbsag': 'hbsag',
    }
    for key, value in replacements.items():
        data['value'] = data['value'].str.lower().apply(lambda x: key if value in str(x) else x)
    return data

@log_time
def apply_filters(data, state_filter=None, city_filter=None, pincode_filter=None, speciality_filter=None,
                  client_filter=None, project_filter=None):
    """Filters medical data based on multiple criteria including state, city, pincode, speciality, client, and project."""

    filtered_data = data.copy()

    if state_filter:
        filtered_data = filtered_data[filtered_data['state_name'].isin(state_filter)]
    if city_filter:
        filtered_data = filtered_data[filtered_data['city'].isin(city_filter)]
    if pincode_filter:
        filtered_data = filtered_data[
            filtered_data['pincode'].str.split(',').apply(lambda x: any(p.strip() in pincode_filter for p in x))]
    if speciality_filter:
        filtered_data = filtered_data[filtered_data['speciality'].isin(speciality_filter)]
    if client_filter:
        filtered_data = filtered_data[filtered_data['client'].isin(client_filter)]
    if project_filter:
        filtered_data = filtered_data[filtered_data['project'].isin(project_filter)]

    return filtered_data

@log_time
def filter_by_date_range(data, start_date, end_date):
    # Ensure 'start_time' is in datetime format
    data['start_time'] = pd.to_datetime(data['start_time'], errors='coerce')  # Handle invalid dates

    # Convert 'start_date' and 'end_date' to datetime.datetime for comparison
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the data within the specified date range
    return data[(data['start_time'] >= start_date) & (data['start_time'] <= end_date)]

@log_time
def display_sidebar_totals(filtered_data):
    st.sidebar.markdown("### Totals in Analytics")
    # total_doctors = filtered_data['doctor_id'].nunique()
    # total_patients = filtered_data['id'].nunique()
    # total_rx = filtered_data['ptp_id'].nunique()
    total_doctors = 1245
    total_patients = 195098
    total_rx = 234135
    st.sidebar.metric("Total Doctors", total_doctors)
    st.sidebar.metric("Total Patients", total_patients)
    st.sidebar.metric("Total Rx", total_rx)


# def aggregate_geo_data(data, group_by_column, count_column):
#     aggregated_data = (
#         data
#         .groupby(group_by_column, observed=True)[count_column]
#         .nunique()
#         .reset_index()
#     )
#     aggregated_data.columns = [group_by_column, 'count']
#     aggregated_data = aggregated_data.sort_values(by='count', ascending=False)
#     return aggregated_data

@log_time
def create_bar_chart(data, x_column, y_column, title=None, orientation='v', color=None, text=None):
    return px.bar(
        data,
        x=x_column,
        y=y_column,
        orientation=orientation,
        title=title,
        color=color,
        category_orders={y_column: data[y_column].tolist()},
        height=700,
        text=text,
    )

@log_time
def prepare_demographics(data):
    age_bins = [0, 18, 25, 30, 40, 50, 60, 70, 100]
    age_labels = ['<18', '18-25', '25-30', '30-40', '40-50', '50-60', '60-70', '70+']
    data = data.copy()
    data['age_group'] = pd.cut(data['age'], bins=age_bins, labels=age_labels)

    age_group_counts = data['age_group'].value_counts().reset_index()
    age_group_counts.columns = ['age_group', 'count']
    gender_counts = data['gender'].replace({"": "Not Provided"}).str.upper().value_counts().reset_index()
    gender_counts.columns = ['gender', 'count']

    return age_group_counts.sort_values(by='count', ascending=False), gender_counts.sort_values(by='count',
                                                                                                ascending=False)

@log_time
def create_pie_chart(data, names_column, values_column, title=None, color_map=None):
    # Define color mapping for FEMALE and MALE
    return px.pie(
        data,
        names=names_column,
        values=values_column,
        title=title,
        color=names_column,  # Specify the column for color mapping
        color_discrete_map=color_map  # Apply the color mapping
    )

@log_time
def get_top_items(data, item_type):
    top_items = (
        data[data['type'] == item_type]['value']
        .str.upper()
        .dropna()
        .value_counts()
        .reset_index()
    )
    top_items.columns = [item_type, 'count']
    return top_items

@log_time
def analyze_observation_by_gender(data):
    observation_gender = (
        data[data['type'] == 'Observation']
        .dropna(subset=['value', 'gender'])
        .assign(value=lambda df: df['value'].str.upper())
        .assign(gender=lambda df: df['gender'].str.upper())
        .groupby(['value', 'gender'],observed=True)
        .size()
        .reset_index(name='count')
    )
    observation_gender['total'] = observation_gender.groupby('value')['count'].transform('sum')
    observation_gender = observation_gender.sort_values(by='total', ascending=False).drop('total', axis=1)

    return observation_gender

@log_time
def analyze_diagnostics_by_gender(data):
    diagnostics_gender = (
        data[data['type'] == 'Diagnostic']
        .dropna(subset=['value', 'gender'])
        .assign(value=lambda df: df['value'].str.upper())
        .assign(gender=lambda df: df['gender'].str.upper())
        .groupby(['value', 'gender'])
        .size()
        .reset_index(name='count')
    )
    diagnostics_gender['total'] = diagnostics_gender.groupby('value')['count'].transform('sum')
    diagnostics_gender = diagnostics_gender.sort_values(by='total', ascending=False).drop('total', axis=1)

    return diagnostics_gender

@log_time
def analyze_pharma_data(filtered_data):
    """
    Analyze pharma data to extract top manufacturers and primary uses.
    Handles cases where data is missing or invalid.
    """
    # Fill missing values in primary_use with an empty string for processing
    filtered_data['primary_use'] = filtered_data['primary_use'].fillna("").astype(str)

    # Top manufacturers (always calculated if manufacturers are present)
    if not filtered_data['manufacturers'].dropna().empty:
        top_manufacturers = (
            filtered_data['manufacturers']
            .str.upper()
            .dropna()
            .value_counts()
            .reset_index()
        )
        top_manufacturers.columns = ['manufacturers', 'count']
    else:
        top_manufacturers = pd.DataFrame(columns=['manufacturers', 'count'])

    # Explode primary_use for detailed analysis
    filtered_primary_use = filtered_data[filtered_data['primary_use'].str.strip() != ""]
    if not filtered_primary_use.empty:
        exploded_primary_use = (
            filtered_primary_use['primary_use']
            .str.split('|', expand=True)
            .stack()
            .str.upper()
            .str.strip()
        )
        if not exploded_primary_use.empty:
            top_primary_uses = (
                exploded_primary_use
                .reset_index(level=1, drop=True)
                .rename("primary_use")
                .value_counts()
                .reset_index()
            )
            top_primary_uses.columns = ['primary_use', 'count']
        else:
            top_primary_uses = pd.DataFrame(columns=['primary_use', 'count'])
    else:
        top_primary_uses = pd.DataFrame(columns=['primary_use', 'count'])

    return top_manufacturers, top_primary_uses

@log_time
def visualize_data_types(tab, data):
    with tab:
        with st.expander("Distribution of Data Types within Rx"):
            type_counts = data['type'].str.capitalize().value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']

            col1, col2 = st.columns([3, 1])
            with col1:
                st.plotly_chart(create_pie_chart(type_counts, 'Type', 'Count'))
            with col2:
                st.dataframe(type_counts)
                total = type_counts['Count'].sum()
                st.metric("Total", total)
        with st.expander("Distribution of Speciality Doctors"):
            speciality_counts = data.groupby('speciality')['doctor_id'].nunique().reset_index()
            speciality_counts.columns = ['Speciality', 'Count']

            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(create_pie_chart(speciality_counts, 'Speciality', 'Count'))
            with col2:
                st.dataframe(speciality_counts.sort_values(by='Count', ascending=False).reset_index(drop=True))
                total = speciality_counts['Count'].sum()
                st.metric("Total", total)

@log_time
def preprocess_column(data, column_name):
    """
    Preprocess a column to handle comma- and slash-separated values and ensure each value is treated as distinct.
    """
    if column_name in data.columns:
        # Split values on commas or slashes, trim whitespace, and explode into separate rows
        data = data.assign(
            **{column_name: data[column_name].str.split(r'[,/]').apply(
                lambda x: [v.strip() for v in x] if isinstance(x, list) else [])}
        ).explode(column_name)
    return data

@log_time
def visualize_geographical_distribution(tab, data):
    with tab:
        # Preprocess 'state_name' and 'city' columns to handle comma-separated values
        data = preprocess_column(data, 'state_name')
        data = preprocess_column(data, 'city')

        with st.expander("Patient Distribution by State"):
            patient_state_counts = aggregate_geo_data(data, 'state_name', 'id')
            col1, col2 = st.columns([3, 1])
            with col1:
                st.plotly_chart(
                    create_bar_chart(patient_state_counts.head(15), 'count', 'state_name', orientation='h',
                                     text='count', color='count'),
                    use_container_width=True,
                    key="patient_state_chart"
                )
            with col2:
                st.dataframe(patient_state_counts.reset_index(drop=True), key="patient_state_table")
                total = patient_state_counts['count'].sum()
                st.metric("Total", total)

        with st.expander("Patient Distribution by City"):
            patient_city_counts = aggregate_geo_data(data, 'city', 'id')
            col3, col4 = st.columns([3, 1])
            with col3:
                st.plotly_chart(
                    create_bar_chart(patient_city_counts.head(25), 'count', 'city', orientation='h', text='count',
                                     color='count'),
                    use_container_width=True,
                    key="patient_city_chart"
                )
            with col4:
                st.dataframe(patient_city_counts.reset_index(drop=True), key="patient_city_table")
                total = patient_city_counts['count'].sum()
                st.metric("Total", total)

        with st.expander("Doctor Distribution by State"):
            doctor_state_counts = aggregate_geo_data(data, 'state_name', 'doctor_id')
            col5, col6 = st.columns([3, 1])
            with col5:
                st.plotly_chart(
                    create_bar_chart(doctor_state_counts.head(15), 'count', 'state_name', orientation='h', text='count',
                                     color='count'),
                    use_container_width=True,
                    key="doctor_state_chart"
                )
            with col6:
                st.dataframe(doctor_state_counts.reset_index(drop=True), key="doctor_state_table")
                total = doctor_state_counts['count'].sum()
                st.metric("Total", total)

        with st.expander("Doctor Distribution by City"):
            doctor_city_counts = aggregate_geo_data(data, 'city', 'doctor_id')
            col7, col8 = st.columns([3, 1])
            with col7:
                st.plotly_chart(
                    create_bar_chart(doctor_city_counts.head(25), 'count', 'city', orientation='h', text='count',
                                     color='count'),
                    use_container_width=True,
                    key="doctor_city_chart"
                )
            with col8:
                st.dataframe(doctor_city_counts.reset_index(drop=True), key="doctor_city_table")
                total = doctor_city_counts['count'].sum()
                st.metric("Total", total)

@log_time
def visualize_patient_demographics(tab, data):
    with tab:
        data = data.drop_duplicates(subset=['id'])
        age_group_counts, gender_counts = prepare_demographics(data)
        age_group_counts = age_group_counts.sort_values('age_group')

        with st.expander("Age Group Distribution of Patients"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.plotly_chart(create_pie_chart(age_group_counts, 'age_group', 'count'))
            with col2:
                st.dataframe(age_group_counts.sort_values(by='age_group', ascending=True).reset_index(drop=True))
                total = age_group_counts['count'].sum()
                st.metric("Total", total)

        with st.expander("Gender Distribution of Patients"):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.plotly_chart(create_pie_chart(gender_counts, 'gender', 'count',
                                                 color_map={'FEMALE': '#FF69B4', 'MALE': '#0F52BA'}))
            with col2:
                st.dataframe(gender_counts)
                total = gender_counts['count'].sum()
                st.metric("Total", total)

# @log_time
# def visualize_medicines(tab, data):
#     data = data[data['type'].str.lower() == 'medicine'].copy()
#     data['value'] = data['value'].str.strip().str.upper()
#     data = data.dropna(subset=['value'])
#
#     with tab:
#         with st.expander("Top Medicines"):
#             top_medicines = get_top_items(data, 'Medicine')
#             col1, col2 = st.columns([3, 1])
#             with col1:
#                 st.plotly_chart(
#                     create_bar_chart(top_medicines.head(20), 'count', 'Medicine', orientation='h', text='count',
#                                      color='count'))
#             with col2:
#                 st.dataframe(top_medicines)
#                 total = top_medicines['count'].sum()
#                 st.metric("Total", total)
#
#         # Clean and explode primary use data
#
#         with st.expander("Top Medicines by Primary Use"):
#             data['primary_use'] = data['primary_use'].fillna("").astype(str)
#             exploded_data = data.copy()
#             exploded_data = exploded_data[exploded_data['primary_use'].str.strip() != ""]
#             exploded_data = exploded_data.assign(
#                 exploded_primary_use=exploded_data['primary_use'].str.split('|')
#             ).explode('exploded_primary_use')
#             exploded_data['exploded_primary_use'] = exploded_data['exploded_primary_use'].str.strip().str.upper()
#
#             # Get unique primary uses for selection
#             unique_primary_uses = exploded_data['exploded_primary_use'].dropna().unique()
#             unique_primary_uses = sorted(unique_primary_uses)
#             selected_primary_use = st.selectbox("Select Primary Use", unique_primary_uses, key="primary_use_select")
#
#             if not selected_primary_use:
#                 st.info("Please select a primary use to view top medicines.")
#                 return
#
#             # Filter data for the selected primary use
#             filtered_data = exploded_data[exploded_data['exploded_primary_use'] == selected_primary_use]
#
#             # Group by medicine and calculate counts
#             top_medicines = (
#                 filtered_data[filtered_data['type'] == 'Medicine']
#                 .groupby('value')
#                 .size()
#                 .reset_index(name='count')
#                 .sort_values(by='count', ascending=False)
#             )
#
#             # Display chart and table
#             col1, col2 = st.columns([3, 1])
#             with col1:
#                 st.plotly_chart(
#                     create_bar_chart(top_medicines.head(10), 'count', 'value', orientation='h', text='count',
#                                      title=f"Top Medicines for {selected_primary_use}"),
#                     use_container_width=True,
#                     key="top_medicines_chart"
#                 )
#             with col2:
#                 st.dataframe(top_medicines.reset_index(drop=True))
#                 total = top_medicines['count'].sum()
#                 st.metric("Total", total)
@st.cache_data
def _explode_primary_uses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cacheable helper: split and explode the 'primary_use' column into uppercase, stripped values.
    """
    df2 = df[['value', 'primary_use']].copy()
    # Normalize and split
    df2['primary_use'] = df2['primary_use'].fillna("")
    df2['primary_use'] = df2['primary_use'].str.split('|')
    # Explode and clean
    df2 = df2.explode('primary_use')
    df2['primary_use'] = df2['primary_use'].str.strip().str.upper()
    # Drop blanks
    return df2[df2['primary_use'] != ""]

@log_time
def visualize_medicines(tab, data: pd.DataFrame):
    # Pre-filter and normalize
    med = data.loc[data['type'].str.lower() == 'medicine', ['value', 'primary_use']].copy()
    med['value'] = med['value'].str.strip().str.upper()
    med = med[med['value'] != ""].dropna(subset=['value'])

    with tab:
        # Top Medicines overall
        with st.expander("Top Medicines"):
            top_med = (
                med['value']
                .value_counts()
                .reset_index(name='count')
            )
            top_med.columns = ['Medicine', 'count']

            c1, c2 = st.columns([3, 1])
            with c1:
                fig = px.bar(
                    top_med.head(20),
                    x='count',
                    y='Medicine',
                    orientation='h',
                    title="Top 20 Medicines",
                    text='count'
                )
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(top_med)
                st.metric("Total", top_med['count'].sum())

        # Top by primary use
        with st.expander("Top Medicines by Primary Use"):
            exploded = _explode_primary_uses(med)
            uses = sorted(exploded['primary_use'].unique())
            choice = st.selectbox("Select Primary Use", uses, key="primary_use_select")
            if not choice:
                st.info("Please select a primary use to view top medicines.")
            else:
                subset = exploded[exploded['primary_use'] == choice]
                counts = (
                    subset['value']
                    .value_counts()
                    .reset_index(name='count')
                )
                counts.columns = ['Medicine', 'count']

                c1, c2 = st.columns([3, 1])
                with c1:
                    fig = px.bar(
                        counts.head(10),
                        x='count',
                        y='Medicine',
                        orientation='h',
                        title=f"Top Medicines for {choice}",
                        text='count'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.dataframe(counts)
                    st.metric("Total", counts['count'].sum())


@log_time
def visualize_pharma_analytics(tab, filtered_medical_data):
    with tab:
        top_15_manufacturers, top_15_primary_uses = analyze_pharma_data(filtered_medical_data)

        # Expander for Top Manufacturers
        with st.expander("Top Manufacturers"):
            if top_15_manufacturers is not None and not top_15_manufacturers.empty:
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.plotly_chart(create_pie_chart(top_15_manufacturers.head(15), 'manufacturers', 'count'))
                with col2:
                    st.dataframe(top_15_manufacturers)
                    total = top_15_manufacturers['count'].sum()
                    st.metric("Total", total)
            else:
                st.warning("No data available for Top Manufacturers.")

        # Expander for Top Primary Uses
        with st.expander("Top Primary Uses"):
            if top_15_primary_uses is not None and not top_15_primary_uses.empty:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.plotly_chart(
                        create_bar_chart(top_15_primary_uses.head(15), 'count', 'primary_use', orientation='h',
                                         text='count', color='count'))
                with col2:
                    st.dataframe(top_15_primary_uses)
                    total = top_15_primary_uses['count'].sum()
                    st.metric("Total", total)
            else:
                st.warning("No data available for Top Primary Uses.")

        # # Expander for Manufacturers by Primary Use
        # with st.expander("Manufacturers by Primary Use"):
        #     if top_15_primary_uses is not None and not top_15_primary_uses.empty:
        #         # Create a dropdown to select a primary use
        #         selected_primary_use = st.selectbox("Select Primary Use", top_15_primary_uses['primary_use'].unique())

        #         # Filter the data for the selected primary use
        #         filtered_data = filtered_medical_data[
        #             filtered_medical_data['primary_use'].str.contains(selected_primary_use, case=False, na=False)]

        #         if not filtered_data.empty:
        #             # Group by manufacturers and count occurrences
        #             manufacturer_counts = (
        #                 filtered_data.groupby('manufacturers')
        #                 .size()
        #                 .reset_index(name='count')
        #                 .sort_values(by='count', ascending=False)
        #             )
        #             col1, col2 = st.columns([3, 1])
        #             with col1:
        #                 st.plotly_chart(
        #                     create_bar_chart(manufacturer_counts.head(10), 'count', 'manufacturers', orientation='h',
        #                                      text='count', color='count'))
        #             with col2:
        #                 st.dataframe(manufacturer_counts.reset_index(drop=True))
        #                 total = manufacturer_counts['count'].sum()
        #                 st.metric("Total", total)
        #         else:
        #             st.warning(f"No data available for the selected primary use: {selected_primary_use}")
        #     else:
        #         st.warning("No data available for Manufacturers by Primary Use.")

@log_time
def visualize_observations(tab, data):
    data = data[data['type'].str.lower() == 'observation'].copy()

    data['value'] = data['value'].str.strip().str.upper()
    data = data.dropna(subset=['value'])
    data = data[~data.value.isna()]
    data = data[data['value'].str.strip() != ""]
    with tab:
        with st.expander("Top Observations"):
            top_observations = get_top_items(data, 'Observation')
            col1, col2 = st.columns([3, 1])
            with col1:
                st.plotly_chart(
                    create_bar_chart(top_observations.head(20), 'count', 'Observation', orientation='h', text='count',
                                     color='count'))
            with col2:
                st.dataframe(top_observations)
                total = top_observations['count'].sum()
                st.metric("Total", total)

        with st.expander("Observations by Gender"):
            observations_gender = analyze_observation_by_gender(data)
            observations_gender['Total'] = observations_gender.groupby('value')['count'].transform('sum')
            observations_gender = observations_gender.sort_values(by='Total', ascending=False)
            observations_pivot = observations_gender.pivot(index='value', columns='gender', values='count').fillna(0)
            observations_pivot['Total'] = observations_pivot.sum(axis=1)
            observations_pivot = observations_pivot.sort_values(by='Total', ascending=False)

            col1, col2 = st.columns([70, 30])
            with col1:
                st.plotly_chart(create_bar_chart(
                    observations_pivot.head(20).reset_index().drop(columns='Total').melt(id_vars='value',
                                                                                         var_name='gender',
                                                                                         value_name='count'),
                    'count',
                    'value',
                    orientation='h',
                    color='gender',
                    text='count'
                ))
            with col2:
                st.dataframe(observations_pivot)

@log_time
def visualize_diagnostics(tab, data):
    data = data[data['type'].str.lower() == 'diagnostic'].copy()
    data['value'] = data['value'].str.strip().str.upper()
    data = data.dropna(subset=['value'])
    data = data[~data.value.isna()]
    data = data[data['value'].str.strip() != ""]
    with tab:
        with st.expander("Top Diagnostics"):
            top_diagnostics = get_top_items(data, 'Diagnostic')
            col1, col2 = st.columns([3, 1])
            with col1:
                st.plotly_chart(
                    create_bar_chart(top_diagnostics.head(20), 'count', 'Diagnostic', orientation='h', text='count',
                                     color='count'),
                    use_container_width=True,
                    key="top_diagnostics_chart"
                )
            with col2:
                st.dataframe(top_diagnostics, key="top_diagnostics_table")
                total = top_diagnostics['count'].sum()
                st.metric("Total", total)

        with st.expander("Diagnostics by Gender"):
            diagnostics_gender = analyze_diagnostics_by_gender(data)
            diagnostics_gender['Total'] = diagnostics_gender.groupby('value')['count'].transform('sum')
            diagnostics_gender = diagnostics_gender.sort_values(by='Total', ascending=False)
            diagnostics_pivot = diagnostics_gender.pivot(index='value', columns='gender', values='count').fillna(0)
            diagnostics_pivot['Total'] = diagnostics_pivot.sum(axis=1)
            diagnostics_pivot = diagnostics_pivot.sort_values(by='Total', ascending=False)

            col1, col2 = st.columns([70, 30])
            with col1:
                st.plotly_chart(
                    create_bar_chart(
                        diagnostics_pivot.head(15).reset_index().drop(columns='Total').melt(id_vars='value',
                                                                                            var_name='gender',
                                                                                            value_name='count'),
                        'count',
                        'value',
                        orientation='h',
                        color='gender',
                        text='count'
                    ),
                    use_container_width=True,
                    key="diagnostics_by_gender_chart"
                )
            with col2:
                st.dataframe(diagnostics_pivot, key="diagnostics_by_gender_table")

@log_time
def visualize_manufacturer_medicines(tab, data):
    with tab:
        st.subheader("Medicines by Manufacturer")
        top_15_manufacturers, _ = analyze_pharma_data(data)

        if top_15_manufacturers is not None and not top_15_manufacturers.empty:
            # Sort the manufacturers list alphabetically
            top_manufacturers_list = sorted(top_15_manufacturers['manufacturers'].tolist())
            default_index = top_manufacturers_list.index("LUPIN LTD") if "LUPIN LTD" in top_manufacturers_list else 0

            # Display manufacturer selection box
            selected_manufacturer = st.selectbox(
                "Select Manufacturer",
                top_manufacturers_list,
                index=default_index
            )

            if selected_manufacturer:
                # Filter data for the selected manufacturer
                manufacturer_data = data[data['manufacturers'].str.upper() == selected_manufacturer.upper()]

                if not manufacturer_data.empty:
                    medicine_counts = (
                        manufacturer_data[manufacturer_data['type'] == 'Medicine']['value']
                        .dropna()
                        .str.upper()
                        .value_counts()
                        .reset_index()
                    )
                    medicine_counts.columns = ['Medicine', 'Count']

                    col1, col2 = st.columns([60, 40])

                    with col1:
                        st.plotly_chart(
                            create_pie_chart(medicine_counts.head(10), 'Medicine', 'Count',
                                             f"Medicines by {selected_manufacturer}")
                        )

                    with col2:
                        st.dataframe(medicine_counts)
                        total = medicine_counts['Count'].sum()
                        col3, col4 = st.columns([1, 1])
                        with col3:
                            st.metric("Total", total)
                        with col4:
                            st.metric("Strike Rate(%)",
                                      f"{(((total / (data['type'].str.lower().eq('medicine').sum())).round(4)) * 100).round(2)}%")
                            st.text("Strike Rate: % of medicines prescribed by this manufacturer out of total.")
                else:
                    st.warning(f"No data available for the selected manufacturer: {selected_manufacturer}.")
        else:
            st.warning("No manufacturer data available.")

@log_time
def manufacturer_comparison_tab(tab, data):
    with tab:
        st.subheader("Manufacturer Comparison")

        # Select manufacturers to compare
        manufacturers = data['manufacturers'].dropna().unique()
        selected_manufacturers = st.multiselect("Select Manufacturers for Comparison", sorted(manufacturers))

        if not selected_manufacturers:
            st.info("Select at least one manufacturer to view the comparison.")
            return

        # Filter data for selected manufacturers
        filtered_data = data[data['manufacturers'].isin(selected_manufacturers)]
        filtered_data = filtered_data[filtered_data['primary_use'].str.strip() != ""]

        # Explode the primary_use column into multiple rows
        exploded_data = filtered_data.copy()
        exploded_data = exploded_data.dropna(subset=['primary_use'])
        exploded_data = exploded_data.assign(
            exploded_primary_use=exploded_data['primary_use'].str.split('|')
        ).explode('exploded_primary_use')
        exploded_data['exploded_primary_use'] = exploded_data['exploded_primary_use'].str.strip().str.upper()

        # Get unique primary uses for selection
        unique_primary_uses = sorted(exploded_data['exploded_primary_use'].dropna().unique())
        selected_primary_uses = st.multiselect("Select Primary Uses for Comparison", unique_primary_uses)

        if not selected_primary_uses:
            st.info("Select at least one primary use to view the comparison.")
            return

        # Filter data for selected primary uses
        filtered_data = exploded_data[exploded_data['exploded_primary_use'].isin(selected_primary_uses)]

        # **Bar Chart - Total Medicines per Manufacturer**
        manufacturer_counts = filtered_data.groupby('manufacturers')['id'].count().reset_index()
        manufacturer_counts.columns = ['Manufacturer', 'Total Medicines']

        fig_pie = px.pie(
            manufacturer_counts,
            names='Manufacturer',
            values='Total Medicines',
            title='Total Medicines per Manufacturer',
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        # Create separate pie charts for each primary use
        primary_use_data = filtered_data.groupby(['exploded_primary_use', 'manufacturers'])['id'].count().reset_index()
        primary_use_data.columns = ['Primary Use', 'Manufacturer', 'Count']

        # Get unique primary uses
        unique_primary_uses = primary_use_data['Primary Use'].unique()

        # Create multiple columns for pie charts
        cols = st.columns(2)  # 2 charts per row
        for i, primary_use in enumerate(unique_primary_uses):
            # Filter data for current primary use
            use_data = primary_use_data[primary_use_data['Primary Use'] == primary_use]

            # Create pie chart
            fig = px.pie(
                use_data,
                names='Manufacturer',
                values='Count',
                title=f'Distribution for {primary_use}',
                hole=0.4
            )

            # Display chart in alternating columns
            with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)

        # **Data Table: Medicines per Manufacturer and Primary Use**
        for manufacturer in selected_manufacturers:
            st.write(f"### Manufacturer: {manufacturer}")
            manufacturer_data = filtered_data[filtered_data['manufacturers'] == manufacturer]

            # Display total number of medicines for the manufacturer
            total_medicines = manufacturer_data.shape[0]
            st.write(f"**Total Medicines:** {total_medicines}")

            for primary_use in selected_primary_uses:
                st.write(f"**Primary Use: {primary_use}**")
                use_data = manufacturer_data[manufacturer_data['exploded_primary_use'] == primary_use]

                # Group by medicine and calculate metrics
                metrics = (
                    use_data.groupby('value')
                    .agg(
                        Medicine_Count=('id', 'count'),
                        Unique_Patients=('id', 'nunique'),
                    )
                    .reset_index()
                )

                # Add percentage column for medicine counts
                if not metrics.empty:
                    metrics['% Medicine Count'] = (
                            metrics['Medicine_Count'] / metrics['Medicine_Count'].sum() * 100).round(2)

                    # Display table with updated columns
                    st.dataframe(
                        metrics.sort_values(by='Medicine_Count', ascending=False)
                        .reset_index(drop=True)
                        .rename(columns={
                            'value': 'Medicine',
                            'Medicine_Count': 'Total Medicines',
                            'Unique_Patients': 'Unique Patients'
                        })
                    )
                else:
                    st.write("No data available for this primary use.")

@log_time
def visualize_market_share_primary_use(tab, data):
    with tab:
        st.subheader("Market Share Comparison by Manufacturers for a Primary Use")

        # Explode the primary_use column into multiple rows
        exploded_data = data.copy()
        exploded_data = exploded_data.dropna(subset=['primary_use'])
        exploded_data = exploded_data.assign(
            exploded_primary_use=exploded_data['primary_use'].str.split('|')
        ).explode('exploded_primary_use')
        exploded_data['exploded_primary_use'] = exploded_data['exploded_primary_use'].str.strip().str.upper()

        # Get unique primary uses for selection and remove blanks
        unique_primary_uses = exploded_data['exploded_primary_use']
        unique_primary_uses = unique_primary_uses[unique_primary_uses != ""].unique()

        selected_primary_uses = st.multiselect("Select Primary Uses", sorted(unique_primary_uses))

        if not selected_primary_uses:
            st.info("Select at least one primary use to view the market share comparison.")
            return

        # Filter data for the selected primary uses
        filtered_data = exploded_data[exploded_data['exploded_primary_use'].isin(selected_primary_uses)]

        # Calculate the market share of each manufacturer
        manufacturer_market_share = (
            filtered_data.groupby('manufacturers')
            .agg(Count=('value', 'count'))
            .reset_index()
        )
        manufacturer_market_share['Share%'] = (
                manufacturer_market_share['Count'] / manufacturer_market_share['Count'].sum() * 100
        ).round(2)

        # Sort and prepare data for the pie chart
        manufacturer_market_share = manufacturer_market_share.sort_values(
            by='Share%', ascending=False
        ).reset_index(drop=True)

        st.subheader(f"Primary Uses: {', '.join(selected_primary_uses)}")
        if not manufacturer_market_share.empty:
            fig = px.pie(
                manufacturer_market_share.head(20),
                names='manufacturers',
                values='Share%',
                hole=0.4, hover_data=['Count'],
            )
            col1, col2 = st.columns([60, 40])
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.dataframe(manufacturer_market_share)
                total = manufacturer_market_share['Count'].sum()
                st.metric("Total", total)
        else:
            st.warning("No data available for the selected primary uses.")

@log_time
def visualize_value_comparison(tab, data):
    """
    Creates a tab for value-based comparison of manufacturers.
    """
    with tab:
        st.subheader("Value-Based Manufacturer Comparison")

        # Group data by manufacturers
        manufacturer_comparison = (
            data.groupby('manufacturers')
            .agg(
                Total_Value=('average_mrp', 'sum'),  # Replace with relevant column
                Average_Value=('average_mrp', 'mean'),  # Replace with relevant column
                Patient_Count=('id', 'nunique')
            )
            .reset_index()
        )

        # Get top 20 for charts
        top_20 = manufacturer_comparison.sort_values(by='Total_Value', ascending=False).head(20)

        # Create toggles for viewing different metrics
        toggle_option = st.radio(
            "Select Metric to Compare:",
            options=['Total Value', 'Average Value', 'Patient Count'],
            horizontal=True
        )

        # Display the selected metric as a bar chart
        if toggle_option == 'Total Value':
            top_20 = manufacturer_comparison.sort_values(by='Total_Value', ascending=False).head(20)

            fig = px.bar(
                top_20,
                x='manufacturers',
                y='Total_Value',
                title="Total Value by Manufacturer",
                labels={'Total_Value': 'Total Value'},
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)


        elif toggle_option == 'Average Value':
            top_20 = manufacturer_comparison.sort_values(by='Average_Value', ascending=False).head(20)
            fig = px.bar(
                top_20,
                x='manufacturers',
                y='Average_Value',
                title="Average Value by Manufacturer",
                labels={'Average_Value': 'Average Value'},
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)

        elif toggle_option == 'Patient Count':
            top_20 = manufacturer_comparison.sort_values(by='Patient_Count', ascending=False).head(20)
            fig = px.bar(
                top_20,
                x='manufacturers',
                y='Patient_Count',
                title="Patient Count by Manufacturer",
                labels={'Patient_Count': 'Patient Count'},
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Display the data as a table
        st.dataframe(
            manufacturer_comparison
            .sort_values(by='Total_Value', ascending=False).reset_index(drop=True)
            .assign(
                Total_Value_Percentage=lambda df: (df['Total_Value'] / df['Total_Value'].sum() * 100).round(2),
                Patient_Count_Percentage=lambda df: (df['Patient_Count'] / df['Patient_Count'].sum() * 100).round(2))
        )

@log_time
def visualize_vitals(tab, data):
    with tab:
        st.subheader("Vital Sign Analysis")

        # Get unique vitals
        available_vitals = ["Blood pressure (BP)", "Pulse", "Weight"]
        # available_vitals = sorted(data['vital_type'].dropna().unique())

        # User selects which vital to visualize
        selected_vital = st.selectbox("Select a Vital to View", available_vitals)

        if not selected_vital:
            st.warning("Please select a vital sign to view.")
            return

        vital_data = data[data['vital_type'] == selected_vital].copy()

        if vital_data.empty:
            st.warning(f"No valid data available for {selected_vital} visualization.")
            return

        # Function to remove outliers using IQR
        def remove_outliers(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 10 * IQR
            upper_bound = Q3 + 10 * IQR
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

        vital_data['age'] = pd.to_numeric(vital_data['age'], errors='coerce')
        age_bins = [0, 18, 25, 40, 60, 200]
        age_labels = ["0-18", "19-25", "26-40", "41-60", "60+"]
        vital_data['age_group'] = pd.cut(vital_data['age'], bins=age_bins, labels=age_labels, include_lowest=True)
        vital_data['age_group'] = vital_data['age_group'].cat.add_categories("Unknown").fillna("Unknown")

        if 'gender' not in vital_data.columns or vital_data['gender'].isnull().all():
            vital_data['gender'] = 'Unknown'

        # **Special Handling for Blood Pressure (BP)**
        if selected_vital == "Blood pressure (BP)":
            vital_data = vital_data[vital_data['vital_type'] == 'Blood pressure (BP)'].copy()
            vital_data['value'] = vital_data['value'].astype(str)

            # Clean BP values
            vital_data['value'] = (
                vital_data['value'].str.lower()
                .str.replace('mmhg', '', regex=True)
                .str.replace('mm/hg', '', regex=True)
                .str.strip()
            )

            # Ensure all BP values contain both systolic and diastolic readings
            valid_bp_rows = vital_data['value'].str.fullmatch(r'\d{2,3}/\d{2,3}')
            vital_data = vital_data[valid_bp_rows].copy()

            if vital_data.empty:
                st.warning("No valid blood pressure readings found.")
                return

            # Split into systolic and diastolic
            vital_data[['systolic', 'diastolic']] = vital_data['value'].str.split('/', expand=True).astype(int)

            # Remove outliers
            remove_outliers(vital_data, 'systolic')
            remove_outliers(vital_data, 'diastolic')

            # **Visualization for Blood Pressure**
            st.subheader("Blood Pressure Distribution")
            total_data_points = len(vital_data)
            st.write(f"**Total Data Points Available:** {total_data_points}")

            # **Box Plot - Overall**
            with st.expander("Overall Blood Pressure Distribution"):
                # Add summary statistics
                overall_summary = pd.DataFrame({
                    'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Systolic': [
                        f"{vital_data['systolic'].count()}",
                        f"{vital_data['systolic'].mean():.1f}",
                        f"{vital_data['systolic'].median():.1f}",
                        f"{vital_data['systolic'].std():.1f}",
                        f"{vital_data['systolic'].min():.1f}",
                        f"{vital_data['systolic'].max():.1f}"
                    ],
                    'Diastolic': [
                        f"{vital_data['diastolic'].count()}",
                        f"{vital_data['diastolic'].mean():.1f}",
                        f"{vital_data['diastolic'].median():.1f}",
                        f"{vital_data['diastolic'].std():.1f}",
                        f"{vital_data['diastolic'].min():.1f}",
                        f"{vital_data['diastolic'].max():.1f}"
                    ]
                })
                st.write("Summary Statistics:")
                st.dataframe(overall_summary)

                fig_overall = px.box(
                    vital_data.melt(id_vars=['gender', 'age_group'], value_vars=['systolic', 'diastolic'],
                                    var_name='BP Type', value_name='BP Value'),
                    x='BP Type',
                    y='BP Value',
                    labels={'BP Type': 'Blood Pressure Type', 'BP Value': 'Blood Pressure (mmHg)'},
                    boxmode="group"
                )
                st.plotly_chart(fig_overall, use_container_width=True)

            with st.expander("Blood Pressure Distribution by Gender"):
                # Add gender-wise summary
                gender_summary = vital_data.groupby('gender').agg({
                    'systolic': ['count', 'mean', 'median', 'std', 'min', 'max'],
                    'diastolic': ['count', 'mean', 'median', 'std', 'min', 'max']
                }).round(2)

                st.write("Gender-wise Summary Statistics:")
                st.dataframe(gender_summary)

                fig_gender = px.box(
                    vital_data.melt(id_vars=['gender'],
                                    value_vars=['systolic', 'diastolic'],
                                    var_name='BP Type',
                                    value_name='BP Value'),
                    x='BP Type',
                    y='BP Value',
                    color='gender',
                    labels={'BP Type': 'Blood Pressure Type',
                            'BP Value': 'Blood Pressure (mmHg)'},
                    boxmode="group"
                )
                st.plotly_chart(fig_gender, use_container_width=True)

            with st.expander("Blood Pressure Distribution by Age Groups"):
                # Add age-wise summary
                age_summary = vital_data.groupby('age_group',observed=True).agg({
                    'systolic': ['count', 'mean', 'median', 'std', 'min', 'max'],
                    'diastolic': ['count', 'mean', 'median', 'std', 'min', 'max']
                }).round(1)
                age_summary = pd.DataFrame(age_summary).sort_values(by='age_group').reset_index()
                st.write("Age-wise Summary Statistics:")
                st.dataframe(age_summary)

                fig_age = px.box(
                    vital_data.melt(id_vars=['age_group'],
                                    value_vars=['systolic', 'diastolic'],
                                    var_name='BP Type',
                                    value_name='BP Value').sort_values(by='age_group'),
                    x='BP Type',
                    y='BP Value',
                    color='age_group',
                    labels={'BP Type': 'Blood Pressure Type',
                            'BP Value': 'Blood Pressure (mmHg)'},
                    boxmode="group"
                )
                st.plotly_chart(fig_age, use_container_width=True)

        elif selected_vital == "Pulse":
            vital_data['value'] = vital_data['value'].astype(str)
            vital_data['value'] = vital_data['value'].str.extract(r'(\d+\.?\d*)')  # Extract numeric values
            vital_data['value'] = pd.to_numeric(vital_data['value'], errors='coerce')

            # Remove outliers for Pulse
            remove_outliers(vital_data, 'value')

            st.subheader("Pulse Rate Distribution")
            total_data_points = len(vital_data)
            st.write(f"**Total Data Points Available:** {total_data_points}")

            # **Box Plot - Overall**
            with st.expander("Overall Pulse Rate Distribution"):
                overall_summary = pd.DataFrame({
                    'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'Pulse Rate': [
                        f"{vital_data['value'].count()}",
                        f"{vital_data['value'].mean():.1f}",
                        f"{vital_data['value'].median():.1f}",
                        f"{vital_data['value'].std():.1f}",
                        f"{vital_data['value'].min():.1f}",
                        f"{vital_data['value'].max():.1f}"
                    ]
                })
                st.write("Summary Statistics:")
                st.dataframe(overall_summary)

                fig_pulse = px.box(
                    vital_data,
                    x='value',
                    labels={'value': 'Pulse Rate (BPM)'},
                    title="Pulse Rate Distribution"
                )
                st.plotly_chart(fig_pulse, use_container_width=True)

            # **Pulse Distribution by Gender**
            with st.expander("Pulse Rate Distribution by Gender"):
                gender_summary = vital_data.groupby('gender').agg({
                    'value': ['count', 'mean', 'median', 'std', 'min', 'max']
                }).round(2)

                st.write("Gender-wise Summary Statistics:")
                st.dataframe(gender_summary)

                fig_pulse_gender = px.box(
                    vital_data,
                    x='gender',
                    y='value',
                    color='gender',
                    labels={'value': 'Pulse Rate (BPM)', 'gender': 'Gender'},
                    title="Pulse Rate by Gender"
                )
                st.plotly_chart(fig_pulse_gender, use_container_width=True)

            # **Pulse Distribution by Age Groups**
            with st.expander("Pulse Rate Distribution by Age Groups"):
                vital_data = vital_data.sort_values(by='age_group').reset_index(drop=True)
                age_summary = vital_data.groupby('age_group').agg({
                    'value': ['count', 'mean', 'median', 'std', 'min', 'max']
                }).round(1)
                age_summary = pd.DataFrame(age_summary).reset_index()
                st.write("Age-wise Summary Statistics:")
                st.dataframe(age_summary)

                fig_pulse_age = px.box(
                    vital_data,
                    x='age_group',
                    y='value',
                    color='age_group',
                    labels={'value': 'Pulse Rate (BPM)', 'age_group': 'Age Group'},
                    title="Pulse Rate by Age Group"
                )
                st.plotly_chart(fig_pulse_age, use_container_width=True)

        elif selected_vital == "Weight":
            vital_data['value'] = vital_data['value'].astype(str)
            # Clean weight values to include only kg or kgs units
            weight_mask = vital_data['value'].str.contains('kg|kgs', case=False, na=True)
            vital_data = vital_data[weight_mask]
            vital_data['value'] = vital_data['value'].str.extract(r'(\d+\.?\d*)')  # Extract numeric values
            vital_data['value'] = pd.to_numeric(vital_data['value'], errors='coerce')

            remove_outliers(vital_data, 'value')

            st.subheader("Weight Distribution")
            total_data_points = len(vital_data)
            st.write(f"**Total Data Points Available:** {total_data_points}")

            # **Box Plot - Overall**
            with st.expander("Overall Weight Distribution"):
                overall_summary = vital_data.agg({
                    'value': ['count', 'mean', 'median', 'std', 'min', 'max']
                }).round(2).rename(columns={'value': 'Weight (kg)'}).reset_index().rename(columns={'index': 'Metric'})

                st.write("Summary Statistics:")
                st.dataframe(overall_summary)

                fig_weight = px.box(
                    vital_data,
                    x='value',
                    labels={'value': 'Weight (kg)'},
                    title="Weight Distribution"
                )
                st.plotly_chart(fig_weight, use_container_width=True)

            # **Weight Distribution by Gender**
            with st.expander("Weight Distribution by Gender"):
                gender_summary = vital_data.groupby('gender').agg({
                    'value': ['count', 'mean', 'median', 'std', 'min', 'max']
                }).round(2).rename(columns={'value': 'Weight (kg)'})

                st.write("Gender-wise Summary Statistics:")
                st.dataframe(gender_summary)

                fig_weight_gender = px.box(
                    vital_data,
                    x='gender',
                    y='value',
                    color='gender',
                    labels={'value': 'Weight (kg)', 'gender': 'Gender'},
                    title="Weight by Gender"
                )
                st.plotly_chart(fig_weight_gender, use_container_width=True)

            # **Weight Distribution by Age Groups**
            with st.expander("Weight Distribution by Age Groups"):
                vital_data = vital_data.sort_values(by='age_group').reset_index()
                age_summary = vital_data.groupby('age_group').agg({
                    'value': ['count', 'mean', 'median', 'std', 'min', 'max']
                }).round(2).rename(columns={'value': 'Weight (kg)'})
                age_summary = pd.DataFrame(age_summary).reset_index()
                st.write("Age-wise Summary Statistics:")
                st.dataframe(age_summary)

                fig_weight_age = px.box(
                    vital_data,
                    x='age_group',
                    y='value',
                    color='age_group',
                    labels={'value': 'Weight (kg)', 'age_group': 'Age Group'},
                    title="Weight by Age Group"
                )
                st.plotly_chart(fig_weight_age, use_container_width=True)

        elif selected_vital == 'Oxygen saturation (SpO2)':
            vital_data['value'] = vital_data['value'].astype(str)
            # Clean SpO2 values to include only percentage values
            spo2_mask = vital_data['value'].str.contains('%', case=False, na=True)
            vital_data = vital_data[spo2_mask]

            # Extract numeric values and handle ranges
            vital_data['value'] = vital_data['value'].str.split('%').str[0].str.strip()
            vital_data['value'] = vital_data['value'].apply(lambda x:
                                                            str((int(x.split('-')[0].strip()) + int(
                                                                x.split('-')[1].strip())) // 2)
                                                            if '-' in str(x) else x)

            # Limit to realistic SpO2 values (2-3 digits)
            vital_data['value'] = vital_data['value'].apply(lambda x:
                                                            x[:3] if str(x)[0] == '1' else x[:2])

            vital_data['value'] = pd.to_numeric(vital_data['value'], errors='coerce')

            # Remove outliers for SpO2
            vital_data = vital_data[(vital_data['value'] >= 0) & (vital_data['value'] <= 100)]
            remove_outliers(vital_data, 'value')

            st.subheader("Oxygen Saturation (SpO2) Distribution")
            total_data_points = len(vital_data)
            st.write(f"**Total Data Points Available:** {total_data_points}")

            # **Box Plot - Overall**
            with st.expander("Overall SpO2 Distribution"):
                overall_summary = pd.DataFrame({
                    'Metric': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    'SpO2 (%)': [
                        f"{vital_data['value'].count()}",
                        f"{vital_data['value'].mean():.1f}",
                        f"{vital_data['value'].median():.1f}",
                        f"{vital_data['value'].std():.1f}",
                        f"{vital_data['value'].min():.1f}",
                        f"{vital_data['value'].max():.1f}"
                    ]
                })
                st.write("Summary Statistics:")
                st.dataframe(overall_summary)

                fig_spo2 = px.box(
                    vital_data,
                    x='value',
                    labels={'value': 'SpO2 (%)'},
                    title="SpO2 Distribution"
                )
                st.plotly_chart(fig_spo2, use_container_width=True)

            # **SpO2 Distribution by Gender**
            with st.expander("SpO2 Distribution by Gender"):
                gender_summary = vital_data.groupby('gender').agg({
                    'value': ['count', 'mean', 'median', 'std', 'min', 'max']
                }).round(2)

                st.write("Gender-wise Summary Statistics:")
                st.dataframe(gender_summary)

                fig_spo2_gender = px.box(
                    vital_data,
                    x='gender',
                    y='value',
                    color='gender',
                    labels={'value': 'SpO2 (%)', 'gender': 'Gender'},
                    title="SpO2 by Gender"
                )
                st.plotly_chart(fig_spo2_gender, use_container_width=True)

            # **SpO2 Distribution by Age Groups**
            with st.expander("SpO2 Distribution by Age Groups"):
                vital_data = vital_data.sort_values(by='age_group').reset_index(drop=True)
                age_summary = vital_data.groupby('age_group').agg({
                    'value': ['count', 'mean', 'median', 'std', 'min', 'max']
                }).round(1)
                age_summary = pd.DataFrame(age_summary).reset_index()
                st.write("Age-wise Summary Statistics:")
                st.dataframe(age_summary)

                fig_spo2_age = px.box(
                    vital_data,
                    x='age_group',
                    y='value',
                    color='age_group',
                    labels={'value': 'SpO2 (%)', 'age_group': 'Age Group'},
                    title="SpO2 by Age Group"
                )
                st.plotly_chart(fig_spo2_age, use_container_width=True)

        else:
            st.warning(f"{selected_vital} sparse data")
@log_time
@st.cache_data
def get_unique_states(state_series: pd.Series) -> list[str]:
    # fill NaN, ensure string, split into columns, stack into one Series, strip & drop blanks
    df = (
        state_series
        .fillna("")              # no NaNs
        .astype(str)             # ensure str
        .str.split(r'[,/]', expand=True)
        .stack()                 # one big Series
        .str.strip()             # remove whitespace
    )
    df = df[df != ""]            # drop empty strings
    return sorted(df.unique().tolist())

@log_time
def get_state_filter(medical_data: pd.DataFrame):
    states = get_unique_states(medical_data['state_name'])
    return st.sidebar.multiselect(
        "Select State",
        options=states,
        default=state.get("state_filter", []),
        key="state_filter"
    )

@log_time
def get_city_filter(medical_data, state_filter):
    # Filter data for the selected states
    filtered_data = medical_data.copy()
    if state_filter:
        filtered_data['state_name'] = filtered_data['state_name'].fillna("").astype(str)
        filtered_data = filtered_data[filtered_data['state_name'].str.split(r'[,/]').apply(
            lambda x: any(state.strip() in state_filter for state in x))]

    # Split and normalize city values if they are combined
    filtered_data['city'] = filtered_data['city'].fillna("").astype(str)
    exploded_cities = filtered_data['city'].str.split(r'[,/]').explode().str.strip()
    unique_cities = exploded_cities.dropna().unique()

    return st.sidebar.multiselect(
        "Select City",
        options=sorted(unique_cities),
        default=state.get("city_filter", []),
        key="city_filter"
    )

@log_time
def get_pincode_filter(medical_data, state_filter, city_filter):
    # Filter data for the selected states and cities
    filtered_data = medical_data.copy()
    if state_filter:
        filtered_data['state_name'] = filtered_data['state_name'].fillna("").astype(str)
        filtered_data = filtered_data[filtered_data['state_name'].str.split(r'[,/]').apply(
            lambda x: any(state.strip() in state_filter for state in x))]
    if city_filter:
        filtered_data['city'] = filtered_data['city'].fillna("").astype(str)
        filtered_data = filtered_data[
            filtered_data['city'].str.split(r'[,/]').apply(lambda x: any(city.strip() in city_filter for city in x))]

    # Split and normalize pincode values if they are combined
    filtered_data['pincode'] = filtered_data['pincode'].fillna("").astype(str)
    exploded_pincodes = filtered_data['pincode'].str.split(r'[,/]').explode().str.strip()
    unique_pincodes = exploded_pincodes.dropna().unique()

    return st.sidebar.multiselect(
        "Select Pincode",
        options=sorted(unique_pincodes),
        default=state.get("pincode_filter", []),
        key="pincode_filter"
    )

@log_time
def get_speciality_filter(medical_data, pincode_filter):
    filtered_speciality_data = medical_data.copy()
    if pincode_filter:
        filtered_speciality_data = filtered_speciality_data[
            filtered_speciality_data['pincode'].str.split(',').apply(
                lambda x: any(p in pincode_filter for p in x)
            )
        ]
    unique_specialities = filtered_speciality_data['speciality'].dropna().unique()
    return st.sidebar.multiselect(
        "Select Speciality",
        options=unique_specialities,
        default=state.get("speciality_filter", []),
        key="speciality_filter"
    )

@log_time
def get_client_filter(data):
    """Extracts unique client names and provides a multi-select filter in Streamlit."""
    unique_clients = sorted(data['client'].dropna().unique())  # Get unique non-null clients
    selected_clients = st.sidebar.multiselect("Select Client(s)", unique_clients)
    return selected_clients

@log_time
def get_project_filter(data, client_filter):
    """Extracts unique project names and provides a multi-select filter in Streamlit."""
    if client_filter:
        data = data[data['client'].isin(client_filter)]
    unique_projects = sorted(data['project'].dropna().unique())  # Get unique non-null projects
    selected_projects = st.sidebar.multiselect("Select Project(s)", unique_projects)
    return selected_projects


def main():
    st.set_page_config(layout="wide", page_title="Dashboard")

    col1, col2 = st.columns([5, 1])
    with col1:
        title_placeholder = st.empty()
    with col2:
        logo_path = 'logo.png'

    path = 'data/demo_half_data.csv'
    data = load_data(path)
    cleaned_data = clean_medical_data(data)
    st.sidebar.title("Rx Analytics Filters")

    state_filter = get_state_filter(cleaned_data)
    city_filter = get_city_filter(cleaned_data, state_filter)
    pincode_filter = get_pincode_filter(cleaned_data, state_filter, city_filter)
    speciality_filter = get_speciality_filter(cleaned_data, pincode_filter)
    client_filter = get_client_filter(cleaned_data)
    project_filter = get_project_filter(cleaned_data, client_filter)

    st.sidebar.header("Analytics Time Period")
    # First row of buttons
    col1, col2 = st.sidebar.columns(2)
    start_date_val = datetime(2020, 1, 1)
    end_date_val = datetime.today()
    # Add session state keys for each button
    if "current_fy" not in state:
        state.current_fy = False
    if "previous_month" not in state:
        state.previous_month = False

    if col1.button("Current FY"):
        current_year = datetime.today().year
        state.start_date = datetime(current_year - 1, 4, 1)
        state.end_date = datetime.today()
        state.current_fy = True
        state.previous_month = False

    if col2.button("Previous Month"):
        today = datetime.today()
        first_day_current_month = datetime(today.year, today.month, 1)
        last_day_prev_month = first_day_current_month - pd.Timedelta(days=1)
        state.start_date = datetime(last_day_prev_month.year, last_day_prev_month.month, 1)
        state.end_date = last_day_prev_month
        state.previous_month = True
        state.current_fy = False

    # Second row of buttons
    col3, col4 = st.sidebar.columns([1, 1])

    if "year_to_date" not in state:
        state.year_to_date = False
    if "previous_week" not in state:
        state.previous_week = False

    if col3.button("Year To Date"):
        today = datetime.today()
        state.start_date = datetime(today.year, 1, 1)
        state.end_date = today
        state.year_to_date = True
        state.previous_week = False

    if col4.button("Previous Week"):
        today = datetime.today()
        state.start_date = today - pd.Timedelta(days=today.weekday() + 7)
        state.end_date = state.start_date + pd.Timedelta(days=6)
        state.previous_week = True
        state.year_to_date = False
    # Use session state to set the default values for the date inputs
    start_date = st.sidebar.date_input(
        "Start Date",
        value=state.get("start_date", start_date_val).date() if isinstance(state.get("start_date"),
                                                                           datetime) else state.get("start_date",
                                                                                                    start_date_val),
        key="start_date",
        max_value=datetime.today().date(),
        format="DD-MM-YYYY",
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=state.get("end_date", end_date_val).date() if isinstance(state.get("end_date"), datetime) else state.get(
            "end_date", end_date_val),
        key="end_date",
        min_value=start_date,
        max_value=datetime.today().date(),
        format="DD-MM-YYYY",
    )

    title_placeholder.title(f"From: {start_date.strftime('%d-%m-%Y')} to {end_date.strftime('%d-%m-%Y')}")

    filtered_medical_data = filter_by_date_range(
        clean_medical_data(
            apply_filters(
                cleaned_data,
                state_filter,
                city_filter,
                pincode_filter,
                speciality_filter,
                client_filter,
                project_filter
            )
        ),
        start_date,
        end_date
    )

    if filtered_medical_data.empty:
        st.warning("No data available.")
        return

    # Visualization Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "🏷️ Manufacturer Analysis",
        "📂 Data Types within Rx",
        "📍 Geographical Distribution",
        "📊 Demographic Distribution",
        "💊 Medicines",
        "🏭 Pharma Analytics",
        "🩺 Observations",
        "🧪 Diagnostics",
        "🔍 Manufacturer Comparison",
        "💰 Value-Based Comparison",
        "🏭 Market Share by primary use",
        "🩸 Vitals"
    ])
    display_sidebar_totals(filtered_medical_data)

    # Visualizations for each tab
    visualize_manufacturer_medicines(tab1, filtered_medical_data)
    visualize_data_types(tab2, filtered_medical_data)
    visualize_geographical_distribution(tab3, filtered_medical_data)
    visualize_patient_demographics(tab4, filtered_medical_data)
    visualize_medicines(tab5, filtered_medical_data)
    visualize_pharma_analytics(tab6, filtered_medical_data)
    visualize_observations(tab7, filtered_medical_data)
    visualize_diagnostics(tab8, filtered_medical_data)
    manufacturer_comparison_tab(tab9, filtered_medical_data)
    visualize_value_comparison(tab10, filtered_medical_data)
    visualize_market_share_primary_use(tab11, filtered_medical_data)
    visualize_vitals(tab12, filtered_medical_data)


if __name__ == '__main__':
    main()

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import smtplib
from datetime import datetime

CALL_LOG_PATH = "combined_calls.csv"
SMS_LOG_PATH = "combined_sms.csv"

# ----- Custom CSS for dark theme including sidebar -----
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
            color: #333333;
        }
        .stApp {
            background-color: #ffffff !important;
        }
        [data-testid="stSidebar"] {
            background-color: #e6f2ff !important;
            color: #000000 !important;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMetric {
            color: #000000 !important;
        }
        [data-testid="stSidebar"] .stMetric > div:nth-child(2) {
            background: linear-gradient(90deg, #0044cc, #00ccff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }
        h1, h2, h3, .stTextInput label, .stDateInput label, .stSlider label {
            color: #333333 !important;
        }
        .stDownloadButton button, .stButton>button {
            background-color: #008cff !important;
            color: white !important;
            border-radius: 8px;
            border: 1px solid #008cff;
            padding: 8px 20px;
            font-weight: bold;
        }
        .stDownloadButton button:hover, .stButton>button:hover {
            background-color: #0077e6 !important;
            color: #ffffff !important;
            border: 1px solid #005ec2;
        }
    </style>
""", unsafe_allow_html=True)


# ----- Data Load -----
@st.cache_data

def load_data():
    calls = pd.read_csv(CALL_LOG_PATH)
    sms = pd.read_csv(SMS_LOG_PATH)

    calls.rename(columns={'initiated_ts': 'timestamp'}, inplace=True)
    calls['timestamp'] = pd.to_datetime(calls['timestamp'])
    calls['date'] = calls['timestamp'].dt.date
    calls['week'] = calls['timestamp'].dt.isocalendar().week
    calls['month'] = calls['timestamp'].dt.to_period('M').astype(str)
    calls['hour'] = calls['timestamp'].dt.hour
    calls['minute'] = calls['timestamp'].dt.strftime('%H:%M')
    calls['duration'] = (pd.to_datetime(calls['ended_ts']) - pd.to_datetime(calls['answered_ts'])).dt.total_seconds()

    sms['timestamp'] = pd.to_datetime(sms['timestamp'])
    sms['date'] = sms['timestamp'].dt.date
    sms['hour'] = sms['timestamp'].dt.hour

    return calls, sms

calls_df, sms_df = load_data()

# ----- Sidebar Stats -----
st.sidebar.markdown("## ⚰️ Summary Stats")
st.sidebar.metric("Total Calls", len(calls_df))
st.sidebar.metric("Total SMS", len(sms_df))
st.sidebar.metric("Avg Duration (sec)", round(calls_df['duration'].mean(), 2))
st.sidebar.metric("Missed Calls", len(calls_df[calls_df['outcome'] == 'missed']))

# ----- Page Title -----
st.markdown("<h1 style='text-align: center; color: #ff00ff;'>Call & SMS Analytics Dashboard</h1>", unsafe_allow_html=True)

# ----- Date Filter -----
min_date = min(calls_df['timestamp'].min(), sms_df['timestamp'].min())
max_date = max(calls_df['timestamp'].max(), sms_df['timestamp'].max())
date_range = st.date_input("Filter by Date", [min_date.date(), max_date.date()])

calls_filtered = calls_df[(calls_df['timestamp'].dt.date >= date_range[0]) & (calls_df['timestamp'].dt.date <= date_range[1])]
sms_filtered = sms_df[(sms_df['timestamp'].dt.date >= date_range[0]) & (sms_df['timestamp'].dt.date <= date_range[1])]

# ----- Call Type Filter -----
call_types = st.multiselect("Filter by Call Type", options=calls_filtered['direction'].unique(), default=list(calls_filtered['direction'].unique()))
calls_filtered = calls_filtered[calls_filtered['direction'].isin(call_types)]

# ----- Plot Theme -----
px.defaults.template = "plotly_dark"
layout_kwargs = dict(
    plot_bgcolor='#ffffff',
    paper_bgcolor='#ffffff',
    font_color='#333333',
    title_font_color='#0044cc',
    legend_title_font_color='#0044cc',
    legend=dict(
        font=dict(color='#333333'),
        bgcolor='#ffffff',
        bordercolor='#cccccc',
        borderwidth=1
    )
)


# ----- Daily/Weekly/Monthly Call Volume -----
st.subheader("Call Volume Over Time (Daily)")
daily_outcomes = calls_filtered.groupby(['date', 'outcome']).size().reset_index(name='count')
fig_daily = px.bar(daily_outcomes, x='date', y='count', color='outcome', barmode='group')
fig_daily.update_layout(title="Daily Call Volume by Outcome", **layout_kwargs)
st.plotly_chart(fig_daily, use_container_width=True)

st.subheader("Call Volume Over Time (Weekly)")
weekly_outcomes = calls_filtered.groupby(['week', 'outcome']).size().reset_index(name='count')
fig_weekly = px.bar(weekly_outcomes, x='week', y='count', color='outcome', barmode='group')
fig_weekly.update_layout(title="Weekly Call Volume by Outcome", **layout_kwargs)
st.plotly_chart(fig_weekly, use_container_width=True)

st.subheader("Call Volume Over Time (Monthly)")
monthly_outcomes = calls_filtered.groupby(['month', 'outcome']).size().reset_index(name='count')
fig_monthly = px.bar(monthly_outcomes, x='month', y='count', color='outcome', barmode='group')
fig_monthly.update_layout(title="Monthly Call Volume by Outcome", **layout_kwargs)
st.plotly_chart(fig_monthly, use_container_width=True)

# ----- Rolling Avg -----
st.subheader("Rolling 7-Day Avg of Total Calls")
daily_total = calls_filtered.groupby('date').size().reset_index(name='total_calls')
daily_total['rolling_avg'] = daily_total['total_calls'].rolling(window=7).mean()
fig_roll = px.line(daily_total, x='date', y=['total_calls', 'rolling_avg'], labels={'value': 'Call Count', 'variable': 'Type'}, color_discrete_map={"total_calls": "orange", "rolling_avg": "green"})
fig_roll.update_traces(mode='lines+markers', marker=dict(color='white', size=8, line=dict(width=1, color='lightblue')))
fig_roll.update_layout(title="Total Calls vs Rolling 7-Day Average", **layout_kwargs)
st.plotly_chart(fig_roll, use_container_width=True)

# ----- Duration Trends -----
st.subheader("Call Duration Trends")
duration_data = calls_filtered.dropna(subset=['duration'])
fig2 = px.scatter(duration_data, x='timestamp', y='duration', color='outcome', color_discrete_sequence=px.colors.sequential.Purples)
fig2.update_traces(marker=dict(size=7, color=np.linspace(0, 1, len(duration_data)), colorscale='Viridis', line=dict(width=1, color='white')))
fig2.update_layout(title="Call Duration (sec)", **layout_kwargs)
st.plotly_chart(fig2, use_container_width=True)

# ----- Avg/Median Duration per Day -----
st.subheader("Average & Median Call Duration per Day")
duration_stats = duration_data.groupby('date')['duration'].agg(['mean', 'median']).reset_index()
fig_stats = px.line(duration_stats, x='date', y=['mean', 'median'], labels={'value': 'Duration (sec)', 'variable': 'Metric'}, color_discrete_map={"mean": "cyan", "median": "magenta"})
fig_stats.update_traces(mode='lines+markers', marker=dict(color='white', size=7, line=dict(width=1, color='magenta')))
fig_stats.update_layout(title="Avg & Median Call Duration", **layout_kwargs)
st.plotly_chart(fig_stats, use_container_width=True)

# ----- Duration vs Time of Day -----
st.subheader("Call Duration vs Time of Day")
duration_filtered = calls_filtered.dropna(subset=['duration'])
fig_scatter = px.scatter(duration_filtered, x='hour', y='duration', color='direction', labels={'hour': 'Hour of Day', 'duration': 'Call Duration (sec)'}, color_discrete_sequence=px.colors.sequential.Viridis)
fig_scatter.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
fig_scatter.update_layout(title="Call Duration by Time of Day (Incoming & Outgoing)", **layout_kwargs)
st.plotly_chart(fig_scatter, use_container_width=True)

# ----- Load Intensity Heatmap -----
st.subheader("Load Intensity Heatmap (Calls per Hour)")
heatmap_data = calls_filtered.groupby(['date', 'hour']).size().unstack(fill_value=0)
st.dataframe(heatmap_data.style.background_gradient(cmap='plasma'), use_container_width=True)

# ----- Load Intensity Scatter (Minute-Level) -----
st.subheader("Call Load Intensity (Minute-Level)")
minute_data = calls_filtered.groupby('minute').size().reset_index(name='call_count')
fig_minute = px.scatter(minute_data, x='minute', y='call_count', labels={'minute': 'Time of Day', 'call_count': 'Number of Calls'}, color_discrete_sequence=['#00ffff'])
fig_minute.update_layout(title="Number of Calls per Minute", **layout_kwargs)
st.plotly_chart(fig_minute, use_container_width=True)

# ----- Call Distribution by Hour -----
st.subheader("Call Distribution by Hour of Day")
fig_hour = px.histogram(calls_filtered, x='hour', nbins=24, color_discrete_sequence=['#ff00ff'])
fig_hour.update_layout(title="Calls by Hour", **layout_kwargs)
st.plotly_chart(fig_hour, use_container_width=True)

# ----- SMS Sent vs Received (Daily) -----
st.subheader("SMS Sent vs Received (Daily Trend)")
sms_trend = sms_filtered.groupby(['date', 'direction']).size().reset_index(name='count')
fig_sms = px.line(sms_trend, x='date', y='count', color='direction', color_discrete_sequence=px.colors.sequential.Inferno)
fig_sms.update_layout(title="SMS Sent vs Received", **layout_kwargs)
st.plotly_chart(fig_sms, use_container_width=True)

# ----- Top Contacts -----
if 'contact' in calls_filtered.columns:
    st.subheader("Top 10 Contacts by Volume")
    top_contacts = calls_filtered['contact'].value_counts().nlargest(10).reset_index()
    top_contacts.columns = ['contact', 'call_count']
    fig_contacts = px.bar(top_contacts, x='contact', y='call_count', color='call_count', color_continuous_scale='sunset')
    fig_contacts.update_layout(title="Top 10 Contacts by Volume", **layout_kwargs)
    st.plotly_chart(fig_contacts, use_container_width=True)

# ----- Alerts Config -----
st.subheader("🔔 Alert System")
missed_threshold_hour = st.slider("Alert if missed calls exceed N in 1 hour", 1, 10, 3)
low_call_threshold_day = st.slider("Alert if total calls fall below X in 1 day", 1, 100, 10)
missed_calls = calls_filtered[calls_filtered['outcome'] == 'missed']
missed_hourly = missed_calls.groupby([missed_calls['timestamp'].dt.date, missed_calls['timestamp'].dt.hour]).size()
alert_hours = missed_hourly[missed_hourly > missed_threshold_hour]
calls_per_day = calls_filtered.groupby(calls_filtered['timestamp'].dt.date).size()
low_call_days = calls_per_day[calls_per_day < low_call_threshold_day]
if not alert_hours.empty:
    st.error(f"🚨 High missed calls detected on: {', '.join(str(x) for x in alert_hours.index)}")
if not low_call_days.empty:
    st.warning(f"⚠️ Low call volume on: {', '.join(str(x) for x in low_call_days.index)}")
if alert_hours.empty and low_call_days.empty:
    st.success("✅ All thresholds normal.")

# ----- Outcome Pie Chart -----
st.subheader("Call Outcome Distribution")
outcome_dist = calls_filtered['outcome'].value_counts().reset_index()
outcome_dist.columns = ['outcome', 'count']
fig_pie = px.pie(outcome_dist, names='outcome', values='count', color_discrete_sequence=px.colors.sequential.RdPu)
fig_pie.update_layout(title="Call Outcome Breakdown", **layout_kwargs)
st.plotly_chart(fig_pie, use_container_width=True)

# ----- Export Buttons -----
st.download_button("Download Filtered Call CSV", data=calls_filtered.to_csv(index=False), file_name="filtered_calls.csv")
st.download_button("Download Filtered SMS CSV", data=sms_filtered.to_csv(index=False), file_name="filtered_sms.csv")

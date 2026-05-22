import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FitMetrics Pro | Gym Analytics",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  — dark athletic theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@700&display=swap');

/* ── Root palette ── */
:root {
    --bg:       #0a0c10;
    --surface:  #12161e;
    --card:     #181d28;
    --border:   #252d3d;
    --accent1:  #00f5a0;   /* neon mint */
    --accent2:  #ff4d6d;   /* vivid coral */
    --accent3:  #ffd60a;   /* electric yellow */
    --text:     #e8edf5;
    --muted:    #6b7897;
    --grad:     linear-gradient(135deg,#00f5a0 0%,#00d9f5 100%);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem !important; max-width: 1400px; }

/* ── HERO BANNER ── */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #0f1b2d 50%, #0d1117 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -50%; left: -10%;
    width: 60%; height: 200%;
    background: radial-gradient(ellipse, rgba(0,245,160,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute; bottom: -40%; right: -5%;
    width: 50%; height: 150%;
    background: radial-gradient(ellipse, rgba(255,77,109,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    letter-spacing: 0.12em;
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin: 0;
}
.hero-sub {
    font-size: 1rem;
    color: var(--muted);
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,245,160,0.12);
    border: 1px solid rgba(0,245,160,0.35);
    color: var(--accent1);
    border-radius: 50px;
    padding: 4px 14px;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ── METRIC CARDS ── */
.metric-grid { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1;
    min-width: 160px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.metric-card:hover { transform: translateY(-3px); border-color: var(--accent1); }
.metric-card::after {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--grad);
    border-radius: 16px 16px 0 0;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: var(--accent1);
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.35rem;
}

/* ── SECTION HEADERS ── */
.section-header {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.8rem;
    letter-spacing: 0.1em;
    color: var(--text);
    border-left: 4px solid var(--accent1);
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── PREDICTION BOX ── */
.pred-box {
    background: linear-gradient(135deg, rgba(0,245,160,0.08), rgba(0,217,245,0.05));
    border: 1px solid rgba(0,245,160,0.3);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
}
.pred-number {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 5rem;
    background: var(--grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
}
.pred-label {
    font-size: 0.85rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.pred-note {
    margin-top: 0.75rem;
    font-size: 0.82rem;
    color: var(--muted);
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stNumberInput label {
    color: var(--muted) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: 0.08em;
}

/* ── BUTTONS ── */
.stButton > button {
    background: var(--grad) !important;
    color: #0a0c10 !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.12em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 2.5rem !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,245,160,0.3) !important;
}

/* ── TAB STYLING ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--card) !important;
    color: var(--accent1) !important;
    border: 1px solid var(--border) !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── DIVIDER ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent1), transparent);
    margin: 1.5rem 0;
    opacity: 0.4;
}

/* ── INSIGHT PILL ── */
.insight-pill {
    display: inline-block;
    background: rgba(255,214,10,0.1);
    border: 1px solid rgba(255,214,10,0.3);
    color: var(--accent3);
    border-radius: 50px;
    padding: 3px 12px;
    font-size: 0.75rem;
    margin-right: 6px;
    margin-bottom: 4px;
}

/* ── Plotly chart background override ── */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("gym_members_exercise_tracking.csv")
    return df

@st.cache_resource
def load_model():
    try:
        return joblib.load("linear_model.pkl")
    except Exception:
        return None

df = load_data()
model = load_model()

FEATURE_COLS = [
    'Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
    'Resting_BPM', 'Session_Duration (hours)', 'Fat_Percentage',
    'Water_Intake (liters)', 'Workout_Frequency (days/week)',
    'Experience_Level', 'BMI', 'Gender_Male',
    'Workout_Type_Cardio', 'Workout_Type_HIIT',
    'Workout_Type_Strength', 'Workout_Type_Yoga'
]

PLOTLY_THEME = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e8edf5', family='DM Sans'),
    xaxis=dict(gridcolor='#252d3d', linecolor='#252d3d'),
    yaxis=dict(gridcolor='#252d3d', linecolor='#252d3d'),
    legend=dict(bgcolor='rgba(24,29,40,0.8)', bordercolor='#252d3d', borderwidth=1),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ─────────────────────────────────────────────
#  SIDEBAR — PREDICTION FORM
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔥 PREDICT\nCalories Burned")
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

    age = st.slider("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    weight = st.slider("Weight (kg)", 40, 130, 70)
    height = st.slider("Height (m)", 1.50, 2.00, 1.70, step=0.01)
    bmi = round(weight / (height ** 2), 2)
    st.markdown(f'<div style="font-size:0.8rem;color:#6b7897;margin-top:-10px;margin-bottom:10px;">BMI: <b style="color:#00f5a0">{bmi}</b></div>', unsafe_allow_html=True)

    workout_type = st.selectbox("Workout Type", ["Cardio", "HIIT", "Strength", "Yoga"])
    session_dur = st.slider("Session Duration (hrs)", 0.5, 3.0, 1.0, step=0.1)
    freq = st.slider("Workout Frequency (days/wk)", 1, 7, 3)
    exp_map = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}
    exp_label = st.selectbox("Experience Level", list(exp_map.values()))
    exp_level = {v: k for k, v in exp_map.items()}[exp_label]

    st.markdown("### VITALS")
    max_bpm = st.slider("Max BPM", 120, 220, 170)
    avg_bpm = st.slider("Avg BPM", 100, 180, 140)
    rest_bpm = st.slider("Resting BPM", 40, 90, 60)
    fat_pct = st.slider("Fat %", 5.0, 40.0, 20.0, step=0.5)
    water = st.slider("Water Intake (L)", 1.0, 4.0, 2.5, step=0.1)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    predict_clicked = st.button("⚡ PREDICT CALORIES", use_container_width=True)

    # Build feature vector
    features = {
        'Age': age, 'Weight (kg)': weight, 'Height (m)': height,
        'Max_BPM': max_bpm, 'Avg_BPM': avg_bpm, 'Resting_BPM': rest_bpm,
        'Session_Duration (hours)': session_dur, 'Fat_Percentage': fat_pct,
        'Water_Intake (liters)': water, 'Workout_Frequency (days/week)': freq,
        'Experience_Level': exp_level, 'BMI': bmi,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Workout_Type_Cardio': 1 if workout_type == "Cardio" else 0,
        'Workout_Type_HIIT': 1 if workout_type == "HIIT" else 0,
        'Workout_Type_Strength': 1 if workout_type == "Strength" else 0,
        'Workout_Type_Yoga': 1 if workout_type == "Yoga" else 0,
    }
    fvec = pd.DataFrame([features])[FEATURE_COLS]
    prediction = None
    if model is not None:
        prediction = float(model.predict(fvec)[0])


# ─────────────────────────────────────────────
#  HERO SECTION
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🏋️ Machine Learning · Fitness Analytics</div>
  <h1 class="hero-title">FITMETRICS PRO</h1>
  <p class="hero-sub">Gym Member Intelligence Dashboard — 973 Athletes Analyzed</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TOP METRICS
# ─────────────────────────────────────────────
total = len(df)
avg_cal = int(df['Calories_Burned'].mean())
avg_dur = round(df['Session_Duration (hours)'].mean(), 2)
avg_bmi_val = round(df['BMI'].mean(), 1)
top_workout = df['Workout_Type'].value_counts().idxmax()

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-val">{total}</div>
    <div class="metric-label">Total Athletes</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">{avg_cal}</div>
    <div class="metric-label">Avg Calories Burned</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">{avg_dur}h</div>
    <div class="metric-label">Avg Session Length</div>
  </div>
  <div class="metric-card">
    <div class="metric-val">{avg_bmi_val}</div>
    <div class="metric-label">Avg BMI</div>
  </div>
  <div class="metric-card">
    <div class="metric-val" style="font-size:1.3rem;padding-top:0.3rem">{top_workout.upper()}</div>
    <div class="metric-label">Top Workout Type</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PREDICTION RESULT  (shown when triggered)
# ─────────────────────────────────────────────
if predict_clicked and prediction is not None:
    intensity = "🟢 Light" if prediction < 600 else ("🟡 Moderate" if prediction < 900 else "🔴 Intense")
    pct_above_avg = round((prediction - avg_cal) / avg_cal * 100, 1)
    sign = "+" if pct_above_avg >= 0 else ""

    st.markdown(f"""
    <div class="pred-box">
      <div class="pred-label">Predicted Calories Burned</div>
      <div class="pred-number">{int(prediction):,}</div>
      <div class="pred-label">kcal per session</div>
      <div class="pred-note">
        Intensity: <b style="color:#ffd60a">{intensity}</b> &nbsp;|&nbsp;
        vs. dataset avg: <b style="color:{'#00f5a0' if pct_above_avg>=0 else '#ff4d6d'}">{sign}{pct_above_avg}%</b>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview", "🏃  Workout Analysis", "🫀  Health Metrics", "🗃️  Raw Data"
])


# ═══════════════════════════════════════════
#  TAB 1 — OVERVIEW
# ═══════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">CALORIE DISTRIBUTION</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])

    with col1:
        fig_hist = px.histogram(
            df, x='Calories_Burned', nbins=40,
            color_discrete_sequence=['#00f5a0'],
            labels={'Calories_Burned': 'Calories Burned (kcal)'}
        )
        fig_hist.update_traces(marker_line_color='#0a0c10', marker_line_width=0.5)
        fig_hist.update_layout(**PLOTLY_THEME, title="Calorie Burn Distribution")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        wt_counts = df['Workout_Type'].value_counts().reset_index()
        fig_pie = px.pie(
            wt_counts, names='Workout_Type', values='count',
            color_discrete_sequence=['#00f5a0', '#ff4d6d', '#ffd60a', '#00d9f5'],
            hole=0.55
        )
        fig_pie.update_layout(**PLOTLY_THEME, title="Workout Distribution",
                              showlegend=True)
        fig_pie.update_traces(textfont_color='#e8edf5')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Calories by workout × gender
    st.markdown('<div class="section-header">CALORIES BY WORKOUT & GENDER</div>', unsafe_allow_html=True)
    fig_box = px.box(
        df, x='Workout_Type', y='Calories_Burned', color='Gender',
        color_discrete_map={'Male': '#00f5a0', 'Female': '#ff4d6d'},
        labels={'Calories_Burned': 'Calories Burned', 'Workout_Type': ''},
    )
    fig_box.update_layout(**PLOTLY_THEME, title="Box Plot — Calories Burned by Workout Type & Gender")
    st.plotly_chart(fig_box, use_container_width=True)


# ═══════════════════════════════════════════
#  TAB 2 — WORKOUT ANALYSIS
# ═══════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">WORKOUT DEEP DIVE</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        avg_by_workout = df.groupby('Workout_Type')['Calories_Burned'].mean().reset_index()
        fig_bar = px.bar(
            avg_by_workout, x='Workout_Type', y='Calories_Burned',
            color='Workout_Type',
            color_discrete_map={'HIIT':'#ff4d6d','Cardio':'#00f5a0','Strength':'#ffd60a','Yoga':'#00d9f5'},
            labels={'Calories_Burned': 'Avg Calories', 'Workout_Type': ''},
        )
        fig_bar.update_layout(**PLOTLY_THEME, title="Average Calories by Workout Type", showlegend=False)
        fig_bar.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        freq_cal = df.groupby('Workout_Frequency (days/week)')['Calories_Burned'].mean().reset_index()
        fig_line = px.line(
            freq_cal, x='Workout_Frequency (days/week)', y='Calories_Burned',
            markers=True, color_discrete_sequence=['#ffd60a'],
            labels={'Calories_Burned': 'Avg Calories Burned'},
        )
        fig_line.update_traces(line_width=2.5, marker_size=9)
        fig_line.update_layout(**PLOTLY_THEME, title="Avg Calories by Workout Frequency")
        st.plotly_chart(fig_line, use_container_width=True)

    # Session duration vs Calories scatter
    st.markdown('<div class="section-header">SESSION DURATION vs CALORIES</div>', unsafe_allow_html=True)
    fig_scatter = px.scatter(
        df, x='Session_Duration (hours)', y='Calories_Burned',
        color='Workout_Type', size='BMI',
        color_discrete_map={'HIIT':'#ff4d6d','Cardio':'#00f5a0','Strength':'#ffd60a','Yoga':'#00d9f5'},
        opacity=0.75, size_max=12,
        labels={'Session_Duration (hours)': 'Session Duration (hrs)', 'Calories_Burned': 'Calories Burned'},
        hover_data=['Age', 'Gender', 'Experience_Level'],
    )
    fig_scatter.update_layout(**PLOTLY_THEME, title="Session Duration vs Calories Burned (bubble size = BMI)")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Experience level breakdown
    st.markdown('<div class="section-header">EXPERIENCE LEVEL BREAKDOWN</div>', unsafe_allow_html=True)
    exp_labels = {1: 'Beginner', 2: 'Intermediate', 3: 'Advanced'}
    df['Experience'] = df['Experience_Level'].map(exp_labels)
    avg_exp = df.groupby(['Experience', 'Workout_Type'])['Calories_Burned'].mean().reset_index()
    fig_grp = px.bar(
        avg_exp, x='Experience', y='Calories_Burned', color='Workout_Type',
        barmode='group',
        color_discrete_map={'HIIT':'#ff4d6d','Cardio':'#00f5a0','Strength':'#ffd60a','Yoga':'#00d9f5'},
        labels={'Calories_Burned': 'Avg Calories'},
        category_orders={'Experience': ['Beginner', 'Intermediate', 'Advanced']},
    )
    fig_grp.update_layout(**PLOTLY_THEME, title="Avg Calories by Experience Level & Workout Type")
    st.plotly_chart(fig_grp, use_container_width=True)


# ═══════════════════════════════════════════
#  TAB 3 — HEALTH METRICS
# ═══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">HEALTH METRICS EXPLORER</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig_bmi = px.histogram(
            df, x='BMI', nbins=35, color='Gender',
            color_discrete_map={'Male': '#00f5a0', 'Female': '#ff4d6d'},
            barmode='overlay', opacity=0.75,
        )
        fig_bmi.update_layout(**PLOTLY_THEME, title="BMI Distribution by Gender")
        st.plotly_chart(fig_bmi, use_container_width=True)

    with col2:
        fig_fat = px.box(
            df, x='Gender', y='Fat_Percentage', color='Gender',
            color_discrete_map={'Male': '#00f5a0', 'Female': '#ff4d6d'},
            points='all',
        )
        fig_fat.update_traces(marker_size=3, opacity=0.5)
        fig_fat.update_layout(**PLOTLY_THEME, title="Fat % Distribution by Gender", showlegend=False)
        st.plotly_chart(fig_fat, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-header">FEATURE CORRELATION</div>', unsafe_allow_html=True)
    numeric_cols = ['Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
                    'Resting_BPM', 'Session_Duration (hours)', 'Calories_Burned',
                    'Fat_Percentage', 'BMI', 'Workout_Frequency (days/week)']
    corr = df[numeric_cols].corr()
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr.values.round(2), x=corr.columns, y=corr.index,
        text=corr.values.round(2), texttemplate='%{text}',
        colorscale=[[0,'#ff4d6d'],[0.5,'#181d28'],[1,'#00f5a0']],
        zmid=0, showscale=True,
        colorbar=dict(tickfont=dict(color='#6b7897'))
    ))
    fig_heat.update_layout(**PLOTLY_THEME, title="Pearson Correlation Matrix",
                           height=480, xaxis_tickangle=-40)
    st.plotly_chart(fig_heat, use_container_width=True)

    # BPM analysis
    st.markdown('<div class="section-header">BPM PROFILE BY WORKOUT</div>', unsafe_allow_html=True)
    bpm_data = df.groupby('Workout_Type')[['Max_BPM','Avg_BPM','Resting_BPM']].mean().reset_index()
    fig_bpm = go.Figure()
    for col_name, color in [('Max_BPM','#ff4d6d'), ('Avg_BPM','#ffd60a'), ('Resting_BPM','#00d9f5')]:
        fig_bpm.add_trace(go.Bar(
            name=col_name.replace('_', ' '),
            x=bpm_data['Workout_Type'], y=bpm_data[col_name],
            marker_color=color, marker_line_width=0,
        ))
    fig_bpm.update_layout(**PLOTLY_THEME, barmode='group', title="Avg BPM Metrics by Workout Type")
    st.plotly_chart(fig_bpm, use_container_width=True)


# ═══════════════════════════════════════════
#  TAB 4 — RAW DATA
# ═══════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">DATASET EXPLORER</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        gender_filter = st.multiselect("Filter by Gender", options=df['Gender'].unique().tolist(), default=df['Gender'].unique().tolist())
    with col2:
        workout_filter = st.multiselect("Filter by Workout", options=df['Workout_Type'].unique().tolist(), default=df['Workout_Type'].unique().tolist())
    with col3:
        exp_opts = [f"Level {i}" for i in sorted(df['Experience_Level'].unique())]
        exp_filter_display = st.multiselect("Filter by Experience", options=exp_opts, default=exp_opts)
        exp_filter = [int(x.split()[-1]) for x in exp_filter_display]

    filtered_df = df[
        df['Gender'].isin(gender_filter) &
        df['Workout_Type'].isin(workout_filter) &
        df['Experience_Level'].isin(exp_filter)
    ].reset_index(drop=True)

    st.markdown(f'<div class="insight-pill">📋 Showing {len(filtered_df)} of {len(df)} records</div>', unsafe_allow_html=True)
    st.dataframe(
        filtered_df.drop(columns=['Experience'], errors='ignore'),
        use_container_width=True,
        height=420
    )

    st.markdown('<div class="section-header">SUMMARY STATISTICS</div>', unsafe_allow_html=True)
    st.dataframe(
        filtered_df.drop(columns=['Experience'], errors='ignore').describe().round(2),
        use_container_width=True
    )


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#6b7897;font-size:0.78rem;padding-bottom:1rem;">
  Built with <b style="color:#00f5a0">Streamlit</b> · ML Model: <b style="color:#00f5a0">Scikit-learn Linear Regression</b> · Dataset: 973 gym members
</div>
""", unsafe_allow_html=True)
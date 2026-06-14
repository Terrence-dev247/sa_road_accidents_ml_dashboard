import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SA Road Accidents · ML Dashboard",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f8f9fb; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 4px solid #1A3A5C;
    }
    .metric-label { font-size: 12px; color: #888; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1A3A5C; margin-top: 2px; }
    .metric-sub   { font-size: 12px; color: #aaa; margin-top: 2px; }
    .section-header {
        font-size: 18px; font-weight: 700; color: #1A3A5C;
        border-bottom: 2px solid #1A3A5C; padding-bottom: 6px; margin-bottom: 1rem;
    }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 12px; font-weight: 600;
    }
    .badge-fatal   { background:#fde8e8; color:#c0392b; }
    .badge-headon  { background:#fef3cd; color:#856404; }
    .badge-bumper  { background:#d4edda; color:#155724; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 500; }
    div[data-testid="stSidebarContent"] { background: #1A3A5C; }
    div[data-testid="stSidebarContent"] * { color: white !important; }
    div[data-testid="stSidebarContent"] .stSelectbox label,
    div[data-testid="stSidebarContent"] .stMultiSelect label { color: #cdd5e0 !important; }
</style>
""", unsafe_allow_html=True)

SEED = 42

# ── Data loading & preprocessing ─────────────────────────────────────────────
@st.cache_data
def load_and_process():
    df = pd.read_excel("data/South Africa Road Accidents Dataset - 2017.xlsx", engine="openpyxl")
    df_c = df.copy()

    # Feature engineering
    df_c["Hour"]  = pd.to_datetime(df_c["Time"], errors="coerce").dt.hour.fillna(12)
    df_c["Month"] = pd.to_datetime(df_c["Date"], errors="coerce").dt.month.fillna(1)
    df_c["SpeedLimit_num"] = df_c["Speed Zone"].str.extract(r"(\d+)").astype(float).fillna(60)

    # Encode target
    le_target = LabelEncoder()
    df_c["Severity_enc"] = le_target.fit_transform(df_c["Accident Severity"])
    severity_classes = list(le_target.classes_)

    # Encode features
    le = LabelEncoder()
    for col in ["Province", "Location", "Vehicle Type", "Occations", "Speed Zone", "City"]:
        df_c[col + "_enc"] = le.fit_transform(df_c[col].astype(str))

    feature_cols = [
        "Province_enc", "Location_enc", "Vehicle Type_enc", "Occations_enc",
        "Speed Zone_enc", "City_enc", "Number of Vehicles", "Number of Casualties",
        "SpeedLimit_num", "Hour", "Month", "Police Force"
    ]

    X = df_c[feature_cols]
    y = df_c["Severity_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    return df, df_c, X, y, X_train_sc, X_test_sc, y_train, y_test, scaler, feature_cols, severity_classes

@st.cache_resource
def train_models(X_train_sc, y_train, X_test_sc, y_test, severity_classes):
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=3,
                                 random_state=SEED, class_weight="balanced")
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                     max_depth=4, subsample=0.8, random_state=SEED)
    voting = VotingClassifier(estimators=[("rf", RandomForestClassifier(n_estimators=200, max_depth=6,
                                                                          min_samples_leaf=3, random_state=SEED,
                                                                          class_weight="balanced")),
                                           ("gb", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                                               max_depth=4, subsample=0.8,
                                                                               random_state=SEED))],
                               voting="soft")
    results = {}
    for name, model in [("Random Forest", rf), ("Gradient Boosting", gb), ("Soft-Voting Ensemble", voting)]:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        cv = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="accuracy")
        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "Accuracy":  round(accuracy_score(y_test, y_pred), 3),
            "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 3),
            "Recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 3),
            "F1-Score":  round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 3),
            "CV_Mean":   round(cv.mean(), 3),
            "CV_Std":    round(cv.std(), 3),
        }
    return results

@st.cache_data
def train_rl():
    """Q-Learning agent for safety interventions."""
    state_names = [
        "Good Road – Low Freq",   "Good Road – Med Freq",   "Good Road – High Freq",
        "Mod Road – Low Freq",    "Mod Road – Med Freq",    "Mod Road – High Freq",
        "Poor Road – Low Freq",   "Poor Road – Med Freq",   "Poor Road – High Freq",
    ]
    action_names = [
        "No Intervention",
        "Safety Awareness Campaign",
        "Increased Law Enforcement",
        "Speed Reduction + Enforcement",
    ]
    n_states, n_actions = 9, 4

    R = np.array([
        [ 0,  3,  5,  4], [ 0,  4,  6,  5], [ 0,  5,  7,  9],
        [ 0,  3,  5,  6], [ 0,  5,  7,  8], [ 0,  6,  8, 10],
        [ 0,  4,  7,  8], [ 0,  6,  9, 10], [ 0,  7, 10, 12],
    ], dtype=float)

    np.random.seed(SEED)
    P = np.random.dirichlet(np.ones(n_states), size=(n_states, n_actions))

    Q = np.zeros((n_states, n_actions))
    alpha, gamma = 0.1, 0.95
    epsilon, eps_min, eps_decay = 1.0, 0.05, 0.995
    episode_rewards, epsilon_history = [], []

    for _ in range(5000):
        state = np.random.randint(n_states)
        total_r = 0
        for _ in range(50):
            if np.random.rand() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q[state])
            next_state = np.random.choice(n_states, p=P[state, action])
            reward = R[state, action]
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
            total_r += reward
        epsilon = max(eps_min, epsilon * eps_decay)
        episode_rewards.append(total_r)
        epsilon_history.append(epsilon)

    optimal_policy = {state_names[s]: action_names[np.argmax(Q[s])] for s in range(n_states)}
    return Q, state_names, action_names, optimal_policy, episode_rewards, epsilon_history, R

# ── Load everything ───────────────────────────────────────────────────────────
df, df_c, X, y, X_train_sc, X_test_sc, y_train, y_test, scaler, feature_cols, severity_classes = load_and_process()

with st.spinner("Training models…"):
    results = train_models(X_train_sc, y_train, X_test_sc, y_test, severity_classes)

Q, state_names, action_names, optimal_policy, episode_rewards, epsilon_history, R = train_rl()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛣️ SA Road Accidents")
    st.markdown("**ML Dashboard**")
    st.markdown("---")
    st.markdown("### Filters")
    province_filter = st.multiselect("Province", options=sorted(df["Province"].unique()),
                                      default=sorted(df["Province"].unique()))
    severity_filter = st.multiselect("Accident Severity", options=sorted(df["Accident Severity"].unique()),
                                      default=sorted(df["Accident Severity"].unique()))
    vehicle_filter  = st.multiselect("Vehicle Type", options=sorted(df["Vehicle Type"].unique()),
                                      default=sorted(df["Vehicle Type"].unique()))
    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Built by **Tebogo Poohe** as part of ML700 at Richfield Graduate Institute of Technology.\n\n"
        "Combines ensemble learning (RF + GB + Voting) with Q-learning for adaptive road safety interventions."
    )

# Apply filters
df_f = df[
    df["Province"].isin(province_filter) &
    df["Accident Severity"].isin(severity_filter) &
    df["Vehicle Type"].isin(vehicle_filter)
]

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("# 🛣️ South African Road Accidents — ML Safety Dashboard")
st.markdown("*Hybrid Reinforcement Learning & Ensemble Learning System · ML700 Project · Tebogo Poohe*")
st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
best_acc = results["Soft-Voting Ensemble"]["Accuracy"]
best_f1  = results["Soft-Voting Ensemble"]["F1-Score"]
fatal_pct = round(len(df_f[df_f["Accident Severity"]=="Fatal Accident"]) / max(len(df_f),1) * 100, 1)

with k1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Total Accidents</div>
        <div class="metric-value">{len(df_f)}</div>
        <div class="metric-sub">filtered records</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Fatal Rate</div>
        <div class="metric-value">{fatal_pct}%</div>
        <div class="metric-sub">of filtered accidents</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Best Model Accuracy</div>
        <div class="metric-value">{best_acc:.1%}</div>
        <div class="metric-sub">Soft-Voting Ensemble</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Best F1-Score</div>
        <div class="metric-value">{best_f1:.1%}</div>
        <div class="metric-sub">weighted average</div>
    </div>""", unsafe_allow_html=True)
with k5:
    total_cas = int(df_f["Number of Casualties"].sum())
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Total Casualties</div>
        <div class="metric-value">{total_cas}</div>
        <div class="metric-sub">across filtered data</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Exploratory Analysis",
    "🤖 Model Performance",
    "🗺️ RL Safety Agent",
    "🔮 Predict & Recommend",
    "📋 Raw Data"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — EDA
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Severity distribution
        sev_counts = df_f["Accident Severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        color_map = {"Fatal Accident": "#c0392b", "Headon Accident": "#e67e22", "Bumper Accident": "#27ae60"}
        fig = px.pie(sev_counts, names="Severity", values="Count",
                     title="Accident Severity Distribution",
                     color="Severity", color_discrete_map=color_map,
                     hole=0.4)
        fig.update_layout(height=340, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Accidents by province
        prov_counts = df_f["Province"].value_counts().reset_index()
        prov_counts.columns = ["Province", "Count"]
        fig = px.bar(prov_counts, x="Province", y="Count",
                     title="Accidents by Province",
                     color="Province", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=340, margin=dict(t=40,b=10,l=10,r=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Hour of day pattern
        df_c_f = df_c[df["Province"].isin(province_filter) & df["Accident Severity"].isin(severity_filter) & df["Vehicle Type"].isin(vehicle_filter)]
        hourly = df_c_f.groupby("Hour").size().reset_index(name="Count")
        fig = px.area(hourly, x="Hour", y="Count",
                      title="Accidents by Hour of Day",
                      color_discrete_sequence=["#1A3A5C"])
        fig.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Casualties by vehicle type
        veh_cas = df_f.groupby("Vehicle Type")["Number of Casualties"].sum().reset_index()
        veh_cas = veh_cas.sort_values("Number of Casualties", ascending=True)
        fig = px.bar(veh_cas, x="Number of Casualties", y="Vehicle Type",
                     orientation="h", title="Total Casualties by Vehicle Type",
                     color="Number of Casualties", color_continuous_scale="Reds")
        fig.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)

    with col5:
        # Severity by Province stacked bar
        prov_sev = df_f.groupby(["Province", "Accident Severity"]).size().reset_index(name="Count")
        fig = px.bar(prov_sev, x="Province", y="Count", color="Accident Severity",
                     title="Severity Breakdown by Province",
                     color_discrete_map=color_map, barmode="stack")
        fig.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        # Speed zone vs severity
        spd_sev = df_f.groupby(["Speed Zone", "Accident Severity"]).size().reset_index(name="Count")
        fig = px.bar(spd_sev, x="Speed Zone", y="Count", color="Accident Severity",
                     title="Severity by Speed Zone",
                     color_discrete_map=color_map, barmode="group")
        fig.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Ensemble Model Performance</div>', unsafe_allow_html=True)

    # Metric comparison
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    model_names = list(results.keys())

    fig = go.Figure()
    colors = ["#1A3A5C", "#2980b9", "#27ae60"]
    for i, name in enumerate(model_names):
        vals = [results[name][m] for m in metrics]
        fig.add_trace(go.Bar(name=name, x=metrics, y=vals, marker_color=colors[i],
                              text=[f"{v:.1%}" for v in vals], textposition="outside"))
    fig.update_layout(
        title="Model Comparison — Accuracy, Precision, Recall, F1",
        barmode="group", height=380,
        yaxis=dict(range=[0, 1.15], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=20, l=20, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    for col, name in zip([col1, col2, col3], model_names):
        with col:
            st.markdown(f"**{name}**")
            res = results[name]
            cm = confusion_matrix(y_test, res["y_pred"])
            fig = px.imshow(cm, text_auto=True,
                            x=severity_classes, y=severity_classes,
                            color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual"),
                            title=f"Confusion Matrix")
            fig.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10),
                               coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # Cross-validation
    st.markdown("---")
    st.markdown("**Cross-Validation Results (5-Fold)**")
    cv_data = []
    for name in model_names:
        cv_data.append({
            "Model": name,
            "CV Accuracy (mean)": f"{results[name]['CV_Mean']:.3f}",
            "CV Std (±)": f"{results[name]['CV_Std']:.3f}",
            "Test Accuracy": f"{results[name]['Accuracy']:.3f}",
            "Test F1": f"{results[name]['F1-Score']:.3f}",
        })
    st.dataframe(pd.DataFrame(cv_data), use_container_width=True, hide_index=True)

    # Feature importance
    st.markdown("---")
    st.markdown("**Random Forest — Feature Importance**")
    rf_model = results["Random Forest"]["model"]
    feat_imp = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=True)
    fig = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues",
                 title="Feature Importance (Mean Decrease in Impurity)")
    fig.update_layout(height=380, margin=dict(t=40,b=10,l=10,r=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — RL AGENT
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Q-Learning Safety Intervention Agent</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Q-Table heatmap
        Q_df = pd.DataFrame(Q, index=state_names, columns=action_names)
        fig = px.imshow(Q_df, text_auto=".2f",
                        color_continuous_scale="YlOrRd",
                        labels=dict(x="Action", y="MDP State", color="Q-Value"),
                        title="Learned Q-Values Heatmap")
        fig.update_layout(height=420, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Optimal policy table
        st.markdown("**Optimal Policy (Greedy)**")
        policy_rows = []
        for s in range(9):
            best_a = np.argmax(Q[s])
            policy_rows.append({
                "MDP State": state_names[s],
                "Recommended Action": action_names[best_a],
                "Q-Value": round(Q[s, best_a], 2),
            })
        policy_df = pd.DataFrame(policy_rows)
        st.dataframe(policy_df, use_container_width=True, hide_index=True, height=390)

    # Convergence chart
    st.markdown("---")
    col3, col4 = st.columns(2)
    window = 200
    smoothed = pd.Series(episode_rewards).rolling(window).mean()

    with col3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=episode_rewards, mode="lines", name="Raw reward",
                                  line=dict(color="#aec6e8", width=1), opacity=0.5))
        fig.add_trace(go.Scatter(y=smoothed, mode="lines", name=f"Rolling avg ({window})",
                                  line=dict(color="#1A3A5C", width=2.5)))
        fig.update_layout(title="Q-Learning Convergence — Episode Rewards",
                           xaxis_title="Episode", yaxis_title="Total Reward",
                           height=320, margin=dict(t=40,b=20,l=20,r=20),
                           legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=epsilon_history, mode="lines", name="ε (Epsilon)",
                                  line=dict(color="#e67e22", width=2.5)))
        fig.update_layout(title="Exploration Rate (ε) Decay",
                           xaxis_title="Episode", yaxis_title="Epsilon",
                           height=320, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig, use_container_width=True)

    st.info("**MDP Design:** 9 states (3 road conditions × 3 frequency levels) · 4 actions · 5,000 training episodes · ε-greedy with exponential decay from 1.0 → 0.05")

# ════════════════════════════════════════════════════════════
# TAB 4 — PREDICT & RECOMMEND
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Live Prediction & Intervention Recommender</div>', unsafe_allow_html=True)
    st.markdown("Enter accident details below to get a severity prediction and a recommended safety intervention.")

    col1, col2, col3 = st.columns(3)

    le_province = LabelEncoder().fit(df["Province"].astype(str))
    le_location = LabelEncoder().fit(df["Location"].astype(str))
    le_vehicle  = LabelEncoder().fit(df["Vehicle Type"].astype(str))
    le_occ      = LabelEncoder().fit(df["Occations"].astype(str))
    le_speed    = LabelEncoder().fit(df["Speed Zone"].astype(str))
    le_city     = LabelEncoder().fit(df["City"].astype(str))

    with col1:
        province   = st.selectbox("Province", sorted(df["Province"].unique()))
        location   = st.selectbox("Location", sorted(df["Location"].unique()))
        vehicle    = st.selectbox("Vehicle Type", sorted(df["Vehicle Type"].unique()))
        occasion   = st.selectbox("Occasion", sorted(df["Occations"].unique()))
    with col2:
        speed_zone = st.selectbox("Speed Zone", sorted(df["Speed Zone"].unique()))
        city       = st.selectbox("City", sorted(df["City"].unique()))
        n_vehicles = st.slider("Number of Vehicles", 1, 10, 3)
        n_casual   = st.slider("Number of Casualties", 0, 10, 1)
    with col3:
        hour        = st.slider("Hour of Day", 0, 23, 8)
        month       = st.slider("Month", 1, 12, 6)
        police      = st.slider("Police Force (Station ID)", 1, 5, 2)
        speed_limit = st.number_input("Speed Limit (km/h)", min_value=20, max_value=200, value=60, step=10)

    if st.button("🔮 Predict Severity & Recommend Intervention", type="primary", use_container_width=True):
        try:
            prov_enc = le_province.transform([province])[0]
        except:
            prov_enc = 0
        try:
            loc_enc = le_location.transform([location])[0]
        except:
            loc_enc = 0
        try:
            veh_enc = le_vehicle.transform([vehicle])[0]
        except:
            veh_enc = 0
        try:
            occ_enc = le_occ.transform([occasion])[0]
        except:
            occ_enc = 0
        try:
            spd_enc = le_speed.transform([speed_zone])[0]
        except:
            spd_enc = 0
        try:
            city_enc = le_city.transform([city])[0]
        except:
            city_enc = 0

        row = np.array([[prov_enc, loc_enc, veh_enc, occ_enc, spd_enc, city_enc,
                         n_vehicles, n_casual, float(speed_limit), float(hour),
                         float(month), float(police)]])
        row_sc = scaler.transform(row)

        best_model = results["Soft-Voting Ensemble"]["model"]
        pred_enc   = best_model.predict(row_sc)[0]
        pred_proba = best_model.predict_proba(row_sc)[0]
        pred_label = severity_classes[pred_enc]

        # Map to MDP state
        province_to_road = {"Gauteng": 0, "Mpumalanga": 1, "Free State": 2, "Limpopo": 2}
        road_cond = province_to_road.get(province, 1)
        sev_to_freq = {0: 0, 1: 1, 2: 2}  # bumper=low, fatal=high, headon=med
        freq_level = sev_to_freq.get(pred_enc, 1)
        mdp_state  = road_cond * 3 + freq_level
        best_action_idx = np.argmax(Q[mdp_state])
        recommended = action_names[best_action_idx]
        q_val = Q[mdp_state, best_action_idx]

        badge_class = {"Fatal Accident": "badge-fatal", "Headon Accident": "badge-headon", "Bumper Accident": "badge-bumper"}.get(pred_label, "badge-bumper")

        st.markdown("---")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Predicted Severity</div>
                <div class="metric-value"><span class="badge {badge_class}">{pred_label}</span></div>
                <div class="metric-sub">Soft-Voting Ensemble</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Recommended Intervention</div>
                <div class="metric-value" style="font-size:18px">{recommended}</div>
                <div class="metric-sub">Q-Value: {q_val:.2f} · MDP State: {state_names[mdp_state]}</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            confidence = pred_proba[pred_enc]
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value">{confidence:.1%}</div>
                <div class="metric-sub">predicted class probability</div>
            </div>""", unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        prob_df = pd.DataFrame({"Severity Class": severity_classes, "Probability": pred_proba})
        fig = px.bar(prob_df, x="Severity Class", y="Probability",
                     color="Severity Class", color_discrete_map=color_map,
                     title="Prediction Probability Breakdown",
                     text=[f"{p:.1%}" for p in pred_proba])
        fig.update_traces(textposition="outside")
        fig.update_layout(height=320, yaxis=dict(range=[0, 1.15], tickformat=".0%"),
                           showlegend=False, margin=dict(t=40,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 5 — RAW DATA
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown(f"Showing **{len(df_f)}** of {len(df)} records (apply sidebar filters to narrow down).")
    st.dataframe(df_f.reset_index(drop=True), use_container_width=True, height=500)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Dataset summary**")
        st.dataframe(df_f.describe(include="all").T, use_container_width=True)

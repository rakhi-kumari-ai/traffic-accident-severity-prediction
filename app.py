# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt
import plotly.graph_objects as go

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Traffic Accident Severity Prediction 🚦",
    layout="wide"
)

# ===============================
# DARK MODE + FULL STYLING
# ===============================
st.markdown("""
<style>

/* ---------- APP BACKGROUND ---------- */
[data-testid="stAppViewContainer"] {
    background-color: #121212;
    color: #ffffff;
}

/* ---------- HEADER ---------- */
.header {
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(90deg,#4e54c8,#8f94fb);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
.subheader {
    text-align: center;
    color: #bbbbbb;
    margin-bottom: 40px;
}

/* ---------- RESULT ---------- */
.result-box {
    padding: 28px;
    font-size: 32px;
    font-weight: bold;
    text-align: center;
    color: white;
    border-radius: 18px;
    background: linear-gradient(135deg,#1f2937,#111827);
    box-shadow: 0px 6px 20px rgba(0,0,0,0.6);
}

/* ---------- DIVIDER ---------- */
.section-line {
    height: 3px;
    width: 55%;
    margin: 30px auto;
    background: linear-gradient(to right,#4e54c8,#8f94fb);
    border-radius: 10px;
}

/* ---------- CARDS ---------- */
.card {
    padding: 20px;
    border-radius: 18px;
    background-color: #1e1e1e;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.5);
    text-align: center;
    color: white;
}

/* ---------- SIDEBAR (DOUBLE DARK GREEN) ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b3d0b, #14532d, #064e3b);
}

/* Sidebar labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: #ffffff !important;
}

/* Sidebar title box */
.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 15px;
}

/* ---------- DROPDOWN FIX ---------- */
div[data-baseweb="select"] span {
    color: #ffffff !important;
}
ul[role="listbox"] {
    background-color: #1e1e1e !important;
}
li[role="option"] {
    color: #ffffff !important;
}
li[role="option"]:hover {
    background-color: #14532d !important;
}

/* ---------- TABS ---------- */
button[data-baseweb="tab"] {
    color: #ffffff !important;
    font-weight: 600;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #10b981 !important;
    border-bottom: 3px solid #10b981;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* ---------- SELECTED VALUE IN DROPDOWN ---------- */
div[data-baseweb="select"] div[data-testid="stSelectbox"] span {
    color: #000000 !important;
}

/* ---------- ALL DROPDOWN OPTIONS ---------- */
ul[role="listbox"] li[role="option"] {
    color: #000000 !important;
}

/* Hover effect for options */
ul[role="listbox"] li[role="option"]:hover {
    background-color: #e5e7eb !important; /* light gray */
}

</style>
""", unsafe_allow_html=True)





# ===============================
# HEADER
# ===============================
st.markdown('<div class="header">🚦 Traffic Accident Severity Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">AI-powered road accident risk analysis</div>', unsafe_allow_html=True)

# ===============================
# LOAD MODEL (UNCHANGED)
# ===============================
saved = joblib.load("fatal_accident_model.joblib")
model = saved["model"]
encoder = saved["encoder"]
num_imputer = saved["num_imputer"]
cat_imputer = saved["cat_imputer"]
features = saved["features"]

num_cols = ["Number_of_Vehicles","Engine_CC_Mean","Speed_limit","Hour"]
cat_cols = ["Weather_Conditions","Road_Surface_Conditions","Light_Conditions","Urban_or_Rural_Area","Day_of_Week"]

# ===============================
# SIDEBAR INPUTS
# ===============================
st.sidebar.markdown('<div class="sidebar-title">🧾 Accident Details</div>', unsafe_allow_html=True)

def user_input():
    data = {}
    data["Number_of_Vehicles"] = st.sidebar.number_input("🚗 Number of Vehicles", 1, 20, 1)
    data["Engine_CC_Mean"] = st.sidebar.number_input("🔧 Engine Capacity (CC)", 50, 8000, 1500)
    data["Speed_limit"] = st.sidebar.number_input("🚧 Speed Limit", 0, 150, 30)
    data["Hour"] = st.sidebar.slider("⏰ Hour of Accident", 0, 23, 12)
    data["Weather_Conditions"] = st.sidebar.selectbox("🌦 Weather Conditions",
        ["Fine no high winds","Raining no high winds","Raining + high winds","Snowing","Fog or mist","Other"])
    data["Road_Surface_Conditions"] = st.sidebar.selectbox("🛣 Road Surface Conditions",
        ["Dry","Wet or damp","Snow","Frost or ice","Flood over 3cm deep"])
    data["Light_Conditions"] = st.sidebar.selectbox("💡 Light Conditions",
        ["Daylight","Darkness - lights lit","Darkness - no lighting","Darkness - lights unlit"])
    data["Urban_or_Rural_Area"] = st.sidebar.selectbox("🏙 Area Type", ["Urban","Rural"])
    data["Day_of_Week"] = st.sidebar.selectbox("📅 Day of Week",
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    return pd.DataFrame([data])

input_df = user_input()

# ===============================
# PREPROCESS (UNCHANGED)
# ===============================
input_df = input_df[features]
input_df[num_cols] = num_imputer.transform(input_df[num_cols])
input_df[cat_cols] = encoder.transform(cat_imputer.transform(input_df[cat_cols].astype(str)))

# ===============================
# PREDICTION
# ===============================
if st.sidebar.button("🚦 Predict Severity"):
    pred = model.predict(input_df)[0]
    severity_map = {0:"Fatal",1:"Serious",2:"Slight"}

    st.markdown(f'<div class="result-box">{severity_map[pred]} Accident</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

    probs = model.predict_proba(input_df)[0]
    prob_df = pd.DataFrame({
        "Severity": ["Fatal","Serious","Slight"],
        "Probability (%)": probs*100
    })

    # ===============================
    # TABS (ORDER FIXED)
    # ===============================
    tab1, tab2, tab3 = st.tabs(["ℹ️ Model Info","📊 Visuals","🧠 Confidence"])


    # ================= TAB 1: MODEL INFO =================
    with tab1:
        st.markdown("### ℹ️ About This Model")

        st.markdown("#### Purpose")
        st.markdown("""
This model predicts the **severity of traffic accidents** into three categories:  
- 🔴 **Fatal**  
- 🟠 **Serious**  
- 🟢 **Slight**
""")

        
        st.markdown("#### Data Overview")
        st.markdown("""
- Data sourced from **official UK road accident records**
- Combined **Accident** and **Vehicle** information
- Trained on **50,000 accident records**
- Final dataset contained **9 input features**
""")

        st.markdown("#### Features Used")
        st.markdown("""
The following features were selected based on their relevance to accident severity:
- Number of vehicles involved  
- Engine capacity (CC)  
- Speed limit  
- Hour of accident  
- Weather conditions  
- Road surface conditions  
- Light conditions  
- Urban or Rural area  
- Day of the week
""")

        st.markdown("#### Preprocessing & Encoding")
        st.markdown("""
- **Numeric features** are imputed using the **median** value  
- **Categorical features** are imputed using the **most frequent value**  
- **Ordinal Encoding** is applied to categorical variables, converting them into numerical form while handling unseen categories safely
""")

        st.markdown("#### Model Details")
        st.markdown("""
- **Model Used:** HistGradientBoostingClassifier (Scikit-learn)
""")

        st.markdown("#### Note")
        st.markdown("""
This model is intended for **educational, learning, and analytical purposes only**.
""")
        

    # -------- TAB 2: VISUALS --------
    with tab2:
        st.markdown("### Probability Distribution")
        chart = alt.Chart(prob_df).mark_bar(size=40).encode(
            x="Severity",
            y="Probability (%)",
            color=alt.Color("Severity", scale=alt.Scale(
                range=["#ef4444","#f59e0b","#10b981"]
            )),
            tooltip=["Severity","Probability (%)"]
        ).properties(height=320)
        st.altair_chart(chart, use_container_width=True)

        st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

        fig = go.Figure()
        for i,row in prob_df.iterrows():
            fig.add_bar(
                y=[row["Severity"]],
                x=[row["Probability (%)"]],
                orientation="h",
                text=f"{row['Probability (%)']:.2f}%",
                textposition="inside"
            )
        fig.update_layout(
            height=320,
            paper_bgcolor="#121212",
            plot_bgcolor="#121212",
            font_color="black",
            xaxis=dict(range=[0,100])
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------- TAB 3: CONFIDENCE --------
    with tab3:
        cols = st.columns(3)
        for col,(label,value) in zip(cols,zip(prob_df["Severity"],prob_df["Probability (%)"])):
            with col:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.metric(label, f"{value:.2f}%")
                st.progress(int(value))
                st.markdown('</div>', unsafe_allow_html=True)

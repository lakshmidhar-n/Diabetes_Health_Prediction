
import io
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# =========================================================
# THEME COLOURS (replaces Streamlit's default red/orange)
# =========================================================
try:
    st._config.set_option("theme.base", "dark")
    st._config.set_option("theme.primaryColor", "#00d9d0")
    st._config.set_option("theme.backgroundColor", "#101426")
    st._config.set_option("theme.secondaryBackgroundColor", "#171c31")
    st._config.set_option("theme.textColor", "#e4e9ff")
except Exception:
    pass

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Diabetes Health Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# DARK NAVY + PURPLE + CYAN THEME
# =========================================================
st.markdown("""
<style>
/* ==========================================
   BASE
   ========================================== */
.stApp {
    background: linear-gradient(135deg, #101426, #151b32 65%, #102638);
    color: #e4e9ff;
    font-family: "Inter", "Segoe UI", sans-serif;
}

[data-testid="stHeader"] { background: #101426; }
[data-testid="stAppViewContainer"] { background: transparent; }

h1, h2, h3, h4 {
    color: #f4f2ff !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}

p, li, label { color: #cbd5f5; }

[data-testid="stCaptionContainer"] p { color: #aab6d6 !important; }

hr { border-color: #303958 !important; }

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0c1122; }
::-webkit-scrollbar-thumb {
    background: #303958;
    border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover { background: #00bfc4; }

/* ==========================================
   SIDEBAR
   ========================================== */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: #0c1122 !important;
}

[data-testid="stSidebar"] { border-right: 1px solid #282e4d; }

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p { color: #dce4ff !important; }

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #5ffff0 !important; }

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #c5d5f5 !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stButton button,
[data-testid="stSidebar"] .stDownloadButton button {
    background: linear-gradient(100deg, #3446a5, #087f91) !important;
    color: #ffffff !important;
    border: 1px solid #00bfc4 !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] .stButton button:hover,
[data-testid="stSidebar"] .stDownloadButton button:hover {
    background: linear-gradient(100deg, #5549d8, #00a9ae) !important;
    border-color: #5ffff0 !important;
}

/* ==========================================
   TABS
   ========================================== */
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    gap: 8px !important;
    background: #11182c !important;
    border: 1px solid #303958 !important;
    border-radius: 14px !important;
    padding: 7px !important;
    margin-top: 14px;
    margin-bottom: 24px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #b9c8e8 !important;
    border-radius: 10px !important;
    border: 1px solid transparent !important;
    padding: 12px 18px;
    font-size: 0.94rem;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
    background: #20304b !important;
    color: #5ffff0 !important;
    border-color: #00bfc4 !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(100deg, #773de2, #356fe8) !important;
    color: #ffffff !important;
    border: 1px solid #8067ff !important;
    box-shadow: 0 3px 12px rgba(94, 91, 245, 0.25);
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #00d9d0 !important;
}

.stTabs [data-baseweb="tab-border"] {
    background-color: #303958 !important;
}

/* ==========================================
   SELECT BOXES (ALL DROPDOWNS)
   ========================================== */
.stApp div[data-baseweb="select"] > div {
    background-color: #1a213b !important;
    border: 1px solid #303958 !important;
    border-radius: 10px !important;
    color: #f0f5ff !important;
    transition: all 0.2s ease;
}

.stApp div[data-baseweb="select"] > div:hover,
.stApp div[data-baseweb="select"] > div:focus-within {
    border-color: #00d9d0 !important;
    box-shadow: 0 0 0 1px #00d9d0 !important;
}

.stApp div[data-baseweb="select"] span,
.stApp div[data-baseweb="select"] div {
    color: #f0f5ff !important;
}

.stApp div[data-baseweb="select"] svg {
    fill: #5ffff0 !important;
    color: #5ffff0 !important;
}

[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #17243d !important;
    border: 1px solid #00bfc4 !important;
}

/* ==========================================
   DROPDOWN LIST (OPEN MENU) — NAVY + PURPLE/BLUE HOVER
   ========================================== */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div {
    background-color: #111a30 !important;
    border-radius: 10px !important;
}

div[data-baseweb="popover"] ul,
ul[role="listbox"],
div[data-baseweb="menu"] {
    background-color: #111a30 !important;
    border: 1px solid #304467 !important;
    border-radius: 10px !important;
    padding: 4px !important;
}

/* normal option */
div[data-baseweb="popover"] li,
ul[role="listbox"] li,
li[role="option"] {
    background-color: transparent !important;
    color: #dce8ff !important;
    border-radius: 8px !important;
    margin: 2px 0 !important;
    transition: all 0.15s ease;
}

/* currently selected option: soft indigo + cyan accent */
div[data-baseweb="popover"] li[aria-selected="true"],
ul[role="listbox"] li[aria-selected="true"],
li[role="option"][aria-selected="true"] {
    background-color: #232b57 !important;
    color: #5ffff0 !important;
    box-shadow: inset 3px 0 0 #00d9d0;
}

/* hovered / keyboard-focused option: purple -> blue gradient */
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] li:focus,
ul[role="listbox"] li:hover,
ul[role="listbox"] li:focus,
li[role="option"]:hover,
li[role="option"]:focus {
    background: linear-gradient(100deg, #773de2, #356fe8) !important;
    color: #ffffff !important;
}

div[data-baseweb="popover"] li:hover *,
ul[role="listbox"] li:hover *,
li[role="option"]:hover * {
    color: #ffffff !important;
}

/* ==========================================
   TEXT / NUMBER INPUTS
   ========================================== */
.stApp div[data-baseweb="input"],
.stApp div[data-baseweb="base-input"] {
    background-color: #1a213b !important;
    border-radius: 9px !important;
}

.stApp div[data-baseweb="input"] {
    border: 1px solid #303958 !important;
}

.stApp div[data-baseweb="input"]:hover,
.stApp div[data-baseweb="input"]:focus-within {
    border-color: #00d9d0 !important;
    box-shadow: 0 0 0 1px #00d9d0 !important;
}

.stApp input {
    color: #f0f5ff !important;
    background-color: transparent !important;
}

.stApp button[kind="stepper"],
[data-testid="stNumberInput"] button {
    background-color: #1a213b !important;
    color: #5ffff0 !important;
}

.stApp button[kind="stepper"]:hover,
[data-testid="stNumberInput"] button:hover {
    background-color: #24566b !important;
    color: #ffffff !important;
}

/* ==========================================
   MULTISELECT TAGS — PURPLE/BLUE (replaces red)
   ========================================== */
[data-baseweb="tag"],
span[data-baseweb="tag"] {
    background: linear-gradient(100deg, #773de2, #356fe8) !important;
    background-color: #5a4de0 !important;
    color: #ffffff !important;
    border: 1px solid #8067ff !important;
    border-radius: 8px !important;
}

[data-baseweb="tag"] span,
[data-baseweb="tag"] div {
    color: #ffffff !important;
}

[data-baseweb="tag"] svg {
    fill: #5ffff0 !important;
    color: #5ffff0 !important;
}

[data-baseweb="tag"]:hover {
    background: linear-gradient(100deg, #8b52f0, #4a85ff) !important;
    border-color: #5ffff0 !important;
}

/* ==========================================
   SLIDERS — CYAN (replaces red/orange)
   ========================================== */
[data-testid="stSlider"] [role="slider"] {
    background-color: #00d9d0 !important;
    border: 2px solid #eafaff !important;
    box-shadow: 0 0 8px rgba(0, 217, 208, 0.35) !important;
}

[data-testid="stSlider"] [role="slider"]:hover,
[data-testid="stSlider"] [role="slider"]:focus {
    background-color: #5ffff0 !important;
    border-color: #ffffff !important;
    box-shadow: 0 0 10px rgba(0, 217, 208, 0.55) !important;
}

[data-testid="stSliderThumbValue"],
[data-testid="stSliderThumbValue"] p,
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: #5ffff0 !important;
}

[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"],
[data-testid="stSlider"] [data-testid="stTickBar"] {
    color: #b9c8e8 !important;
}

/* ==========================================
   CHECKBOX / RADIO / TOGGLE
   ========================================== */
input[type="checkbox"],
input[type="radio"] { accent-color: #00d9d0 !important; }

/* ==========================================
   METRIC CARDS
   ========================================== */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #c45be8 0%, #8b5cf6 100%);
    border: 1px solid #b66bf5;
    border-radius: 16px;
    padding: 20px 18px;
    box-shadow: 0 6px 20px rgba(139, 92, 246, 0.18);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #f8eaff !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #ffffff !important;
    font-weight: 800 !important;
}

[data-testid="stMetricDelta"] { color: #e0fff9 !important; }

[data-testid="stHorizontalBlock"] [data-testid="stColumn"]:nth-child(even)
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #12cfc8 0%, #438dff 100%);
    border-color: #38dce0;
}

/* ==========================================
   BUTTONS
   ========================================== */
.stButton button,
.stDownloadButton button,
.stFormSubmitButton button,
button[kind="primary"],
button[kind="secondaryFormSubmit"] {
    background: linear-gradient(100deg, #8b42df, #4a65ee) !important;
    color: #ffffff !important;
    border: 1px solid #7954e8 !important;
    border-radius: 10px !important;
    font-weight: 650 !important;
}

.stButton button:hover,
.stDownloadButton button:hover,
.stFormSubmitButton button:hover {
    background: linear-gradient(100deg, #a653ef, #587dff) !important;
    border-color: #00d9d0 !important;
    color: #ffffff !important;
}

.stButton button:focus,
.stDownloadButton button:focus,
.stFormSubmitButton button:focus {
    outline: 2px solid #00d9d0 !important;
    outline-offset: 2px !important;
    box-shadow: none !important;
}

/* ==========================================
   TABLES, EXPANDERS, FORMS, UPLOADER, ALERTS
   ========================================== */
[data-testid="stDataFrame"] {
    border: 1px solid #303958;
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stExpander"] {
    background: #171d33;
    border: 1px solid #303958;
    border-radius: 12px;
}

[data-testid="stForm"] {
    background: rgba(23, 29, 51, 0.55);
    border: 1px solid #303958;
    border-radius: 14px;
}

[data-testid="stFileUploader"] {
    background: #171d33;
    border: 1px dashed #58618b;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stFileUploader"] button {
    background: linear-gradient(100deg, #3446a5, #087f91) !important;
    color: #ffffff !important;
    border: 1px solid #00bfc4 !important;
}

/* alerts: navy cards with a theme-coloured left accent (no orange/yellow) */
[data-testid="stAlert"],
[data-testid="stAlert"] > div {
    background: #1a213b !important;
    border: 1px solid #3b456b !important;
    border-left: 4px solid #8067ff !important;
    border-radius: 12px !important;
    color: #e4e9ff !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] div { color: #e4e9ff !important; }

[data-testid="stAlert"] svg {
    fill: #5ffff0 !important;
    color: #5ffff0 !important;
}

a { color: #5ffff0 !important; }

[data-testid="stProgress"] > div > div > div > div {
    background-color: #00d9d0 !important;
}

/* ==========================================
   WELCOME / UPLOAD SCREEN
   ========================================== */
.welcome-card {
    background: linear-gradient(135deg, #171d3a, #12263a);
    border: 1px solid #303958;
    border-left: 4px solid #00d9d0;
    border-radius: 16px;
    padding: 28px 30px;
    margin: 18px 0 22px;
}

.welcome-card h3 {
    margin: 0 0 8px 0;
    color: #5ffff0 !important;
}

.welcome-card p {
    margin: 0;
    color: #cbd5f5;
    font-size: 1rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# FILES AND CONSTANTS
# =========================================================
DATA_FILE = "diabetes_binary_health_indicators_BRFSS2015.csv"
COMPARISON_FILE = "diabetes_model_comparison.csv"

LOGISTIC_FILE = "logistic_regression_model.pkl"
FOREST_FILE = "random_forest_model.pkl"
SCALER_FILE = "diabetes_feature_scaler.pkl"

FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker",
    "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits",
    "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
    "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth",
    "DiffWalk", "Sex", "Age", "Education", "Income"
]

BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
    "DiffWalk", "Sex"
]

LABELS = {
    "HighBP": "High Blood Pressure",
    "HighChol": "High Cholesterol",
    "CholCheck": "Cholesterol Check",
    "BMI": "Body Mass Index",
    "Smoker": "Smoker",
    "Stroke": "History of Stroke",
    "HeartDiseaseorAttack": "Heart Disease / Attack",
    "PhysActivity": "Physical Activity",
    "Fruits": "Fruit Consumption",
    "Veggies": "Vegetable Consumption",
    "HvyAlcoholConsump": "Heavy Alcohol Consumption",
    "AnyHealthcare": "Healthcare Coverage",
    "NoDocbcCost": "Could Not Visit Doctor Due to Cost",
    "GenHlth": "General Health",
    "MentHlth": "Mental Health Days",
    "PhysHlth": "Physical Health Days",
    "DiffWalk": "Difficulty Walking",
    "Sex": "Sex",
    "Age": "Age Category",
    "Education": "Education",
    "Income": "Income"
}

AGE_LABELS = {
    1: "18–24", 2: "25–29", 3: "30–34",
    4: "35–39", 5: "40–44", 6: "45–49",
    7: "50–54", 8: "55–59", 9: "60–64",
    10: "65–69", 11: "70–74", 12: "75–79", 13: "80+"
}

HEALTH_LABELS = {
    1: "Excellent",
    2: "Very Good",
    3: "Good",
    4: "Fair",
    5: "Poor"
}

EDUCATION_LABELS = {
    1: "Never attended / Kindergarten",
    2: "Grades 1–8",
    3: "Grades 9–11",
    4: "Grade 12 / GED",
    5: "College 1–3 years",
    6: "College 4+ years"
}

INCOME_LABELS = {
    1: "Below $10,000",
    2: "$10,000–$14,999",
    3: "$15,000–$19,999",
    4: "$20,000–$24,999",
    5: "$25,000–$34,999",
    6: "$35,000–$49,999",
    7: "$50,000–$74,999",
    8: "$75,000 or more"
}

COLORS = ["#b54cff", "#00d9d0", "#438dff", "#7958ff", "#40c4ff"]

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def style_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#dfe7ff",
            family="Inter, Segoe UI, sans-serif"
        ),
        title_font=dict(color="#f1f3ff", size=18),
        margin=dict(l=20, r=20, t=60, b=35),
        xaxis=dict(
            gridcolor="#292f4c",
            zerolinecolor="#292f4c",
            tickfont=dict(color="#dfe7ff"),
            title_font=dict(color="#cbd4ff")
        ),
        yaxis=dict(
            gridcolor="#292f4c",
            zerolinecolor="#292f4c",
            tickfont=dict(color="#dfe7ff"),
            title_font=dict(color="#cbd4ff")
        ),
        legend=dict(font=dict(color="#dfe7ff")),
        legend_title_font=dict(color="#cbd4ff"),
        hoverlabel=dict(
            bgcolor="#1a213b",
            bordercolor="#00d9d0",
            font=dict(color="#ffffff")
        )
    )
    return fig


def vertical_box_plot(data, title, y_label, color, fill):
    fig = px.box(
        y=data,
        points="outliers",
        title=title,
        labels={"y": y_label}
    )

    fig.update_traces(
        marker=dict(color=color, size=6),
        line=dict(color=color, width=2),
        fillcolor=fill,
        width=0.4
    )

    fig = style_chart(fig)

    fig.update_layout(
        height=520,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(title=y_label, range=[-1, 31])
    )

    return fig


def get_target(data):
    for col in [
        "Diabetes_binary", "diabetes_binary",
        "Diabetes_012", "diabetes_012"
    ]:
        if col in data.columns:
            return col
    return None


def clean_dataset(data):
    data.columns = data.columns.astype(str).str.strip()

    for col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col])
        except (ValueError, TypeError):
            pass

    return data


@st.cache_data(show_spinner="Loading dataset...")
def load_csv_from_path(path):
    return clean_dataset(pd.read_csv(path))


@st.cache_data(show_spinner="Reading uploaded file...")
def load_csv_from_bytes(content):
    return clean_dataset(pd.read_csv(io.BytesIO(content)))


@st.cache_resource(show_spinner=False)
def _load_model_cached(filename):
    return joblib.load(filename)


def load_model(filename):
    if not os.path.exists(filename):
        return None
    try:
        return _load_model_cached(filename)
    except Exception as error:
        st.error(f"Could not load {filename}: {error}")
        return None


def diabetes_positive(series, target_name):
    if target_name.lower() == "diabetes_012":
        return series.isin([1, 2])
    return series.eq(1)


# =========================================================
# HEADER
# =========================================================
st.title("Diabetes Health Indicators")
st.caption(
    "Interactive analytics  •  Risk factors  •  "
    "Model performance  •  Prediction"
)

# =========================================================
# SIDEBAR HEADER + RESET
# =========================================================
st.sidebar.markdown("""
<div style="padding:8px 0 16px;border-bottom:1px solid #303958;margin-bottom:18px">
    <div style="font-size:21px;font-weight:750;color:#f4f5ff">
        ◈ Health Analytics
    </div>
    <div style="font-size:12px;color:#8e9abd;margin-top:5px">
        DASHBOARD FILTERS
    </div>
</div>
""", unsafe_allow_html=True)

if "reset_count" not in st.session_state:
    st.session_state.reset_count = 0

if st.sidebar.button("↻ Reset All Filters", use_container_width=True):
    st.session_state.reset_count += 1
    st.rerun()

reset = st.session_state.reset_count

# =========================================================
# DATA LOADING — ASK FOR UPLOAD WHEN NO DATA FILE IS AVAILABLE
# =========================================================
sidebar_upload = st.sidebar.file_uploader(
    "Upload CSV", type=["csv"], key="sidebar_upload"
)

local_file_exists = os.path.exists(DATA_FILE)
main_upload = None

if sidebar_upload is None and not local_file_exists:
    # No dataset yet: show a clear upload screen in the main area
    st.markdown("""
    <div class="welcome-card">
        <h3>Welcome — upload your dataset to begin</h3>
        <p>
            Upload the Diabetes Health Indicators CSV file
            (<b>diabetes_binary_health_indicators_BRFSS2015.csv</b>)
            below. Once it loads, the full dashboard with filters,
            risk-factor charts, model results and predictions
            will appear here.
        </p>
    </div>
    """, unsafe_allow_html=True)

    main_upload = st.file_uploader(
        "Drag and drop your CSV file here, or click Browse files",
        type=["csv"],
        key="main_upload"
    )

data_source = sidebar_upload if sidebar_upload is not None else main_upload

try:
    if data_source is not None:
        df = load_csv_from_bytes(data_source.getvalue())
    elif local_file_exists:
        df = load_csv_from_path(DATA_FILE)
    else:
        st.stop()
except Exception as error:
    st.error(f"Unable to load CSV: {error}")
    st.stop()

TARGET = get_target(df)

if TARGET:
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    if TARGET.lower() == "diabetes_012":
        status_labels = {
            0: "No Diabetes",
            1: "Prediabetes",
            2: "Diabetes"
        }
    else:
        status_labels = {
            0: "No Diabetes",
            1: "Diabetes"
        }

    df["Diabetes Status"] = (
        df[TARGET].map(status_labels).fillna("Unknown")
    )

view = df.copy()

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.divider()
st.sidebar.subheader("Age & BMI")

if "Age" in view:
    age_values = pd.to_numeric(
        view["Age"], errors="coerce"
    ).dropna()

    if not age_values.empty:
        age_range = st.sidebar.slider(
            "Age Category",
            int(age_values.min()),
            int(age_values.max()),
            (int(age_values.min()), int(age_values.max())),
            key=f"age_{reset}"
        )

        view = view[
            pd.to_numeric(view["Age"], errors="coerce").between(*age_range)
        ]

if "BMI" in view:
    bmi_values = pd.to_numeric(
        view["BMI"], errors="coerce"
    ).dropna()

    if not bmi_values.empty and bmi_values.min() < bmi_values.max():
        bmi_range = st.sidebar.slider(
            "BMI Range",
            float(bmi_values.min()),
            float(bmi_values.max()),
            (float(bmi_values.min()), float(bmi_values.max())),
            key=f"bmi_{reset}"
        )

        view = view[
            pd.to_numeric(view["BMI"], errors="coerce").between(*bmi_range)
        ]

with st.sidebar.expander(
    "Blood Pressure, Cholesterol & Lifestyle",
    expanded=True
):
    for col in BINARY_FEATURES:
        if col in view:
            choice = st.selectbox(
                LABELS.get(col, col),
                ["All", "Yes", "No"],
                key=f"{col}_{reset}"
            )

            if choice != "All":
                expected = 1 if choice == "Yes" else 0
                view = view[
                    pd.to_numeric(view[col], errors="coerce") == expected
                ]

for col, mapping, heading in [
    ("GenHlth", HEALTH_LABELS, "General Health"),
    ("Education", EDUCATION_LABELS, "Education Level"),
    ("Income", INCOME_LABELS, "Income Level")
]:
    if col in view:
        codes = sorted(
            pd.to_numeric(view[col], errors="coerce")
            .dropna().astype(int).unique()
        )

        if codes:
            options = {
                mapping.get(int(value), str(value)): int(value)
                for value in codes
            }

            selected = st.sidebar.multiselect(
                heading,
                list(options.keys()),
                default=list(options.keys()),
                key=f"{col}_{reset}"
            )

            view = view[
                view[col].isin([options[item] for item in selected])
            ]

# Download filtered data
st.sidebar.divider()
st.sidebar.subheader("Export Data")

st.sidebar.download_button(
    "⬇ Download Filtered CSV",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="filtered_diabetes_health_data.csv",
    mime="text/csv",
    use_container_width=True
)

# =========================================================
# TOP NAVIGATION
# =========================================================
overview_tab, risk_tab, model_tab, prediction_tab = st.tabs([
    "◈ Overview",
    "✦ Risk Factor Analysis",
    "▥ Model Performance",
    "◎ Diabetes Prediction"
])

# =========================================================
# TAB 1: OVERVIEW
# =========================================================
with overview_tab:
    st.subheader("Dashboard Overview")

    if view.empty:
        st.warning(
            "No records match these filters. Reset filters and try again."
        )

    else:
        total_records = len(view)

        avg_bmi = (
            pd.to_numeric(view["BMI"], errors="coerce").mean()
            if "BMI" in view else np.nan
        )

        diabetes_rate = np.nan

        if TARGET:
            diabetes_rate = (
                diabetes_positive(view[TARGET], TARGET).mean() * 100
            )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Filtered Records", f"{total_records:,}")

        col2.metric(
            "Diabetes Rate",
            f"{diabetes_rate:.1f}%"
            if pd.notna(diabetes_rate) else "N/A"
        )

        col3.metric(
            "Average BMI",
            f"{avg_bmi:.1f}" if pd.notna(avg_bmi) else "N/A"
        )

        col4.metric(
            "Available Indicators",
            len([feature for feature in FEATURES if feature in view])
        )

        st.divider()
        st.subheader("Automatic Insights")

        if pd.notna(diabetes_rate):
            st.write(
                f"• Diabetes-positive rate in the selected records: "
                f"**{diabetes_rate:.1f}%**."
            )

        if pd.notna(avg_bmi):
            st.write(
                f"• Average BMI in the selected records: **{avg_bmi:.1f}**."
            )

        if "GenHlth" in view:
            average_health = pd.to_numeric(
                view["GenHlth"], errors="coerce"
            ).mean()

            if pd.notna(average_health):
                st.write(
                    f"• Average general-health code: "
                    f"**{average_health:.2f}** "
                    "(1 = Excellent, 5 = Poor)."
                )

        st.divider()

        left, right = st.columns(2)

        with left:
            if "Diabetes Status" in view:
                counts = (
                    view["Diabetes Status"]
                    .value_counts()
                    .rename_axis("Status")
                    .reset_index(name="Records")
                )

                fig = px.pie(
                    counts,
                    names="Status",
                    values="Records",
                    hole=0.55,
                    title="Diabetes Status Distribution",
                    color_discrete_sequence=COLORS
                )

                st.plotly_chart(
                    style_chart(fig), use_container_width=True
                )

            elif "BMI" in view:
                fig = px.histogram(
                    view,
                    x="BMI",
                    title="BMI Distribution",
                    color_discrete_sequence=["#00d9d0"]
                )

                st.plotly_chart(
                    style_chart(fig), use_container_width=True
                )

        with right:
            if TARGET and "Age" in view:
                temp = view.dropna(subset=["Age", TARGET]).copy()

                if not temp.empty:
                    temp["_positive"] = diabetes_positive(
                        temp[TARGET], TARGET
                    ).astype(int)

                    rates = (
                        temp.groupby("Age", as_index=False)["_positive"]
                        .mean()
                    )

                    rates["Rate (%)"] = rates["_positive"] * 100

                    rates["Age Group"] = (
                        rates["Age"].map(AGE_LABELS)
                        .fillna(rates["Age"].astype(str))
                    )

                    fig = px.bar(
                        rates,
                        x="Age Group",
                        y="Rate (%)",
                        title="Diabetes Rate by Age",
                        color_discrete_sequence=["#b54cff"]
                    )

                    st.plotly_chart(
                        style_chart(fig), use_container_width=True
                    )

        if "GenHlth" in view:
            temp = view.copy()
            temp["Health Description"] = temp["GenHlth"].map(HEALTH_LABELS)

            counts = (
                temp["Health Description"]
                .value_counts()
                .rename_axis("General Health")
                .reset_index(name="Records")
            )

            fig = px.bar(
                counts,
                x="General Health",
                y="Records",
                title="General Health Distribution",
                color_discrete_sequence=["#438dff"]
            )

            st.plotly_chart(
                style_chart(fig), use_container_width=True
            )


# =========================================================
# TAB 2: RISK FACTOR ANALYSIS
# =========================================================
with risk_tab:
    st.subheader("Risk Factor Analysis")

    if view.empty:
        st.warning("No records match the selected filters.")

    elif not TARGET:
        st.info("A diabetes target column is required for this analysis.")

    else:
        available_factors = [
            col for col in BINARY_FEATURES if col in view
        ]

        selected_factors = st.multiselect(
            "Choose risk factors to compare",
            available_factors,
            default=available_factors[:5],
            format_func=lambda value: LABELS.get(value, value)
        )

        metric = st.selectbox(
            "Choose comparison metric",
            ["Diabetes Rate (%)", "Number of Records"]
        )

        # Map the dropdown choice to the real column name in the table
        metric_column = {
            "Diabetes Rate (%)": "Diabetes Rate (%)",
            "Number of Records": "Records"
        }[metric]

        results = []

        for factor in selected_factors:
            temp = view[[factor, TARGET]].copy()

            temp[factor] = pd.to_numeric(
                temp[factor], errors="coerce"
            )

            temp[TARGET] = pd.to_numeric(
                temp[TARGET], errors="coerce"
            )

            temp = temp.dropna()

            for value in sorted(temp[factor].unique()):
                group = temp[temp[factor] == value]

                positive = diabetes_positive(group[TARGET], TARGET)

                results.append({
                    "Risk Factor": LABELS.get(factor, factor),
                    "Response": (
                        "Yes" if value == 1
                        else "No" if value == 0
                        else str(value)
                    ),
                    "Records": len(group),
                    "Diabetes Rate (%)": positive.mean() * 100
                })

        if results:
            results_df = pd.DataFrame(results)

            fig = px.bar(
                results_df,
                x="Risk Factor",
                y=metric_column,
                color="Response",
                barmode="group",
                title=f"{metric} by Risk Factor",
                labels={metric_column: metric},
                color_discrete_sequence=["#00d9d0", "#b54cff"]
            )

            fig.update_layout(xaxis_tickangle=-25)

            st.plotly_chart(
                style_chart(fig), use_container_width=True
            )

        elif not selected_factors:
            st.info("Select at least one risk factor to display the chart.")

    # Physical & Mental health box plots (vertical, side by side)
    st.divider()

    phys_col, ment_col = st.columns(2)

    with phys_col:
        st.subheader("Physical Health Days — Outlier Analysis")

        if "PhysHlth" in view:
            physical_data = pd.to_numeric(
                view["PhysHlth"], errors="coerce"
            ).dropna()

            if not physical_data.empty:
                fig = vertical_box_plot(
                    physical_data,
                    "Physical Health Days Not Good",
                    "Physical Health Days (0–30)",
                    "#00d9d0",
                    "rgba(0,217,208,0.22)"
                )

                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "The box represents the middle 50% of values. "
                    "Points beyond the whiskers are potential outliers."
                )
            else:
                st.info("No valid physical health data is available.")
        else:
            st.info("Physical health data is unavailable.")

    with ment_col:
        st.subheader("Mental Health Days — Outlier Analysis")

        if "MentHlth" in view:
            mental_data = pd.to_numeric(
                view["MentHlth"], errors="coerce"
            ).dropna()

            if not mental_data.empty:
                fig = vertical_box_plot(
                    mental_data,
                    "Mental Health Days Not Good",
                    "Mental Health Days (0–30)",
                    "#b54cff",
                    "rgba(181,76,255,0.22)"
                )

                st.plotly_chart(fig, use_container_width=True)

                st.caption(
                    "The box represents the middle 50% of values. "
                    "Points beyond the whiskers are potential outliers."
                )
            else:
                st.info("No valid mental health data is available.")
        else:
            st.info("Mental health data is unavailable.")

    # Correlation heatmap
    numeric_data = view.select_dtypes(include=np.number)

    if len(numeric_data.columns) >= 2:
        st.divider()
        st.subheader("Indicator Correlations")

        correlation_options = [
            col for col in numeric_data.columns if col != TARGET
        ]

        selected_columns = st.multiselect(
            "Choose indicators for correlation chart",
            correlation_options,
            default=correlation_options[:10],
            key="correlation_columns"
        )

        if len(selected_columns) >= 2:
            corr_matrix = view[selected_columns].corr(numeric_only=True)

            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                title="Correlation Heatmap",
                color_continuous_scale=[
                    [0.0, "#9b4dff"],   # strong negative: purple
                    [0.5, "#1c2342"],   # near zero: dark navy
                    [1.0, "#0fb5b8"]    # strong positive: teal
                ],
                zmin=-1,
                zmax=1
            )

            fig = style_chart(fig)

            fig.update_traces(
                textfont=dict(color="#ffffff", size=13),
                xgap=3,
                ygap=3
            )

            fig.update_layout(
                height=max(480, 48 * len(selected_columns) + 160),
                margin=dict(l=20, r=20, t=60, b=90),
                xaxis=dict(
                    tickangle=-35,
                    showgrid=False,
                    tickfont=dict(color="#e4ecff", size=12)
                ),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(color="#e4ecff", size=12)
                ),
                coloraxis_colorbar=dict(
                    tickfont=dict(color="#dfe7ff"),
                    outlinewidth=0,
                    thickness=14
                )
            )

            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# TAB 3: MODEL PERFORMANCE
# =========================================================
with model_tab:
    st.subheader("Model Performance Comparison")

    if os.path.exists(COMPARISON_FILE):
        try:
            comp = pd.read_csv(COMPARISON_FILE)
            comp.columns = comp.columns.astype(str).str.strip()

            st.dataframe(
                comp,
                use_container_width=True,
                hide_index=True
            )

            normalized = {
                str(col).lower().replace("_", "").replace("-", "").replace(" ", ""): col
                for col in comp.columns
            }

            model_col = next(
                (
                    normalized[key]
                    for key in [
                        "model", "modelname", "classifier", "algorithm"
                    ]
                    if key in normalized
                ),
                None
            )

            aliases = {
                "Accuracy": ["accuracy"],
                "Precision": ["precision"],
                "Recall": ["recall"],
                "F1-Score": ["f1score", "f1"],
                "AUC": ["auc", "rocauc", "rocaucscore"]
            }

            metric_cols = {}

            for display_name, keys in aliases.items():
                for key in keys:
                    if key in normalized:
                        column = normalized[key]

                        if pd.api.types.is_numeric_dtype(comp[column]):
                            metric_cols[display_name] = column
                            break

            if model_col and metric_cols:
                selected_metrics = st.multiselect(
                    "Choose metrics to compare",
                    list(metric_cols.keys()),
                    default=[
                        metric_name for metric_name in [
                            "Accuracy", "Precision", "Recall", "F1-Score"
                        ] if metric_name in metric_cols
                    ]
                )

                if selected_metrics:
                    chart_data = comp[
                        [model_col] +
                        [metric_cols[m] for m in selected_metrics]
                    ].copy()

                    rename_columns = {
                        metric_cols[m]: m for m in selected_metrics
                    }

                    chart_data = chart_data.rename(
                        columns=rename_columns
                    )

                    long_data = chart_data.melt(
                        id_vars=[model_col],
                        var_name="Metric",
                        value_name="Score"
                    )

                    long_data["Score"] = pd.to_numeric(
                        long_data["Score"], errors="coerce"
                    )

                    long_data = long_data.dropna(subset=["Score"])

                    fig = px.bar(
                        long_data,
                        x=model_col,
                        y="Score",
                        color="Metric",
                        barmode="group",
                        title="Grouped Model Performance",
                        color_discrete_sequence=COLORS
                    )

                    if (
                        not long_data.empty
                        and long_data["Score"].min() >= 0
                        and long_data["Score"].max() <= 1
                    ):
                        fig.update_yaxes(range=[0, 1])

                    st.plotly_chart(
                        style_chart(fig), use_container_width=True
                    )

            else:
                st.info(
                    "The comparison CSV needs a model-name column and "
                    "numeric performance columns such as Accuracy, "
                    "Precision, Recall, F1-Score or AUC."
                )

        except Exception as error:
            st.error(f"Could not read model comparison: {error}")

    else:
        st.warning(
            f"{COMPARISON_FILE} was not found in the app folder. "
            "Add your model comparison CSV to display the results."
        )


# =========================================================
# TAB 4: DIABETES PREDICTION
# =========================================================
with prediction_tab:
    st.subheader("Try a Diabetes Prediction")
    st.caption(
        "Enter health-indicator values and compare your saved models."
    )

    logistic_model = load_model(LOGISTIC_FILE)
    forest_model = load_model(FOREST_FILE)
    scaler = load_model(SCALER_FILE)

    available_models = {}

    if logistic_model is not None:
        available_models["Logistic Regression"] = logistic_model

    if forest_model is not None:
        available_models["Random Forest"] = forest_model

    if not available_models:
        st.warning(
            "Saved prediction models were not found. "
            "Place the model PKL files beside app.py."
        )

    else:
        with st.form("prediction_form"):
            st.markdown("#### Personal & Health Indicators")

            col1, col2, col3 = st.columns(3)

            with col1:
                highbp = st.selectbox(
                    "High Blood Pressure", [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                highchol = st.selectbox(
                    "High Cholesterol", [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                cholcheck = st.selectbox(
                    "Cholesterol Check", [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                bmi = st.number_input(
                    "BMI", min_value=10.0, max_value=100.0,
                    value=27.0, step=0.5
                )

                smoker = st.selectbox(
                    "Smoker", [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                stroke = st.selectbox(
                    "History of Stroke", [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                heart = st.selectbox(
                    "Heart Disease / Attack", [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

            with col2:
                activity = st.selectbox(
                    "Physical Activity", [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                fruits = st.selectbox(
                    "Fruit Consumption", [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                veggies = st.selectbox(
                    "Vegetable Consumption", [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                alcohol = st.selectbox(
                    "Heavy Alcohol Consumption", [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                healthcare = st.selectbox(
                    "Healthcare Coverage", [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                no_doctor = st.selectbox(
                    "Could Not Visit Doctor Due to Cost", [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                genhlth = st.selectbox(
                    "General Health",
                    list(HEALTH_LABELS),
                    format_func=lambda x: f"{x} - {HEALTH_LABELS[x]}"
                )

            with col3:
                menthlth = st.number_input(
                    "Mental Health Days", 0, 30, 0
                )

                physhlth = st.number_input(
                    "Physical Health Days", 0, 30, 0
                )

                diffwalk = st.selectbox(
                    "Difficulty Walking", [0, 1],
                    format_func=lambda x: "Yes" if x else "No"
                )

                sex = st.selectbox(
                    "Sex Code", [0, 1],
                    format_func=lambda x:
                    "Female (0)" if x == 0 else "Male (1)"
                )

                age = st.selectbox(
                    "Age Category",
                    list(AGE_LABELS),
                    index=8,
                    format_func=lambda x: f"{x} - {AGE_LABELS[x]}"
                )

                education = st.selectbox(
                    "Education",
                    list(EDUCATION_LABELS),
                    index=5,
                    format_func=lambda x:
                    f"{x} - {EDUCATION_LABELS[x]}"
                )

                income = st.selectbox(
                    "Income",
                    list(INCOME_LABELS),
                    index=4,
                    format_func=lambda x:
                    f"{x} - {INCOME_LABELS[x]}"
                )

            selected_model_name = st.selectbox(
                "Prediction Model",
                list(available_models.keys())
            )

            submitted = st.form_submit_button(
                "Predict", use_container_width=True
            )

        if submitted:
            input_values = {
                "HighBP": highbp,
                "HighChol": highchol,
                "CholCheck": cholcheck,
                "BMI": bmi,
                "Smoker": smoker,
                "Stroke": stroke,
                "HeartDiseaseorAttack": heart,
                "PhysActivity": activity,
                "Fruits": fruits,
                "Veggies": veggies,
                "HvyAlcoholConsump": alcohol,
                "AnyHealthcare": healthcare,
                "NoDocbcCost": no_doctor,
                "GenHlth": genhlth,
                "MentHlth": menthlth,
                "PhysHlth": physhlth,
                "DiffWalk": diffwalk,
                "Sex": sex,
                "Age": age,
                "Education": education,
                "Income": income
            }

            input_df = pd.DataFrame(
                [[input_values[feature] for feature in FEATURES]],
                columns=FEATURES
            )

            model = available_models[selected_model_name]

            try:
                prepared_data = input_df

                if (
                    selected_model_name == "Logistic Regression"
                    and scaler is not None
                    and not hasattr(model, "named_steps")
                ):
                    prepared_data = scaler.transform(input_df)

                prediction = model.predict(prepared_data)[0]

                probabilities = (
                    model.predict_proba(prepared_data)[0]
                    if hasattr(model, "predict_proba")
                    else None
                )

                model_classes = (
                    model.classes_.tolist()
                    if hasattr(model, "classes_")
                    else None
                )

                st.divider()
                st.subheader("Prediction Result")

                if int(prediction) == 0:
                    st.success("Model Output: No Diabetes")
                elif int(prediction) == 1:
                    st.warning("Model Output: Diabetes")
                else:
                    st.warning(f"Model Output Class: {prediction}")

                st.write(
                    f"Selected model: **{selected_model_name}**"
                )

                if probabilities is not None:
                    if model_classes is None:
                        model_classes = list(range(len(probabilities)))

                    probability_values = {
                        int(cls): float(prob) * 100
                        for cls, prob in zip(
                            model_classes, probabilities
                        )
                    }

                    st.subheader("Prediction Probability")

                    col1, col2 = st.columns(2)

                    with col1:
                        if 0 in probability_values:
                            st.metric(
                                "No Diabetes",
                                f"{probability_values[0]:.2f}%"
                            )

                    with col2:
                        if 1 in probability_values:
                            st.metric(
                                "Diabetes",
                                f"{probability_values[1]:.2f}%"
                            )

                else:
                    st.info(
                        "This saved model does not provide class "
                        "probabilities, so a percentage cannot be shown."
                    )

            except Exception as error:
                st.error(
                    "Prediction failed. Check that your model and scaler "
                    "were trained with the same 21 features in the same "
                    f"order. Details: {error}"
                )


# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption("Diabetes Health Indicators Analysis and Prediction")

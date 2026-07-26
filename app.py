"""
Step 3: Streamlit Frontend Application
This is the main web application built using Streamlit.
It allows users to interactively:
1. Input health metrics to analyze patient risk using our trained Machine Learning model.
2. View historical dashboard trends for the week.
3. Access personalized dietary recommendations based on their health results.

To run this app, execute in your terminal:
streamlit run app.py
install numpy , pandas , matplotlib , scikit-learn , streamlit , seaborn
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
from datetime import datetime
import json
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
from vani_ai import ask_vani
import re
from report_vlm import analyze_report
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="SANJEEVANI - Your AI Health Companion",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

/* Hide Streamlit Menu */
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}

/* Main Theme */
.stApp {
    background-color: #f8fff8;
}

/* Title */
.main-title {
    font-size: 3rem;
    font-weight: 700;
    color: #166534;
    text-align: center;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #4b5563;
    font-size: 1.2rem;
    margin-bottom: 30px;
}

/* Cards */
.health-card {
    background: white;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border: 1px solid #dcfce7;
    margin-bottom: 20px;
}

/* Healthy Result */
.status-healthy {
    border-left: 8px solid #22c55e;
}

/* Risk Result */
.status-risk {
    border-left: 8px solid #ef4444;
}

/* Section Headers */
.section-title {
    font-size: 1.3rem;
    font-weight: 600;
    color: #14532d;
    margin-bottom: 10px;
}

/* Landing Page Button */
.stButton > button {
    background-color: #166534;
    color: white;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    background-color: #15803d;
    color: white;
}

/* Inputs */
.stTextInput input,
.stNumberInput input {
    border-radius: 10px;
}

/* Diet Boxes */
.diet-box {
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 12px;
}

.diet-include {
    background-color: #f0fdf4;
    border-left: 5px solid #22c55e;
}

.diet-avoid {
    background-color: #fef2f2;
    border-left: 5px solid #ef4444;
}

/* Landing Page Logo Section */
.logo-subtitle {
    text-align:center;
    color:#6b7280;
    font-size:1.2rem;
    margin-bottom:20px;
}

.center-text {
    text-align:center;
    font-size:1.1rem;
    color:#374151;
}
            
</style>
""", unsafe_allow_html=True)

# Helper function to load the trained model
@st.cache_resource
def load_model():
    model_path = "model/healthcare_model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as file:
                model_data = pickle.load(file)
            return model_data
        except Exception as e:
            st.error(f"Error loading model pickle: {e}")
            return None
    else:
        st.warning("Model file not found! Please run train_model.py first.")
        return None

# Load the model and features
model_data = load_model()

# =====================================================
# WATER RECOMMENDATION HELPER
# =====================================================

def get_water_recommendation(age, bmi):
    """Return a personalised daily water intake (in litres/day) based on BMI and age."""
    if bmi < 18.5:
        base_water = 2.2
    elif bmi < 25:
        base_water = 2.5
    elif bmi < 30:
        base_water = 3.0
    else:
        base_water = 3.5

    if age > 50:
        base_water += 0.3

    return round(base_water, 1)


# =====================================================
# DIET SUMMARY HELPER (used in the downloadable PDF report)
# =====================================================

def get_diet_summary_lines(glucose, blood_pressure, cholesterol, bmi):
    """Return short, plain-text diet recommendation bullet points."""
    glucose_bad = glucose > 130
    bp_bad = blood_pressure > 135
    chol_bad = cholesterol > 200
    bmi_bad = bmi > 27

    lines = []

    if glucose_bad:
        lines.append("Glucose Regulation Plan - low-GI, sugar-free meals with high-fibre lunches and dinners.")
    if bp_bad:
        lines.append("Sodium Restriction & Heart Health Plan - low-salt meals, potassium-rich fruits and vegetables.")
    if chol_bad:
        lines.append("Cholesterol Control Plan - healthy fats, oats, fish/tofu, and reduced fried food.")
    if bmi_bad:
        lines.append("Weight Management Plan - portion-controlled, high-protein meals with lighter dinners.")

    if not lines:
        lines.append("General Wellness Plan - all vitals in range; maintain a balanced diet of whole grains, lean protein, and fresh vegetables.")

    return lines


# =====================================================
# AI HEALTH REPORT PDF GENERATOR
# =====================================================

def _clean_for_pdf(text):
    """Strip/convert HTML from AI-generated text so it's safe inside a reportlab Paragraph.

    VANI's output (and report analysis output) contains raw <br> tags meant for
    st.markdown(unsafe_allow_html=True) on the web page. reportlab's Paragraph uses
    its own strict mini-XML parser, so those tags (and any stray & < >) need to be
    converted to plain line breaks / escaped before being passed in.
    """
    if not text:
        return ""
    # Convert common HTML line/paragraph breaks into real newlines
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*/?\s*p\s*>", "\n", text, flags=re.IGNORECASE)
    # Strip any other stray HTML tags (bold/italic/etc. from the source)
    text = re.sub(r"<[^>]+>", "", text)
    # Escape remaining XML-special characters so reportlab's parser doesn't choke
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text


def generate_health_report_pdf(patient_name, age, bmi, glucose, blood_pressure,
                                cholesterol, status_text, probability, vani_text,
                                report_text):
    """Builds the 'Complete AI Health Report' PDF and returns it as a BytesIO buffer."""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=32, bottomMargin=32, leftMargin=40, rightMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SJTitle", parent=styles["Title"],
        textColor=colors.HexColor("#166534"), fontSize=26,
        alignment=TA_CENTER, spaceAfter=2
    )
    sub_style = ParagraphStyle(
        "SJSub", parent=styles["Normal"],
        textColor=colors.HexColor("#4b5563"), fontSize=12,
        alignment=TA_CENTER, spaceAfter=18
    )
    heading_style = ParagraphStyle(
        "SJHeading", parent=styles["Heading2"],
        textColor=colors.HexColor("#14532d"), fontSize=13,
        spaceBefore=16, spaceAfter=6
    )
    body_style = ParagraphStyle(
        "SJBody", parent=styles["Normal"],
        fontSize=10.5, leading=15, textColor=colors.HexColor("#374151")
    )

    elements = []

    elements.append(Paragraph("SANJEEVANI", title_style))
    elements.append(Paragraph("AI HEALTH REPORT", sub_style))

    info_table = Table(
        [["Patient", _clean_for_pdf(patient_name) or "N/A"],
         ["Date", datetime.now().strftime("%d %B %Y")]],
        colWidths=[110, 350],
    )
    info_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#166534")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Risk Assessment", heading_style))
    elements.append(Paragraph(f"<b>Risk:</b> {_clean_for_pdf(status_text)}", body_style))
    elements.append(Paragraph(f"<b>Probability:</b> {probability:.1f}%", body_style))

    elements.append(Paragraph("Vitals", heading_style))
    vitals_table = Table(
        [["Age", str(age)],
         ["BMI", str(bmi)],
         ["Glucose", f"{glucose} mg/dL"],
         ["Blood Pressure", f"{blood_pressure} mmHg"],
         ["Cholesterol", f"{cholesterol} mg/dL"]],
        colWidths=[160, 300],
    )
    vitals_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dcfce7")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0fdf4")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(vitals_table)

    elements.append(Paragraph("AI Future Analysis (VANI Output)", heading_style))
    vani_raw = vani_text or ""
    marker = "Future Health Analysis"
    marker_idx = vani_raw.find(marker)
    if marker_idx != -1:
        # Drop VANI's typing-animation greeting; keep only the actual analysis.
        vani_raw = vani_raw[marker_idx:]
    vani_clean = _clean_for_pdf(vani_raw).strip()
    if vani_clean:
        for para in vani_clean.split("\n"):
            if para.strip():
                elements.append(Paragraph(para.strip(), body_style))
    else:
        elements.append(Paragraph(
            "Not generated yet - visit the Future Self page to unlock this section.",
            body_style
        ))

    elements.append(Paragraph("Diet Recommendations", heading_style))
    for line in get_diet_summary_lines(glucose, blood_pressure, cholesterol, bmi):
        elements.append(Paragraph(f"&bull; {_clean_for_pdf(line)}", body_style))

    elements.append(Paragraph("Daily Water Recommendation", heading_style))
    elements.append(Paragraph(
        f"{get_water_recommendation(age, bmi)} Litres / day", body_style
    ))

    elements.append(Paragraph("AI Report Analysis", heading_style))
    report_clean = _clean_for_pdf(report_text).strip()
    if report_clean:
        for para in report_clean.split("\n"):
            if para.strip():
                elements.append(Paragraph(para.strip(), body_style))
    else:
        elements.append(Paragraph(
            "No report was analysed in this session. Visit the Reports page to scan a document.",
            body_style
        ))

    elements.append(Paragraph("Disclaimer", heading_style))
    elements.append(Paragraph(
        "This report is generated by an AI system for informational purposes only and does not "
        "constitute medical advice, diagnosis, or treatment. Please consult a qualified healthcare "
        "provider for any medical concerns.",
        body_style
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# =====================================================
# BRAND HEADER (shown above the nav bar on every inner page except Home,
# which already has its own "Hello, {name}!" greeting in that spot)
# =====================================================

def render_page_brand_header():
    st.markdown("""
    <div style="text-align:center; margin-bottom:4px;">
        <h1 style="
            color:#166534;
            font-family:'Fraunces', serif;
            font-size:2.2rem;
            font-weight:700;
            margin-bottom:0px;
        ">
            🩺 SANJEEVANI
        </h1>
        <p style="
            color:#6b7280;
            font-size:1rem;
            margin-top:2px;
            margin-bottom:18px;
        ">
            Your AI Health Companion
        </p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# HORIZONTAL TOP NAVIGATION BAR (used across all inner pages)
# =====================================================

def render_top_nav():

    nav_items = [
        ("🏠", "Home", "home"),
        ("📈", "Weekly", "weekly"),
        ("🔮", "Future Self", "future_self"),
        ("🥗", "Diet Chart", "diet"),
        ("🔬", "Reports", "report_analyser"),
        ("ℹ️", "About", "about"),
        ("❓", "FAQs", "faq"),
    ]

    active_index = next(
        (i for i, (_, _, key) in enumerate(nav_items) if key == st.session_state.page),
        0
    )

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700;900&family=Inter:wght@500;600;700;800&display=swap');

    @keyframes navGradientShift {{
        0%   {{ background-position: 0% 50%; }}
        50%  {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    @keyframes activeGlowPulse {{
        0%, 100% {{
            box-shadow: 0 8px 20px rgba(0,0,0,0.3), 0 0 0 4px rgba(255,255,255,0.16), 0 0 18px rgba(74,222,128,0.55);
        }}
        50% {{
            box-shadow: 0 10px 26px rgba(0,0,0,0.34), 0 0 0 6px rgba(255,255,255,0.26), 0 0 34px rgba(74,222,128,0.95);
        }}
    }}

    .top-nav-wrap {{
        position: relative;
        overflow: hidden;
        background: linear-gradient(115deg, #0a2e19, #166534, #22c55e, #0f3d22, #166534);
        background-size: 300% 300%;
        animation: navGradientShift 12s ease infinite;
        border-radius: 26px;
        padding: 22px 24px;
        margin-bottom: 30px;
        border: 1px solid rgba(255,255,255,0.22);
        box-shadow:
            0 20px 45px rgba(15,61,34,0.45),
            0 4px 10px rgba(15,61,34,0.3),
            inset 0 1px 0 rgba(255,255,255,0.22);
    }}

    .top-nav-wrap::before {{
        content:"";
        position:absolute;
        top:-90px; left:-40px;
        width:220px; height:220px;
        background: radial-gradient(circle, rgba(255,255,255,0.30), transparent 70%);
        filter: blur(6px);
        pointer-events:none;
    }}
    .top-nav-wrap::after {{
        content:"";
        position:absolute;
        bottom:-110px; right:-30px;
        width:260px; height:260px;
        background: radial-gradient(circle, rgba(163,230,53,0.28), transparent 70%);
        filter: blur(8px);
        pointer-events:none;
    }}

    .top-nav-wrap div[data-testid="stHorizontalBlock"] {{
        gap: 10px;
        align-items: center;
        position: relative;
        z-index: 2;
    }}

    .top-nav-wrap .stButton > button {{
        font-family: 'Inter', sans-serif;
        background: rgba(255,255,255,0.10) !important;
        color: rgba(255,255,255,0.95) !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 999px !important;
        height: 58px;
        font-size: 15px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.22s cubic-bezier(.4,0,.2,1);
        box-shadow: none !important;
    }}

    .top-nav-wrap .stButton > button:hover {{
        background: rgba(255,255,255,0.26) !important;
        border-color: rgba(255,255,255,0.75) !important;
        transform: translateY(-3px) scale(1.03);
    }}

    .top-nav-wrap div[data-testid="stHorizontalBlock"] > div:nth-child({active_index + 1}) .stButton > button {{
        background: white !important;
        color: #14532d !important;
        border-color: white !important;
        font-weight: 800;
        transform: translateY(-2px) scale(1.05);
        animation: activeGlowPulse 2.2s ease-in-out infinite;
    }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="top-nav-wrap">', unsafe_allow_html=True)

    nav_cols = st.columns(len(nav_items))

    for nav_col, (icon, label, page_key) in zip(nav_cols, nav_items):
        with nav_col:
            if st.button(f"{icon} {label}", use_container_width=True, key=f"topnav_{page_key}"):
                st.session_state.page = page_key
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# SANJEEVANI NAVIGATION SYSTEM - PART 1
# =====================================================

if "page" not in st.session_state:
    st.session_state.page = "landing"

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "show_menu" not in st.session_state:
    st.session_state.show_menu = False

if "greeting_done" not in st.session_state:
    st.session_state.greeting_done = False

# Initialize form values in session state
if "form_age" not in st.session_state:
    st.session_state.form_age = 45

if "form_bmi" not in st.session_state:
    st.session_state.form_bmi = 25.0

if "form_glucose" not in st.session_state:
    st.session_state.form_glucose = 120.0

if "form_blood_pressure" not in st.session_state:
    st.session_state.form_blood_pressure = 120.0

if "form_skin_thickness" not in st.session_state:
    st.session_state.form_skin_thickness = 28.0

if "form_insulin" not in st.session_state:
    st.session_state.form_insulin = 150.0

if "form_pregnancies" not in st.session_state:
    st.session_state.form_pregnancies = 0

if "form_diabetes_pedigree" not in st.session_state:
    st.session_state.form_diabetes_pedigree = 0.5

if "form_cholesterol" not in st.session_state:
    st.session_state.form_cholesterol = 195.0

if "form_heart_rate" not in st.session_state:
    st.session_state.form_heart_rate = 72

if "show_report" not in st.session_state:
    st.session_state.show_report = False

if "vani_chat" not in st.session_state:
    st.session_state.vani_chat = ""

if "show_vani_report" not in st.session_state:
    st.session_state.show_vani_report = False

# =====================================================
# LANDING PAGE
# =====================================================

if st.session_state.page == "landing":

    st.markdown("<br><br>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns([2,2,1,2,2])

    with col3:
        st.image("logo2.jpg", use_container_width=True)

    st.markdown("""
    <h1 style="
        text-align:center;
        color:#166534;
        font-size:3rem;
        margin-bottom:0px;
    ">
    SANJEEVANI
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="
        text-align:center;
        color:#6b7280;
        font-size:22px;
        margin-top:0px;
    ">
    Your AI Health Companion
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <p style="
        text-align:center;
        font-size:22px;
        color:#374151;
    ">
    Ready to get your personalized health report?
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    left_btn, center_btn, right_btn = st.columns([1,2,1])

    with center_btn:
        if st.button(
            "🩺 Yes, I want my Health Report",
            use_container_width=True
        ):
            st.session_state.page = "home"
            st.rerun()
# =====================================================
# HOME PAGE
# =====================================================

elif st.session_state.page == "home":
    
    st.markdown("""
    <style>
    .stApp{
        background-color:#d8f3dc;
    }

    .welcome-card{
        background:white;
        padding:25px;
        border-radius:20px;
        box-shadow:0px 4px 12px rgba(0,0,0,0.1);
        text-align:center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-card">
        <h1 style="color:#166534;">
            Welcome to Sanjeevani
        </h1>
        <p style="font-size:18px;">
            Your AI-powered healthcare companion
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.user_name == "":

        name = st.text_input(
            "Enter your name"
        )

        if name.strip() != "":
            st.session_state.user_name = name
            st.session_state.greeting_done = False
            st.rerun()

    else:

        import time

        if not st.session_state.greeting_done:

            placeholder = st.empty()

            text = f"Hello, {st.session_state.user_name}!"

            current = ""

            for letter in text:

                current += letter

                placeholder.markdown(f"""
                <div style="
                    text-align:center;
                    padding:40px 20px;
                ">
                    <h1 style="
                        color:#166534;
                        font-size:3rem;
                    ">
                        {current}
                    </h1>
                </div>
                """, unsafe_allow_html=True)

                time.sleep(0.08)

            st.session_state.greeting_done = True

        else:

            st.markdown(f"""
            <div style="
                text-align:center;
                padding:40px 20px;
            ">
                <h1 style="
                    color:#166534;
                    font-size:3rem;
                ">
                    Hello, {st.session_state.user_name}! How are you doing?
                </h1>
            </div>
            """, unsafe_allow_html=True)

         # ================= TOP NAVIGATION BAR =================

        render_top_nav()

            # ================= HEALTH ASSESSMENT SECTION =================

        st.markdown("<br><br>", unsafe_allow_html=True)


        st.markdown("""
            <div class="welcome-card">

            <h1 style="
            color:#166534;
            text-align:center;
            ">
            🩺 AI Health Assessment
            </h1>

            <p style="
            text-align:center;
            font-size:18px;
            color:#374151;
            ">
            Provide your health details and let Sanjeevani analyze
            your health risk profile using Artificial Intelligence.
            </p>

            </div>
            """, unsafe_allow_html=True)


        st.markdown("<br>", unsafe_allow_html=True)



        if model_data is None:

                st.error(
                    "Please run the train_model.py script first to generate the ML model."
                )


        else:

                model = model_data['model']
                feature_names = model_data['features']


                st.markdown("""
                <div class="health-card">

                <h2 style="color:#14532d;">
                📋 Enter Health Information
                </h2>

                </div>
                """, unsafe_allow_html=True)


                with st.form("health_form"):


                    col1, col2 = st.columns(2)


                    # ---------------- LEFT COLUMN ----------------

                    with col1:

                        st.markdown("### 👤 Personal & Lifestyle Details")


                        age = st.number_input(
                        "Age (years)",
                        min_value=18,
                        max_value=90,
                        value=st.session_state.form_age
                    )


                        bmi = st.number_input(
                        "BMI",
                        min_value=10.0,
                        max_value=60.0,
                        step=0.1,
                        value=st.session_state.form_bmi
                    )


                        pregnancies = st.number_input(
                        "Pregnancies",
                        min_value=0,
                        max_value=15,
                        value=st.session_state.form_pregnancies
                    )


                        diabetes_pedigree = st.number_input(
                        "Diabetes Pedigree Function",
                        min_value=0.05,
                        max_value=2.5,
                        step=0.01,
                        value=st.session_state.form_diabetes_pedigree
                    )


                        heart_rate = st.number_input(
                        "Resting Heart Rate (bpm)",
                        min_value=40,
                        max_value=140,
                        value=st.session_state.form_heart_rate
                    )



                    # ---------------- RIGHT COLUMN ----------------

                    with col2:

                        st.markdown("### 🧪 Clinical Measurements")


                        glucose = st.number_input(
                        "Glucose Level (mg/dL)",
                        min_value=50.0,
                        max_value=300.0,
                        value=st.session_state.form_glucose
                    )


                        blood_pressure = st.number_input(
                        "Blood Pressure (mmHg)",
                        min_value=50.0,
                        max_value=250.0,
                        value=st.session_state.form_blood_pressure
                    )


                        cholesterol = st.number_input(
                        "Cholesterol Level (mg/dL)",
                        min_value=100.0,
                        max_value=400.0,
                        value=st.session_state.form_cholesterol
                    )


                        skin_thickness = st.number_input(
                        "Skin Thickness (mm)",
                        min_value=5.0,
                        max_value=60.0,
                        value=st.session_state.form_skin_thickness
                    )


                        insulin = st.number_input(
                        "Insulin Level (μU/mL)",
                        min_value=10.0,
                        max_value=500.0,
                        value=st.session_state.form_insulin
                    )



                    st.markdown("<br>", unsafe_allow_html=True)


                    col_btn1, col_btn2 = st.columns(2)

                    with col_btn1:
                        submit_button = st.form_submit_button(
                        "🩺 Analyze My Health",
                        use_container_width=True
                        )

                    with col_btn2:
                        reset_button = st.form_submit_button(
                        "🔄 Reset Values",
                        use_container_width=True
                        )

                    if reset_button:
                        # Reset all form values to defaults
                        st.session_state.form_age = 45
                        st.session_state.form_bmi = 25.0
                        st.session_state.form_glucose = 120.0
                        st.session_state.form_blood_pressure = 120.0
                        st.session_state.form_skin_thickness = 28.0
                        st.session_state.form_insulin = 150.0
                        st.session_state.form_pregnancies = 0
                        st.session_state.form_diabetes_pedigree = 0.5
                        st.session_state.form_cholesterol = 195.0
                        st.session_state.form_heart_rate = 72

                        st.session_state.vani_chat = ""
                        st.session_state.show_vani_report = False
                        
                        # Clear the report
                        st.session_state.show_report = False
                        if "latest_analysis" in st.session_state:
                            del st.session_state["latest_analysis"]
                        
                        

                        st.rerun()

                # ================= PREDICTION =================

                if submit_button:
                    # Save form values to session state
                    st.session_state.form_age = age
                    st.session_state.form_bmi = bmi
                    st.session_state.form_glucose = glucose
                    st.session_state.form_blood_pressure = blood_pressure
                    st.session_state.form_skin_thickness = skin_thickness
                    st.session_state.form_insulin = insulin
                    st.session_state.form_pregnancies = pregnancies
                    st.session_state.form_diabetes_pedigree = diabetes_pedigree
                    st.session_state.form_cholesterol = cholesterol
                    st.session_state.form_heart_rate = heart_rate

                    # A fresh analysis invalidates any previous Future Self report
                    st.session_state.vani_chat = ""
                    st.session_state.show_vani_report = False
                    
                    input_features = [
                        age,
                        bmi,
                        glucose,
                        blood_pressure,
                        skin_thickness,
                        insulin,
                        pregnancies,
                        diabetes_pedigree,
                        cholesterol,
                        heart_rate
                    ]


                    input_df = pd.DataFrame(
                        [input_features],
                        columns=feature_names
                    )


                    prediction = model.predict(input_df)[0]
                    probability = model.predict_proba(input_df)[0][1]


                    st.session_state['latest_analysis'] = {
                        'age': age,
                        'bmi': bmi,
                        'glucose': glucose,
                        'blood_pressure': blood_pressure,
                        'cholesterol': cholesterol,
                        'risk_level': int(prediction),
                        'risk_percentage': float(probability * 100)
                    }
                    st.session_state.show_report = True

                    timestamp = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )


                    new_record = pd.DataFrame([{
                        'timestamp': timestamp,
                        'age': age,
                        'bmi': bmi,
                        'glucose': glucose,
                        'blood_pressure': blood_pressure,
                        'skin_thickness': skin_thickness,
                        'insulin': insulin,
                        'pregnancies': pregnancies,
                        'diabetes_pedigree': diabetes_pedigree,
                        'cholesterol': cholesterol,
                        'heart_rate': heart_rate,
                        'risk_level': int(prediction),
                        'risk_percentage': round(
                            float(probability * 100), 1
                        )
                    }])


                    import os

                    file_exists = os.path.exists("history.csv")

                    new_record.to_csv(
                        "history.csv",
                        mode="a",
                        header=not file_exists,
                        index=False
                    )


                    import time

                    with st.spinner("🧠 Sanjeevani AI is analyzing your health profile..."):
                        time.sleep(2)

                # Display report if it exists
                if st.session_state.show_report and 'latest_analysis' in st.session_state:
                    analysis = st.session_state["latest_analysis"]
                    age = analysis["age"]
                    bmi = analysis["bmi"]
                    glucose = analysis["glucose"]
                    blood_pressure = analysis["blood_pressure"]
                    cholesterol = analysis["cholesterol"]
                    prediction = analysis["risk_level"]
                    probability = analysis["risk_percentage"] / 100

                    risk_score = probability * 100
                    health_score = 100 - risk_score

                    # ==========================================
                    # HEALTH CATEGORIES
                    # ==========================================

                    if risk_score <= 20:
                        status = "🏆 Excellent Health"
                        badge_color = "#16a34a"
                        glow = "#22c55e"

                    elif risk_score <= 40:
                        status = "✅ Healthy"
                        badge_color = "#22c55e"
                        glow = "#4ade80"

                    elif risk_score <= 60:
                        status = "⚠️ Moderate Risk"
                        badge_color = "#eab308"
                        glow = "#facc15"

                    elif risk_score <= 80:
                        status = "🚨 High Risk"
                        badge_color = "#f97316"
                        glow = "#fb923c"

                    else:
                        status = "🔴 Critical Risk"
                        badge_color = "#dc2626"
                        glow = "#ef4444"


                    # ==========================================
                    # CONFETTI
                    # ==========================================

                    from streamlit_lottie import st_lottie
                    import requests

                    def load_lottie(url):
                        r = requests.get(url)
                        return r.json()
                    if risk_score <= 20:
                        
                        try:

                           with open(
                           "result page success motion design.json",
                           "r",
                           encoding="utf-8"
                           ) as f:

                                confetti = json.load(f)

                           st_lottie(
                            confetti,
                            height=250,
                            key="health_confetti"
                            )

                        except Exception as e:

                            st.error(f"Animation Error: {e}")
                    

                    # ==========================================
                    # HEARTBEAT + GLOW CARD
                    # ==========================================

                    st.markdown(f"""
                    <style>

                    @keyframes heartbeat {{
                        0% {{transform: scale(1);}}
                        25% {{transform: scale(1.15);}}
                        50% {{transform: scale(1);}}
                        75% {{transform: scale(1.15);}}
                        100% {{transform: scale(1);}}
                    }}

                    .heartbeat {{
                        text-align:center;
                        font-size:70px;
                        animation: heartbeat 1s infinite;
                    }}

                    .glow-card {{
                        background:white;
                        border-radius:25px;
                        padding:35px;
                        text-align:center;
                        box-shadow:0px 0px 35px {glow};
                        border:3px solid {badge_color};
                    }}

                    </style>

                    <div class="heartbeat">
                    ❤️
                    </div>

                    <div class="glow-card">

                    <h3 style="color:{badge_color};">
                    Health Report for {st.session_state.user_name}
                    </h3>

                    <h1 style="
                    color:{badge_color};
                    font-size:3rem;
                    ">
                    {status}
                    </h1>

                    <h2>
                    Risk Score: {risk_score:.1f}%
                    </h2>

                    <h2>
                    Health Score: {health_score:.1f}/100
                    </h2>

                    </div>
                    """, unsafe_allow_html=True)


                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown("### 📊 Risk Meter")
                    st.progress(int(risk_score))


                    # ==========================================
                    # AI INSIGHTS
                    # ==========================================

                    st.markdown("### 🧠 AI Insights")

                    insights = []

                    if glucose > 130:
                        insights.append("• Glucose level is above the healthy range.")

                    if blood_pressure > 135:
                        insights.append("• Blood pressure appears elevated.")

                    if cholesterol > 200:
                        insights.append("• Cholesterol requires monitoring.")

                    if bmi > 27:
                        insights.append("• BMI indicates excess body weight.")

                    if len(insights) == 0:
                        insights.append("• All major health indicators appear within healthy limits.")

                    for item in insights:
                        st.write(item)

                    st.markdown("### 🔍 Health Metrics Overview")


                    m1, m2, m3, m4 = st.columns(4)


                    with m1:
                        st.metric(
                            "Glucose",
                            f"{glucose} mg/dL"
                                            )

                    with m2:
                        st.metric(
                            "Blood Pressure",
                            f"{blood_pressure} mmHg"
                                            )

                    with m3:
                        st.metric(
                            "Cholesterol",
                            f"{cholesterol} mg/dL"
                                            )

                    with m4:
                        st.metric(
                            "BMI",
                            f"{bmi}"
                                            )

                    # ==========================================
                    # DOWNLOAD COMPLETE AI HEALTH REPORT
                    # ==========================================

                    st.markdown("<br>", unsafe_allow_html=True)

                    status_plain = re.sub(r"[^\x00-\x7F]+", "", status).strip()

                    pdf_buffer = generate_health_report_pdf(
                        patient_name=st.session_state.user_name,
                        age=age,
                        bmi=bmi,
                        glucose=glucose,
                        blood_pressure=blood_pressure,
                        cholesterol=cholesterol,
                        status_text=status_plain,
                        probability=risk_score,
                        vani_text=st.session_state.get("vani_chat", ""),
                        report_text=st.session_state.get("report_analysis_result", ""),
                    )

                    dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
                    with dl_col2:
                        st.download_button(
                            label="⬇ Download My Complete Health Report",
                            data=pdf_buffer,
                            file_name=f"Health_Report_{(st.session_state.user_name or 'Patient').replace(' ', '_')}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )

# ----------------- PAGE 2: WEEKLY ANALYSIS GRAPH -----------------
elif st.session_state.page == "weekly":
        render_page_brand_header()
        render_top_nav()

        st.markdown("""
    <style>

    .dashboard-header{
        background:linear-gradient(135deg,#166534,#22c55e);
        padding:30px;
        border-radius:25px;
        text-align:center;
        color:white;
        box-shadow:0px 10px 30px rgba(0,0,0,0.15);
        margin-bottom:20px;
    }

    .glass-card{
        background:rgba(255,255,255,0.75);
        backdrop-filter:blur(12px);
        padding:20px;
        border-radius:20px;
        box-shadow:0px 6px 20px rgba(0,0,0,0.1);
    }

    </style>
    """, unsafe_allow_html=True)

        st.markdown(f"""
    <div class="dashboard-header">

    <h1>
    Health Analytics Center
    </h1>

    <h3>
    Welcome Back, {st.session_state.user_name}
    </h3>

    <p>
    Real-time health monitoring and risk intelligence dashboard
    </p>

    </div>
    """, unsafe_allow_html=True)
        if not os.path.exists("history.csv"):

            st.warning(
                "No assessment history found. Analyze a health profile first."
            )
            st.stop()

        else:

            history_df = pd.read_csv("history.csv")
            
            if history_df.empty:
                st.info(
                    "No health assessments available yet. Complete your first analysis to generate your weekly dashboard."
                    )
                st.stop()

            history_df["timestamp"] = pd.to_datetime(
                history_df["timestamp"]
            )

            history_df["date"] = (
                history_df["timestamp"]
                .dt.strftime("%d %b")
            )

            total_checks = len(history_df)

            high_risk_count = len(
                history_df[
                    history_df["risk_level"] == 1
                ]
            )

            healthy_count = len(
                history_df[
                    history_df["risk_level"] == 0
                ]
            )

            avg_risk_prob = (
                history_df["risk_percentage"]
                .mean()
            )

            latest = history_df.iloc[-1]

            latest_risk = latest["risk_percentage"]
            health_score = 100 - latest_risk

            st.markdown("## 📊 Weekly Performance Summary")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                st.metric(
                    "🩺 Assessments",
                    total_checks
                )

            with c2:
                st.metric(
                    "✅ Healthy Cases",
                    healthy_count
                )

            with c3:
                st.metric(
                    "⚠️ At Risk",
                    high_risk_count
                )

            with c4:
                st.metric(
                    "❤️ Health Score",
                    f"{health_score:.1f}/100"
                )

            st.markdown("<br>", unsafe_allow_html=True)
            gauge_col1, gauge_col2 = st.columns([2,1])
            latest = history_df.iloc[-1]

            latest_risk = latest["risk_percentage"]
            health_score = 100 - latest_risk
            with gauge_col1:

                gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",

                        value=health_score,

                        title={
                            "text":"Overall Health Score"
                        },

                        gauge={
                            "axis":{
                                "range":[0,100]
                            },

                            "bar":{
                                "color":"green"
                            },

                            "steps":[
                                {
                                    "range":[0,40],
                                    "color":"#ef4444"
                                },

                                {
                                    "range":[40,70],
                                    "color":"#f59e0b"
                                },

                                {
                                    "range":[70,100],
                                    "color":"#22c55e"
                                }
                            ]
                        }
                    )
                )

                gauge.update_layout(
                    height=350
                )

                st.plotly_chart(
                    gauge,
                    use_container_width=True
                )
            with gauge_col2:

                st.markdown("### 🧠 AI Insights")

                if latest_risk < 20:

                    st.success(
                        "Excellent overall health trend detected this week."
                    )

                elif latest_risk < 60:

                    st.warning(
                        "Moderate health risk trend detected."
                    )

                else:

                    st.error(
                        "High health risk trend detected."
                    )

                st.metric(
                    "Current Risk",
                    f"{latest_risk:.1f}%"
                    )

                st.metric(
                    "Healthy Ratio",
                    f"{healthy_count}/{total_checks}"
                )

            st.markdown("---")

            pie_col1, pie_col2 = st.columns([2,1])

            with pie_col1:

                pie_fig = px.pie(
                    values=[
                        healthy_count,
                        high_risk_count
                    ],

                    names=[
                        "Healthy",
                        "At Risk"
                    ],

                    hole=0.65
                )

                pie_fig.update_layout(
                    title="All Recorded Assessments",
                    height=450
                )

                st.plotly_chart(
                    pie_fig,
                    use_container_width=True
                )

                st.caption(
                "Historical distribution of all analyzed patients recorded in the Sanjeevani system."
                )
            with pie_col2:

                st.markdown("### 🏆 Weekly Rating")

                if health_score >= 90:

                    badge = "🥇 Elite Health"

                elif health_score >= 75:

                    badge = "🥈 Excellent"

                elif health_score >= 60:

                    badge = "🥉 Good"

                elif health_score >= 40:

                    badge = "⚠️ Needs Attention"

                else:

                    badge = "🚨 Critical"

                st.markdown(f"""
                <div style="
                background:white;
                padding:30px;
                border-radius:20px;
                text-align:center;
                box-shadow:0px 6px 20px rgba(0,0,0,0.1);
                ">

                <h2>{badge}</h2>

                <h1 style="
                color:#166534;
                ">
                {health_score:.1f}
                </h1>

                <p>
                Weekly Health +
                </p>

                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("## 🔬 Vital Status Dashboard")

            v1, v2, v3, v4 = st.columns(4)

            with v1:
                if latest["glucose"] <= 140:
                    st.success("🟢 Glucose\n\nNormal")
                else:
                    st.error("🔴 Glucose\n\nHigh")

            with v2:
                if latest["blood_pressure"] <= 130:
                    st.success("🟢 Blood Pressure\n\nNormal")
                else:
                    st.error("🔴 Blood Pressure\n\nHigh")

            with v3:
                if latest["cholesterol"] <= 200:
                    st.success("🟢 Cholesterol\n\nNormal")
                else:
                    st.error("🔴 Cholesterol\n\nHigh")

            with v4:
                if latest["bmi"] <= 25:
                    st.success("🟢 BMI\n\nHealthy")
                else:
                    st.warning("🟡 BMI\n\nElevated")

            st.markdown("---")
            st.markdown("## 🎯 Health Breakdown")

            score_glucose = max(0, 100 - latest["glucose"]/2)
            score_bp = max(0, 100 - latest["blood_pressure"]/2)
            score_chol = max(0, 100 - latest["cholesterol"]/3)
            score_bmi = max(0, 100 - latest["bmi"]*2)

            breakdown = pd.DataFrame({
                "Metric":["Glucose","Blood Pressure","Cholesterol","BMI"],
                "Score":[score_glucose, score_bp, score_chol, score_bmi]
            })

            fig = px.bar(
                breakdown,
                x="Metric",
                y="Score",
                title="Health Component Scores"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("## 🧠 Sanjeevani Summary")

            summary = []

            if latest["glucose"] > 140:
                summary.append("Elevated glucose detected.")

            if latest["blood_pressure"] > 130:
                summary.append("Blood pressure requires monitoring.")

            if latest["cholesterol"] > 200:
                summary.append("Cholesterol level is above ideal range.")

            if latest["bmi"] > 25:
                summary.append("Weight management may improve health outcomes.")

            if len(summary) == 0:
                summary.append("All major health indicators are within healthy ranges.")

            for item in summary:
                st.write("•", item)

# ==========================================
# AI FUTURE SELF PAGE
# ==========================================

elif st.session_state.page == "future_self":

    render_page_brand_header()
    render_top_nav()

    st.markdown("""
    <style>

    .future-header{
        background:linear-gradient(135deg,#166534,#22c55e);
        padding:30px;
        border-radius:25px;
        text-align:center;
        color:white;
        box-shadow:0px 10px 30px rgba(0,0,0,0.15);
        margin-bottom:20px;
    }

    .future-card{
        background:rgba(255,255,255,0.85);
        backdrop-filter:blur(15px);
        padding:25px;
        border-radius:20px;
        box-shadow:0px 6px 20px rgba(0,0,0,0.1);
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="future-header">

    <h1>
    AI Future Self
    </h1>

    <h3>
    Meet the Future Version of Yourself
    </h3>

    <p>
    Powered by AI-driven health forecasting and predictive wellness insights
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <h4 style="
    text-align:center;
    color:#166534;
    letter-spacing:3px;
    ">
    SCANNING HEALTH TIMELINE
    </h4>
    """, unsafe_allow_html=True)

    with open(
        "Looping Energy Orb.json",
        "r",
        encoding="utf-8"
    ) as f:

        future_orb = json.load(f)

    st_lottie(
        future_orb,
        height=320,
        key="future_orb"
    )

    st.markdown("""
    <h5 style="
    text-align:center;
    color:#6b7280;
    letter-spacing:2px;
    ">
    PROJECTING FUTURE OUTCOMES
    </h5>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="future-card">

    <h2 style="
    color:#166534;
    text-align:center;
    ">
    🌌 Future Health Projection Engine
    </h2>

    <p style="
    text-align:center;
    font-size:18px;
    color:#4b5563;
    ">

    Analyze your latest health assessment and
    discover what your future self may look like
    1, 3 and 5 years from now.

    </p>

    </div>
    """, unsafe_allow_html=True)

    #------VANI--------

    # Gate the entire Future Self / VANI experience behind an existing analysis.
    # Nothing below (avatar, chat box, button) renders until a report exists.
    if "latest_analysis" not in st.session_state:

        st.warning(
            "Please generate a health analysis first."
        )

    else:

        # ---- CASE 1: Report already generated earlier -> just redisplay it ----
        if st.session_state.show_vani_report and st.session_state.vani_chat:

            left_col, right_col = st.columns([2, 1])

            with right_col:

                with open(
                    "Female avatar.json",
                    "r",
                    encoding="utf-8"
                ) as f:

                    vani_avatar = json.load(f)

                st_lottie(
                    vani_avatar,
                    height=350,
                    key="vani_avatar_static"
                )

            with left_col:

                st.markdown(f"""
                <div style="
                background:rgba(255,255,255,0.95);
                padding:35px;
                border-radius:25px;
                box-shadow:0px 6px 25px rgba(0,0,0,0.12);
                font-size:15px;
                line-height:1.8;
                color:#166534;
                font-family:'Trebuchet MS', sans-serif;
                min-height:250px;
                ">
                {st.session_state.vani_chat}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button(
                "🥗 See My Diet Chart Next",
                use_container_width=True
            ):
                st.session_state.page = "diet"
                st.rerun()

        # ---- CASE 2: No report generated yet -> show the trigger button ----
        else:

            generate_future = st.button(
                "🔮 Generate My Future Self",
                use_container_width=True
            )

            if generate_future:

                left_col, right_col = st.columns([2, 1])

                with right_col:

                    with open(
                        "Female avatar.json",
                        "r",
                        encoding="utf-8"
                    ) as f:

                        vani_avatar = json.load(f)

                    st_lottie(
                        vani_avatar,
                        height=350,
                        key="vani_avatar_live"
                    )

                with left_col:

                    typing_box = st.empty()

                latest = st.session_state["latest_analysis"]

                # Play VANI's intro typing animation FIRST, instantly on click,
                # before we ever touch the (slow) AI API call.
                messages = [

                    "INITIALIZING FUTURE HEALTH ENGINE...",

                    "Meet VANI",

                    "Virtual Assistant for Nutritional & Intelligence Insights",

                    f"Hi {st.session_state.user_name}! 💚",

                    "I'm VANI.",

                    "I've received your latest health assessment.",

                    "Let's look into your future health journey..."
                ]

                full_conversation = ""

                for msg in messages:

                    current = ""

                    for char in msg:

                        current += char

                        typing_box.markdown(f"""
                        <div style="
                        background:rgba(255,255,255,0.95);
                        padding:35px;
                        border-radius:25px;
                        box-shadow:0px 6px 25px rgba(0,0,0,0.12);
                        font-size:15px;
                        line-height:1.8;
                        color:#166534;
                        font-family:'Trebuchet MS', sans-serif;
                        min-height:250px;
                        ">
                        {full_conversation}<br><b>{current}</b>
                        </div>
                        """, unsafe_allow_html=True)

                        time.sleep(0.025)

                    full_conversation += f"<p>{msg}</p>"

                    st.session_state.vani_chat = full_conversation

                    typing_box.markdown(f"""
                    <div style="
                    background:rgba(255,255,255,0.95);
                    padding:35px;
                    border-radius:25px;
                    box-shadow:0px 6px 25px rgba(0,0,0,0.12);
                    font-size:15px;
                    line-height:1.8;
                    color:#166534;
                    font-family:'Trebuchet MS', sans-serif;
                    min-height:250px;
                    ">
                    {full_conversation}
                    </div>
                    """, unsafe_allow_html=True)

                    time.sleep(0.8)

                # Now that the greeting has played, show a "please wait" state
                # while the actual AI report is being generated in the background.
                typing_box.markdown(f"""
                <div style="
                background:rgba(255,255,255,0.95);
                padding:35px;
                border-radius:25px;
                box-shadow:0px 6px 25px rgba(0,0,0,0.12);
                font-size:15px;
                line-height:1.8;
                color:#166534;
                font-family:'Trebuchet MS', sans-serif;
                min-height:250px;
                ">
                {full_conversation}
                <p><i>⏳ Please wait a few seconds while VANI analyzes your future health metrics...</i></p>
                </div>
                """, unsafe_allow_html=True)

                prompt = f"""
                Health Assessment Data

                Age: {latest['age']}
                BMI: {latest['bmi']}
                Glucose: {latest['glucose']}
                Blood Pressure: {latest['blood_pressure']}
                Cholesterol: {latest['cholesterol']}
                Risk Percentage: {latest['risk_percentage']:.1f}
                Risk Level: {latest['risk_level']}

                Generate a friendly future health projection.

                Base your analysis on these values.

                Mention specific improvements or concerns that are relevant to the user's data.

                Focus on healthy habits and possible improvements.

                Do not make medical diagnoses.

                Do not make alarming predictions.

                Keep the tone positive and motivating.

                Write only in English.
                """

                report = ask_vani(prompt)

                report = re.sub(
                    r'[^\x00-\x7F]+',
                    ' ',
                    report
                )

                formatted_report = report.replace("\n", "<br>")

                full_conversation += f"""
                <hr>

                <h3 style='color:#166534;'>
                Future Health Analysis
                </h3>

                <p>
                {formatted_report}
                </p>
                """

                st.session_state.vani_chat = full_conversation
                st.session_state.show_vani_report = True

                st.rerun()




# ----------------- PAGE 3: DIET CHART -----------------
elif st.session_state.page == "diet":

    render_page_brand_header()
    render_top_nav()

    # ---------- Design tokens / fonts for this page ----------
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    :root{
        --sj-ink:#14532d;
        --sj-green:#166534;
        --sj-mint:#22c55e;
        --sj-pale:#f0fdf4;
        --sj-cream:#fefaf3;
        --sj-line:#e7dfca;
        --sj-muted:#5c6b60;
        --sj-red:#b91c1c;
        --sj-red-bg:#fef2f2;
        --sj-red-line:#fecaca;
        --sj-amber:#92400e;
        --sj-amber-bg:#fffbeb;
        --sj-amber-line:#fde68a;
    }

    .sj-page-title{
        font-family:'Fraunces',serif;
        font-weight:700;
        font-size:2.4rem;
        color:var(--sj-ink);
        text-align:center;
        margin-bottom:2px;
    }
    .sj-page-sub{
        font-family:'Inter',sans-serif;
        text-align:center;
        color:var(--sj-muted);
        font-size:1.02rem;
        margin-bottom:22px;
    }

    /* ---- Prescription-style summary card ---- */
    .rx-card{
        background:var(--sj-cream);
        border:1px solid var(--sj-line);
        border-left:6px dashed var(--sj-mint);
        border-radius:16px;
        padding:26px 30px;
        box-shadow:0 8px 24px rgba(22,101,52,0.08);
        margin-bottom:10px;
    }
    .rx-eyebrow{
        font-family:'DM Mono',monospace;
        font-size:.72rem;
        letter-spacing:.12em;
        text-transform:uppercase;
        color:var(--sj-mint);
        font-weight:600;
    }
    .rx-title{
        font-family:'Fraunces',serif;
        font-size:1.5rem;
        font-weight:600;
        color:var(--sj-ink);
        margin:2px 0 16px 0;
    }
    .vital-pill{
        display:inline-flex;
        align-items:center;
        gap:6px;
        padding:8px 14px;
        border-radius:999px;
        font-family:'DM Mono',monospace;
        font-size:.82rem;
        font-weight:600;
        margin:4px 8px 4px 0;
        border:1px solid transparent;
    }
    .vital-in{background:#eafcef;color:var(--sj-green);border-color:#bbf0cd;}
    .vital-out{background:var(--sj-red-bg);color:var(--sj-red);border-color:var(--sj-red-line);}
    .vital-warn{background:var(--sj-amber-bg);color:var(--sj-amber);border-color:var(--sj-amber-line);}

    /* ---- Active condition chips ---- */
    .condition-row{
        display:flex;
        flex-wrap:wrap;
        gap:10px;
        margin:20px 0 4px 0;
    }
    .condition-chip{
        padding:9px 18px;
        border-radius:999px;
        font-size:.85rem;
        font-weight:600;
        font-family:'Inter',sans-serif;
        white-space:nowrap;
    }
    .chip-active{background:var(--sj-ink);color:white;}
    .chip-inactive{background:#eef2ef;color:#a2ada5;}

    /* ---- Meal plan cards ---- */
    .plan-card{
        background:white;
        border-radius:20px;
        margin:24px 0;
        box-shadow:0 6px 20px rgba(0,0,0,.07);
        overflow:hidden;
        border:1px solid #e5f3ea;
    }
    .plan-head{
        background:linear-gradient(120deg,var(--sj-ink),var(--sj-mint));
        padding:18px 28px;
        color:white;
    }
    .plan-head .plan-icon{font-size:1.5rem;margin-right:10px;}
    .plan-head h3{
        font-family:'Fraunces',serif;
        font-size:1.28rem;
        font-weight:600;
        margin:0;
        display:inline;
    }
    .plan-head p{
        font-family:'Inter',sans-serif;
        font-size:.85rem;
        margin:6px 0 0 0;
        opacity:.9;
    }

    .timeline{
        display:grid;
        grid-template-columns:1fr 1fr 1fr;
        align-items:start;
        padding:24px 28px 2px 28px;
        position:relative;
    }
    .timeline::before{
        content:"";
        position:absolute;
        top:36px;
        left:14%;
        right:14%;
        height:2px;
        background:repeating-linear-gradient(to right,#cfe8d6 0 8px,transparent 8px 14px);
    }
    .tl-node{text-align:center;position:relative;z-index:1;}
    .tl-icon{
        width:42px;height:42px;
        border-radius:50%;
        background:var(--sj-pale);
        border:2px solid var(--sj-mint);
        display:flex;align-items:center;justify-content:center;
        font-size:19px;
        margin:0 auto 6px auto;
    }
    .tl-label{
        font-family:'DM Mono',monospace;
        font-size:.68rem;
        letter-spacing:.08em;
        text-transform:uppercase;
        color:var(--sj-muted);
    }

    .meal-grid{
        display:grid;
        grid-template-columns:1fr 1fr 1fr;
        gap:16px;
        padding:12px 28px 28px 28px;
    }
    .food-chip{
        background:var(--sj-pale);
        border-radius:12px;
        padding:12px 14px;
        margin-bottom:10px;
    }
    .food-name{
        font-family:'Inter',sans-serif;
        font-weight:700;
        color:var(--sj-ink);
        font-size:.9rem;
    }
    .food-desc{
        font-family:'Inter',sans-serif;
        color:var(--sj-muted);
        font-size:.8rem;
        margin-top:2px;
        line-height:1.4;
    }

    .sj-note{
        font-family:'Inter',sans-serif;
        text-align:center;
        color:var(--sj-muted);
        font-size:.95rem;
        margin:6px 0 20px 0;
    }

    @media (max-width:700px){
        .timeline, .meal-grid{ grid-template-columns:1fr; }
        .timeline::before{ display:none; }
        .tl-node{ margin-bottom:14px; }
    }

    /* ---- Water Recommendation card ---- */
    .water-card{
        background:linear-gradient(135deg,#eff9ff,#f0fdf4);
        border:1px solid #bae6fd;
        border-radius:20px;
        padding:28px 30px;
        margin:28px 0 20px 0;
        box-shadow:0 8px 24px rgba(14,116,144,0.10);
    }
    .water-card h3{
        font-family:'Fraunces',serif;
        font-size:1.3rem;
        font-weight:600;
        color:#0c4a6e;
        margin:0 0 14px 0;
    }
    .water-amount-row{
        display:flex;
        align-items:baseline;
        gap:16px;
        flex-wrap:wrap;
        margin-bottom:14px;
    }
    .water-amount{
        font-family:'Fraunces',serif;
        font-size:2.6rem;
        font-weight:700;
        color:#0369a1;
        line-height:1;
    }
    .water-amount-label{
        font-family:'Inter',sans-serif;
        font-size:.85rem;
        color:var(--sj-muted);
    }
    .water-basis-row{
        display:flex;
        flex-wrap:wrap;
        gap:10px;
        margin-bottom:4px;
    }
    .water-basis-chip{
        background:white;
        border:1px solid #bae6fd;
        border-radius:999px;
        padding:6px 14px;
        font-family:'Inter',sans-serif;
        font-size:.8rem;
        font-weight:600;
        color:#0369a1;
    }
    .water-tips{
        background:white;
        border-radius:16px;
        padding:18px 22px;
        margin-top:16px;
        border:1px solid #dcfce7;
    }
    .water-tips h4{
        font-family:'Inter',sans-serif;
        font-size:.95rem;
        font-weight:700;
        color:var(--sj-ink);
        margin:0 0 8px 0;
    }
    .water-tips ul{
        margin:0;
        padding-left:18px;
        color:var(--sj-muted);
        font-family:'Inter',sans-serif;
        font-size:.88rem;
        line-height:1.7;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sj-page-title">🥗 Tailored Diet &amp; Nutrition Guidance</div>', unsafe_allow_html=True)

    # ---------- helper builders ----------

    def vital_pill(label, value, unit, bad):
        cls = "vital-out" if bad else "vital-in"
        icon = "⚠" if bad else "✓"
        return f'<span class="vital-pill {cls}">{icon} {label}: {value}{unit}</span>'

    def food_chip(name, desc):
        return f"""<div class="food-chip">
            <div class="food-name">{name}</div>
            <div class="food-desc">{desc}</div>
        </div>"""

    def render_plan(icon, title, subtitle, breakfast, lunch, dinner):
        b_html = "".join(food_chip(n, d) for n, d in breakfast)
        l_html = "".join(food_chip(n, d) for n, d in lunch)
        d_html = "".join(food_chip(n, d) for n, d in dinner)

        html = f"""
        <div class="plan-card">
            <div class="plan-head">
                <span class="plan-icon">{icon}</span><h3>{title}</h3>
                <p>{subtitle}</p>
            </div>
            <div class="timeline">
                <div class="tl-node"><div class="tl-icon">🌅</div><div class="tl-label">Breakfast</div></div>
                <div class="tl-node"><div class="tl-icon">☀️</div><div class="tl-label">Lunch</div></div>
                <div class="tl-node"><div class="tl-icon">🌙</div><div class="tl-label">Dinner</div></div>
            </div>
            <div class="meal-grid">
                <div>{b_html}</div>
                <div>{l_html}</div>
                <div>{d_html}</div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    # ---------- main content ----------

    if 'latest_analysis' in st.session_state:

        analysis = st.session_state['latest_analysis']

        glucose = analysis['glucose']
        bp = analysis['blood_pressure']
        chol = analysis['cholesterol']
        bmi = analysis['bmi']

        glucose_bad = glucose > 130
        bp_bad = bp > 135
        chol_bad = chol > 200
        bmi_bad = bmi > 27
        bmi_warn = bmi > 25

        bmi_status = "Overweight/Obese" if bmi > 25 else "Healthy Range"

        # ---- Rx style summary card ----
        pills = (
            vital_pill("Glucose", glucose, " mg/dL", glucose_bad)
            + vital_pill("Blood Pressure", bp, " mmHg", bp_bad)
            + vital_pill("Cholesterol", chol, " mg/dL", chol_bad)
            + f'<span class="vital-pill {"vital-warn" if bmi_warn else "vital-in"}">{"⚠" if bmi_warn else "✓"} BMI: {bmi} ({bmi_status})</span>'
        )

        st.markdown(f"""
        <div class="rx-card">
            <div class="rx-eyebrow">Sanjeevani AI &middot; Health Profile Summary</div>
            <div class="rx-title">Prescribed for {st.session_state.user_name or "you"}</div>
            {pills}
        </div>
        """, unsafe_allow_html=True)

        # ---- Active condition chips ----
        chips = [
            ("🩸 Glucose Care", glucose_bad),
            ("❤️ Heart &amp; BP", bp_bad),
            ("🫀 Cholesterol", chol_bad),
            ("⚖️ Weight Management", bmi_bad),
        ]
        chip_html = "".join(
            f'<span class="condition-chip {"chip-active" if active else "chip-inactive"}">{label}</span>'
            for label, active in chips
        )
        st.markdown(f'<div class="condition-row">{chip_html}</div>', unsafe_allow_html=True)
        st.markdown('<p class="sj-note">Your plan below adapts automatically to the readings above — it stays saved here until you reset your values on the Home page.</p>', unsafe_allow_html=True)

        any_active = glucose_bad or bp_bad or chol_bad or bmi_bad

        if glucose_bad:
            render_plan(
                "🩸", "Glucose Regulation Plan", "For your elevated glucose reading",
                breakfast=[
                    ("Sugar-free Oatmeal", "Topped with chia seeds and raw almonds."),
                    ("Veggie Omelette", "2 egg whites with spinach and mushrooms."),
                    ("Greek Yogurt", "Unsweetened, with a handful of fresh blueberries."),
                ],
                lunch=[
                    ("Grilled Chicken/Paneer Salad", "On a bed of spinach, cucumber, and cherry tomatoes."),
                    ("Quinoa Bowl", "With broccoli, bell peppers, and boiled chickpeas."),
                    ("Lentil Salad", "Sprouted lentils tossed with onion, cucumber, and lemon."),
                ],
                dinner=[
                    ("Baked Fish/Tofu", "With sautéed cauliflower, asparagus, and green beans."),
                    ("Lentil Soup (Dal)", "A bowl of yellow/black dal with stir-fried zucchini."),
                    ("Stir-fried Greens", "Cabbage, bell peppers, and paneer in light olive oil."),
                ],
            )

        if bp_bad:
            render_plan(
                "❤️", "Sodium Restriction &amp; Heart Health Plan", "For your elevated blood pressure",
                breakfast=[
                    ("Banana Oatmeal", "Oats cooked in low-fat milk, topped with banana."),
                    ("Avocado Toast", "Whole-wheat toast with mashed avocado, no added salt."),
                    ("Green Smoothie Bowl", "Spinach, banana, almond milk, unsalted seeds."),
                ],
                lunch=[
                    ("Mixed Bean Salad", "Kidney beans, chickpeas, onion, cilantro, lemon."),
                    ("Paneer Veggie Wrap", "Grilled paneer and lettuce in a whole-wheat wrap."),
                    ("Tomato Soup", "Unsalted, fresh, with a side of boiled vegetables."),
                ],
                dinner=[
                    ("Brown Rice &amp; Dal", "Low-sodium lentil dal with steamed broccoli."),
                    ("Baked Chicken Breast", "Garlic and lemon seasoned, with steamed spinach."),
                    ("Stir-fried Tofu", "Bell peppers, carrots, mushrooms in minimal oil."),
                ],
            )

        if chol_bad:
            render_plan(
                "🫀", "Cholesterol Control Plan", "For your elevated cholesterol reading",
                breakfast=[
                    ("Oat Bran", "Sprinkled with ground flaxseed and apple slices."),
                    ("Berry Smoothie", "Soy/almond milk, mixed berries, chia seeds."),
                    ("Multigrain Toast", "With unsweetened peanut butter, no palm oil."),
                ],
                lunch=[
                    ("Salmon/Tofu Salad", "Mixed greens, walnuts, extra virgin olive oil."),
                    ("Barley Soup", "Vegetable barley soup with steamed spinach."),
                    ("Chickpea Bowl", "Boiled chickpeas, cucumber, avocado, bell pepper."),
                ],
                dinner=[
                    ("Grilled Mackerel/Sardines", "With roasted sweet potato and asparagus."),
                    ("Dal Palak", "Lentils with fresh spinach and a small cup of brown rice."),
                    ("Vegetable Stir-fry", "Broccoli, beans, sprouts in healthy canola oil."),
                ],
            )

        if bmi_bad:
            render_plan(
                "⚖️", "Weight Management Plan", "For your elevated BMI reading",
                breakfast=[
                    ("Boiled Eggs", "2 hard-boiled eggs with unsweetened green tea."),
                    ("Moong Dal Sprouts", "Chopped cucumber, tomato, and lime juice."),
                    ("Chia Seed Pudding", "Skimmed milk, flavored with vanilla or cinnamon."),
                ],
                lunch=[
                    ("Chapati &amp; Sabzi", "1 whole-wheat chapati, mixed veg sabzi, curd."),
                    ("Grilled Tofu Salad", "High-protein, with cucumber, radish, and lettuce."),
                    ("Boiled Chicken Breast", "Sliced, with a side of steamed green peas."),
                ],
                dinner=[
                    ("Clear Vegetable Soup", "With boiled chicken breast or tofu chunks."),
                    ("Paneer Tikka", "Grilled with bell peppers and a cup of buttermilk."),
                    ("Sautéed Vegetables", "Broccoli, mushroom, zucchini in minimal spray oil."),
                ],
            )

        if not any_active:
            st.success("🎉 All your vitals are in the normal range! Here is your daily wellness maintenance plan:")
            render_plan(
                "🌿", "General Wellness Plan", "All vitals within healthy range — keep it up",
                breakfast=[
                    ("Poha/Upma", "Vegetable poha or suji upma with roasted peanuts."),
                    ("Paneer Toast", "Multigrain toast topped with low-fat paneer."),
                    ("Fresh Fruits", "A bowl of seasonal fruit with warm milk."),
                ],
                lunch=[
                    ("Balanced Thali", "2 multigrain chapatis, veg curry, dal, and salad."),
                    ("Chicken Pulao", "Brown rice pulao with cucumber raita."),
                    ("Soy Bean Curry", "With steamed basmati rice and sliced cucumber."),
                ],
                dinner=[
                    ("Vegetable Khichdi", "Light dal-rice khichdi with a teaspoon of ghee."),
                    ("Grilled Paneer/Chicken", "With steamed carrot, beans, and baby corn."),
                    ("Minestrone Soup", "Tomato vegetable soup with whole-wheat pasta shells."),
                ],
            )

    else:
        # Fallback when no patient has been analyzed yet in the current session
        st.markdown("""
        <div class="rx-card" style="border-left-color:#cbd5c9;">
            <div class="rx-eyebrow">No Profile On File</div>
            <div class="rx-title" style="margin-bottom:6px;">Run a health assessment to unlock your personal plan</div>
            <p class="food-desc" style="margin:0;font-size:.9rem;">
                Head to <b>Home &rarr; AI Health Assessment</b> first, and Sanjeevani will build a diet chart
                tailored to your glucose, blood pressure, cholesterol, and BMI. Until then, here is a
                general wellness chart to get you started.
            </p>
        </div>
        """, unsafe_allow_html=True)

        render_plan(
            "🥦", "General Balanced Wellness Plan", "A safe starting point for anyone",
            breakfast=[
                ("Oatmeal or Porridge", "Cooked with sliced apples and a handful of nuts."),
                ("Paneer/Egg Scramble", "Served with multigrain toast."),
                ("Vegetable Poha", "Curry leaves, mustard seeds, roasted peanuts."),
            ],
            lunch=[
                ("Roti, Sabzi &amp; Dal", "Wheat chapatis, mixed veg, and lentil dal."),
                ("Brown Rice Chicken Curry", "With a bowl of cooling cucumber raita."),
                ("Chickpea Salad", "Bell peppers, cucumber, paneer, and lime."),
            ],
            dinner=[
                ("Light Khichdi", "Easy-to-digest rice and yellow lentil stew."),
                ("Grilled Fish or Paneer", "Seasoned with herbs, with steamed vegetables."),
                ("Vegetable Soup", "Clear soup with cabbage, spinach, and tofu/chicken."),
            ],
        )

    # ---------- Daily Water Recommendation ----------

    if 'latest_analysis' in st.session_state:
        water_age = st.session_state['latest_analysis']['age']
        water_bmi = st.session_state['latest_analysis']['bmi']
        water_basis_note = ""
    else:
        water_age = 30
        water_bmi = 22.0
        water_basis_note = '<p class="sj-note" style="margin-top:-6px;">Showing a general recommendation — run a health assessment for a value personalised to you.</p>'

    daily_water = get_water_recommendation(water_age, water_bmi)

    st.markdown(f"""
    <div class="water-card">
        <h3>💧 Daily Water Recommendation</h3>
        <div class="water-amount-row">
            <div class="water-amount">{daily_water} L</div>
            <div class="water-amount-label">Recommended intake / day</div>
        </div>
        <div class="water-basis-row">
            <span class="water-basis-chip">✔ Age</span>
            <span class="water-basis-chip">✔ BMI</span>
            <span class="water-basis-chip">✔ General Health</span>
        </div>
        <div class="water-tips">
            <h4>Tips</h4>
            <ul>
                <li>Drink throughout the day rather than all at once</li>
                <li>Increase intake after exercise or on hot days</li>
                <li>Keep a bottle nearby as a reminder to sip regularly</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if water_basis_note:
        st.markdown(water_basis_note, unsafe_allow_html=True)


# =====================================================
# ABOUT SANJEEVANI PAGE
# =====================================================

elif st.session_state.page == "about":

    render_page_brand_header()
    render_top_nav()

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    .about-header{
        background:linear-gradient(135deg,#166534,#22c55e);
        padding:36px 30px;
        border-radius:25px;
        text-align:center;
        color:white;
        box-shadow:0px 10px 30px rgba(0,0,0,0.15);
        margin-bottom:24px;
    }
    .about-header h1{
        font-family:'Fraunces',serif;
        margin-bottom:6px;
    }
    .about-header p{
        font-family:'Inter',sans-serif;
        font-size:1.05rem;
        opacity:0.95;
    }

    .about-card{
        background:white;
        border-radius:20px;
        padding:26px 28px;
        box-shadow:0 6px 20px rgba(0,0,0,0.08);
        border:1px solid #dcfce7;
        margin-bottom:22px;
    }
    .about-card h3{
        color:#14532d;
        font-family:'Fraunces',serif;
        margin-bottom:10px;
    }
    .about-card p, .about-card li{
        color:#374151;
        font-size:1rem;
        line-height:1.6;
    }

    .feature-grid{
        display:grid;
        grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
        gap:16px;
        margin-top:10px;
    }
    .feature-tile{
        background:#f0fdf4;
        border:1px solid #bbf7d0;
        border-radius:16px;
        padding:18px;
        text-align:center;
        transition:transform 0.15s ease;
    }
    .feature-tile:hover{
        transform:translateY(-3px);
    }
    .feature-tile .icon{
        font-size:1.8rem;
        margin-bottom:6px;
    }
    .feature-tile .title{
        font-weight:700;
        color:#166534;
        margin-bottom:4px;
    }
    .feature-tile .desc{
        font-size:0.88rem;
        color:#4b5563;
    }

    .helpline-grid{
        display:grid;
        grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));
        gap:14px;
        margin-top:12px;
    }
    .helpline-card{
        background:#fef2f2;
        border-left:5px solid #ef4444;
        border-radius:14px;
        padding:16px 18px;
    }
    .helpline-card .name{
        font-weight:700;
        color:#7f1d1d;
        font-size:0.95rem;
        margin-bottom:2px;
    }
    .helpline-card .number{
        font-size:1.3rem;
        font-weight:700;
        color:#b91c1c;
        letter-spacing:0.5px;
    }
    .helpline-card .desc{
        font-size:0.82rem;
        color:#7c2d2d;
        margin-top:2px;
    }

    .contact-grid{
        display:grid;
        grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
        gap:14px;
        margin-top:12px;
    }
    .contact-tile{
        background:#f0fdf4;
        border:1px solid #bbf7d0;
        border-radius:14px;
        padding:16px 18px;
        text-align:center;
    }
    .contact-tile .icon{
        font-size:1.5rem;
        margin-bottom:6px;
    }
    .contact-tile .label{
        font-weight:700;
        color:#166534;
        margin-bottom:2px;
        font-size:0.9rem;
    }
    .contact-tile .value{
        font-size:0.88rem;
        color:#374151;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- Header ----
    st.markdown("""
    <div class="about-header">
        <h1>ℹ️ About Sanjeevani</h1>
        <p>Your AI-powered companion for smarter, simpler, more personal healthcare</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Our Story / Mission ----
    st.markdown("""
    <div class="about-card">
        <h3>🌱 Our Mission</h3>
        <p>
        Sanjeevani was built with one simple belief — everyone deserves easy access to
        clear, personalized health guidance. Named after the mythical life-restoring herb,
        Sanjeevani combines Artificial Intelligence with everyday wellness data to help you
        understand your body better, spot risks early, and make informed lifestyle choices.
        We are not a replacement for a doctor, but we aim to be the friendly first step on
        your health journey — available anytime, anywhere.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ---- What Sanjeevani Offers ----
    st.markdown("""
    <div class="about-card">
        <h3>✨ What Sanjeevani Offers</h3>
        <div class="feature-grid">
            <div class="feature-tile">
                <div class="icon">🩺</div>
                <div class="title">AI Health Assessment</div>
                <div class="desc">Get an instant risk profile from your vitals using our trained ML model.</div>
            </div>
            <div class="feature-tile">
                <div class="icon">📈</div>
                <div class="title">Weekly Analysis</div>
                <div class="desc">Track how your health metrics trend across the week.</div>
            </div>
            <div class="feature-tile">
                <div class="icon">🔮</div>
                <div class="title">AI Future Self</div>
                <div class="desc">See a projection of your future health based on current habits.</div>
            </div>
            <div class="feature-tile">
                <div class="icon">🥗</div>
                <div class="title">Diet Chart</div>
                <div class="desc">Receive a personalized meal plan tailored to your health profile.</div>
            </div>
            <div class="feature-tile">
                <div class="icon">🔬</div>
                <div class="title">Health Report Analyser</div>
                <div class="desc">Upload lab reports or scans for an AI-assisted read (coming soon).</div>
            </div>
            <div class="feature-tile">
                <div class="icon">💬</div>
                <div class="title">Vani — Your AI Assistant</div>
                <div class="desc">Chat with Vani for quick answers to your health questions.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Helplines ----
    st.markdown("""
    <div class="about-card">
        <h3>🚑 Emergency Helplines (India)</h3>
        <p style="margin-bottom:4px;">If this is a medical emergency, please contact a doctor or hospital immediately. These helplines are available 24x7:</p>
        <div class="helpline-grid">
            <div class="helpline-card">
                <div class="name">National Emergency Number</div>
                <div class="number">112</div>
                <div class="desc">Police, fire &amp; medical emergencies</div>
            </div>
            <div class="helpline-card">
                <div class="name">Ambulance</div>
                <div class="number">108</div>
                <div class="desc">Free emergency ambulance service</div>
            </div>
            <div class="helpline-card">
                <div class="name">National Health Helpline</div>
                <div class="number">104</div>
                <div class="desc">Medical advice &amp; health information</div>
            </div>
            <div class="helpline-card">
                <div class="name">KIRAN Mental Health Helpline</div>
                <div class="number">1800-599-0019</div>
                <div class="desc">24x7 toll-free mental health support</div>
            </div>
            <div class="helpline-card">
                <div class="name">Child Helpline</div>
                <div class="number">1098</div>
                <div class="desc">Support &amp; protection for children</div>
            </div>
            <div class="helpline-card">
                <div class="name">Women's Helpline</div>
                <div class="number">181</div>
                <div class="desc">24x7 support for women in distress</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Contact Us ----
    st.markdown("""
    <div class="about-card">
        <h3>📩 Contact Us</h3>
        <p style="margin-bottom:4px;">Have questions, feedback, or need support? We'd love to hear from you.</p>
        <div class="contact-grid">
            <div class="contact-tile">
                <div class="icon">📧</div>
                <div class="label">Email</div>
                <div class="value">support@sanjeevani-health.ai</div>
            </div>
            <div class="contact-tile">
                <div class="icon">📞</div>
                <div class="label">Phone</div>
                <div class="value">+91 98765 43210</div>
            </div>
            <div class="contact-tile">
                <div class="icon">🕒</div>
                <div class="label">Support Hours</div>
                <div class="value">Mon–Sat, 9 AM – 7 PM IST</div>
            </div>
            <div class="contact-tile">
                <div class="icon">📍</div>
                <div class="label">Based In</div>
                <div class="value">India</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center; color:#6b7280; font-size:0.85rem; margin-top:10px;">
    ⚠️ Sanjeevani provides AI-generated wellness guidance and does not substitute professional medical advice, diagnosis, or treatment.
    </p>
    """, unsafe_allow_html=True)


# =====================================================
# FAQ PAGE
# =====================================================

elif st.session_state.page == "faq":

    render_page_brand_header()
    render_top_nav()

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    .faq-header{
        background:linear-gradient(135deg,#166534,#22c55e);
        padding:36px 30px;
        border-radius:25px;
        text-align:center;
        color:white;
        box-shadow:0px 10px 30px rgba(0,0,0,0.15);
        margin-bottom:24px;
    }
    .faq-header h1{
        font-family:'Fraunces',serif;
        margin-bottom:6px;
    }
    .faq-header p{
        font-family:'Inter',sans-serif;
        font-size:1.05rem;
        opacity:0.95;
    }

    div[data-testid="stExpander"]{
        background:white;
        border:1px solid #dcfce7;
        border-radius:16px !important;
        box-shadow:0 4px 14px rgba(0,0,0,0.06);
        margin-bottom:14px;
    }
    div[data-testid="stExpander"] summary{
        font-weight:600;
        color:#14532d;
        font-size:1.02rem;
    }

    .faq-contact-card{
        background:#f0fdf4;
        border:1px solid #bbf7d0;
        border-radius:18px;
        padding:22px 26px;
        text-align:center;
        margin-top:10px;
    }
    .faq-contact-card h4{
        color:#166534;
        margin-bottom:6px;
    }
    .faq-contact-card p{
        color:#374151;
        margin-bottom:2px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- Header ----
    st.markdown("""
    <div class="faq-header">
        <h1>❓ Frequently Asked Questions</h1>
        <p>Everything you need to know about using Sanjeevani</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    faqs = [
        ("🩺 What is Sanjeevani?",
         "Sanjeevani is an AI-powered health companion that analyzes your vitals and lifestyle "
         "inputs to give you a personalized health risk profile, diet suggestions, weekly trend "
         "analysis, and a glimpse into your future health — all in one place."),

        ("🤖 How does the AI Health Assessment work?",
         "You enter details like age, BMI, glucose, blood pressure, cholesterol and more. "
         "Our trained Machine Learning model then analyzes this data to estimate your health "
         "risk and highlights which areas need attention."),

        ("🔒 Is my health data safe?",
         "Your inputs are used only within this session to generate your report, charts and "
         "recommendations. We recommend not entering sensitive personal identifiers, and always "
         "reviewing your organization's data policies if this app is self-hosted."),

        ("🥗 How is my diet chart created?",
         "Once you complete a health assessment, Sanjeevani looks at your glucose, blood pressure, "
         "cholesterol and BMI readings and automatically builds a breakfast-lunch-dinner plan "
         "targeted at the specific areas that need improvement."),

        ("🔮 What is the AI Future Self feature?",
         "It's a predictive wellness tool that projects how your health metrics could evolve over "
         "time if your current habits continue, helping you visualize the long-term impact of "
         "today's choices."),

        ("🔬 What will the Health Report Analyser do?",
         "This upcoming feature will let you upload medical reports, scans or lab results as images, "
         "which Sanjeevani will read using AI (Vision-Language Models) to help you understand the "
         "key findings in simple language. This feature is currently under active development."),

        ("💬 Who is Vani?",
         "Vani is Sanjeevani's built-in AI chat assistant, available to answer general health and "
         "wellness questions in a conversational manner."),

        ("👩‍⚕️ Does Sanjeevani replace a doctor?",
         "No. Sanjeevani is designed to support and inform — not replace — professional medical "
         "advice. Always consult a qualified healthcare provider for diagnosis and treatment."),

        ("🌐 Do I need an internet connection to use Sanjeevani?",
         "Yes, an active internet connection is required for the AI features, charts and "
         "assistant to function correctly."),
    ]

    for question, answer in faqs:
        with st.expander(question):
            st.write(answer)

    # ---- Contact Us callout ----
    st.markdown("""
    <div class="faq-contact-card">
        <h4>📩 Still have questions?</h4>
        <p>Reach out to us at <b>support@sanjeevani-health.ai</b> or call <b>+91 98765 43210</b></p>
        <p style="font-size:0.85rem; color:#4b5563;">Mon–Sat, 9 AM – 7 PM IST</p>
        <p style="font-size:0.85rem; color:#4b5563; margin-top:8px;">
        For medical emergencies, please see the Helplines listed on our <b>About Sanjeevani</b> page.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =====================================================
# HEALTH REPORT ANALYSER PAGE
# =====================================================

elif st.session_state.page == "report_analyser":

    render_page_brand_header()
    render_top_nav()

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    .hra-header{
        background:linear-gradient(135deg,#166534,#22c55e);
        padding:36px 30px;
        border-radius:25px;
        text-align:center;
        color:white;
        box-shadow:0px 10px 30px rgba(0,0,0,0.15);
        margin-bottom:24px;
    }
    .hra-header h1{
        font-family:'Fraunces',serif;
        margin-bottom:6px;
    }
    .hra-header p{
        font-family:'Inter',sans-serif;
        font-size:1.05rem;
        opacity:0.95;
    }

    .hra-card{
        background:white;
        border-radius:20px;
        padding:26px 28px;
        box-shadow:0 6px 20px rgba(0,0,0,0.08);
        border:1px solid #dcfce7;
        margin-bottom:22px;
    }
    .hra-card h3{
        color:#14532d;
        font-family:'Fraunces',serif;
        margin-bottom:10px;
    }

    .hra-steps{
        display:grid;
        grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));
        gap:14px;
        margin-top:6px;
    }
    .hra-step{
        background:#f0fdf4;
        border:1px solid #bbf7d0;
        border-radius:14px;
        padding:14px 16px;
        text-align:center;
    }
    .hra-step .num{
        display:inline-block;
        background:#166534;
        color:white;
        width:26px;
        height:26px;
        border-radius:50%;
        font-size:0.85rem;
        font-weight:700;
        line-height:26px;
        margin-bottom:6px;
    }
    .hra-step .txt{
        font-size:0.85rem;
        color:#374151;
    }

    .hra-upload-zone{
        border:2px dashed #86efac;
        border-radius:18px;
        padding:10px 10px 4px 10px;
        background:#f8fff8;
    }

    .hra-preview-tile{
        background:white;
        border:1px solid #dcfce7;
        border-radius:14px;
        padding:10px;
        box-shadow:0 4px 12px rgba(0,0,0,0.06);
        text-align:center;
        margin-bottom:12px;
    }
    .hra-preview-tile .fname{
        font-size:0.78rem;
        color:#4b5563;
        margin-top:6px;
        word-break:break-all;
    }

    .hra-result-placeholder{
        background:#fefaf3;
        border:1px dashed #e7dfca;
        border-radius:18px;
        padding:30px;
        text-align:center;
        color:#8a7f63;
    }

    .hra-badge{
        display:inline-block;
        background:#fffbeb;
        color:#92400e;
        border:1px solid #fde68a;
        border-radius:999px;
        padding:4px 14px;
        font-size:0.78rem;
        font-weight:700;
        letter-spacing:0.03em;
        margin-bottom:14px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- Header ----
    st.markdown("""
    <div class="hra-header">
        <h1>🔬 Health Report Analyser</h1>
        <p>Upload your medical reports or scans and let Sanjeevani AI help you understand them</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- How it works ----
    st.markdown("""
    <div class="hra-card">
        <h3>📋 How It Will Work</h3>
        <div class="hra-steps">
            <div class="hra-step"><div class="num">1</div><div class="txt">Upload a photo or scan of your report</div></div>
            <div class="hra-step"><div class="num">2</div><div class="txt">Add any notes or symptoms (optional)</div></div>
            <div class="hra-step"><div class="num">3</div><div class="txt">Sanjeevani's AI reads and interprets the report</div></div>
            <div class="hra-step"><div class="num">4</div><div class="txt">Get a simple, easy-to-understand summary</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Upload + Notes Section ----
    col_upload, col_notes = st.columns([1.2, 1])

    with col_upload:
        st.markdown("""
        <div class="hra-card">
            <h3>📤 Upload Report / Scan</h3>
        """, unsafe_allow_html=True)

        st.markdown('<div class="hra-upload-zone">', unsafe_allow_html=True)
        uploaded_reports = st.file_uploader(
            "Upload one or more images (JPG, PNG) or PDFs of your health report",
            type=["jpg", "jpeg", "png", "pdf"],
            accept_multiple_files=True,
            key="hra_uploaded_reports"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Preview uploaded images ----
        if uploaded_reports:
            st.markdown("""
            <div class="hra-card">
                <h3>🖼️ Uploaded Files Preview</h3>
            """, unsafe_allow_html=True)

            preview_cols = st.columns(3)
            for idx, file in enumerate(uploaded_reports):
                with preview_cols[idx % 3]:
                    st.markdown('<div class="hra-preview-tile">', unsafe_allow_html=True)
                    if file.type in ["image/jpeg", "image/png", "image/jpg"]:
                        st.image(file, use_container_width=True)
                    else:
                        st.markdown("📄", unsafe_allow_html=True)
                    st.markdown(f'<div class="fname">{file.name}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    with col_notes:
        st.markdown("""
        <div class="hra-card">
            <h3>📝 Additional Notes</h3>
        """, unsafe_allow_html=True)

        report_notes = st.text_area(
            "Any symptoms, concerns, or context you'd like Sanjeevani to consider (optional)",
            placeholder="e.g. I've had these results for a fasting blood test, feeling more tired than usual...",
            height=160,
            key="hra_report_notes"
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="hra-card">
            <h3>💡 Tips for Best Results</h3>
            <ul style="color:#374151; font-size:0.9rem; padding-left:18px;">
                <li>Use clear, well-lit photos of the full report</li>
                <li>Avoid glare or blurred text</li>
                <li>Upload one report per file for clarity</li>
                <li>PDFs work best when text-based, not scanned images</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ---- Analyze button ----
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    with analyze_col2:
        analyze_clicked = st.button(
            "🔍 Analyze Report",
            use_container_width=True,
            key="hra_analyze_button"
        )

    # ---- Results placeholder ----
    if analyze_clicked:
        if not uploaded_reports:
            st.warning("Please upload at least one report or scan before analyzing.")
        else:

            with st.spinner("🧠 Sanjeevani is analyzing your report..."):

                report_file = uploaded_reports[0]

                result = analyze_report(report_file, report_notes)
                st.session_state["report_analysis_result"] = result

            st.markdown(f"""
            <div style="
            background:white;
            padding:25px;
            border-radius:20px;
            box-shadow:0px 6px 20px rgba(0,0,0,0.08);
            border:1px solid #dcfce7;
            ">
            <h3 style="color:#166534;">
            📋 Report Analysis
            </h3>

            <div style="
            color:#374151;
            line-height:1.8;
            font-size:16px;
            ">
            {result.replace("\n", "<br>")}
            </div>

            </div>
            """, unsafe_allow_html=True)
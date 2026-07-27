# SANJEEVANI - Your AI Health Companion

An AI-powered healthcare assistant built using **Machine Learning**, **Streamlit**, and **OpenRouter LLMs**.

SANJEEVANI is an intelligent healthcare application that evaluates vital health parameters, predicts health risk, provides AI-powered health insights, analyzes medical reports, and maintains patient assessment history using SQLite.

---

## Live Demo

🔗 **Streamlit App:** https://sanjeevani-healthcare-ai-fvhusawvwt8f4vifpzbkft.streamlit.app/

---

## Features

### AI Health Risk Prediction

- Predicts diabetes risk using a trained Machine Learning model
- Calculates Risk Score and Health Score
- Classifies users into:
  - Excellent Health
  - Healthy
  - Moderate Risk
  - High Risk
  - Critical Risk

---

### AI Health Insights

- Personalized health observations
- Risk interpretation
- Dynamic health recommendations

---

### VANI AI Assistant

Powered using OpenRouter LLM.

Users can:

- Ask questions about their health report
- Understand medical values
- Receive AI-generated explanations

---

### Medical Report Analyzer

Upload medical reports and receive AI-powered analysis.

---

### Weekly Analytics Dashboard

Visual dashboard including:

- Total Assessments
- Healthy vs At-Risk Cases
- Health Score Gauge
- Pie Charts
- Health Breakdown
- Vital Status Dashboard
- Weekly Summary

---

### PDF Report Generation

Generate and download a complete health report including:

- Patient Details
- Risk Assessment
- AI Insights
- VANI Responses
- Medical Report Analysis

---

### SQLite Database

Assessment history is stored using SQLite.

No CSV files are required.

---

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Plotly
- ReportLab
- SQLite
- OpenRouter API
- OpenAI SDK

---

## 📂 Project Structure

```text
SANJEEVANI-HEALTHCARE-AI/
│
├── app.py                         # Main Streamlit application
├── model/                         # Trained machine learning model files
├── health.db                      # SQLite database for assessment history
├── init_db.py                     # Creates and initializes the database
├── report_vlm.py                  # AI-powered medical report analysis
├── vani_ai.py                     # AI Health Assistant (Vani)
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
│
├── logo2.jpg                      # Application logo
├── Female avatar.json             # Lottie animation
├── Looping Energy Orb.json        # UI animation
├── result page success motion design.json
│
├── 2_model_training.ipynb         # Model training notebook
└── test_healthcare_data.csv       # Sample healthcare dataset
```
---

## Installation

Clone the repository

```bash
git clone <your-repository-link>
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```text
OPENROUTER_API_KEY=your_api_key
```

Run

```bash
streamlit run app.py
```

---

## Environment Variables

Required:

```text
OPENROUTER_API_KEY
```

On Streamlit Cloud, add this key inside **Secrets**.

---

## Future Improvements

- Multi-user login
- Doctor dashboard
- Appointment booking
- Mobile responsive UI
- Health trend forecasting

---

## Author

**Priyanshi**

BCA Student | Machine Learning Enthusiast

Built as an internship project.

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide"
)

# ---------------- ADVANCED CUSTOM CSS ----------------
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

/* Global Styles */
* {
    font-family: 'Inter', sans-serif;
}

body {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
}

.main {
    background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Header Gradient Text */
h1 {
    background: linear-gradient(135deg, #00F5FF 0%, #7B2FF7 50%, #FF00E5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    font-size: 3rem;
    text-align: center;
    margin-bottom: 0.5rem;
    animation: gradient-shift 3s ease infinite;
}

@keyframes gradient-shift {
    0%, 100% { filter: hue-rotate(0deg); }
    50% { filter: hue-rotate(20deg); }
}

h2, h3 {
    color: #00F5FF;
    font-weight: 600;
}

/* Subtitle styling */
.subtitle {
    text-align: center;
    color: #9CA3AF;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    font-weight: 300;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 2rem;
    box-shadow: 0 8px 32px 0 rgba(0, 245, 255, 0.1);
    transition: all 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(0, 245, 255, 0.2);
    border: 1px solid rgba(0, 245, 255, 0.3);
}

/* Input fields styling */
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(0, 245, 255, 0.3) !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
}

.stNumberInput > div > div > input:focus,
.stSelectbox > div > div > div:focus {
    border: 1px solid #00F5FF !important;
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.3) !important;
}

/* Labels */
label {
    color: #E5E7EB !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

/* Premium Button */
.stButton > button {
    background: linear-gradient(135deg, #00F5FF 0%, #7B2FF7 50%, #FF00E5 100%);
    color: white;
    font-weight: 700;
    border-radius: 16px;
    height: 3.5em;
    width: 100%;
    border: none;
    font-size: 1.1rem;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(123, 47, 247, 0.4);
    position: relative;
    overflow: hidden;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(123, 47, 247, 0.6);
}

.stButton > button:active {
    transform: translateY(-1px);
}

/* Metrics */
.stMetric {
    background: rgba(255, 255, 255, 0.03);
    padding: 1.5rem;
    border-radius: 15px;
    border: 1px solid rgba(0, 245, 255, 0.2);
    text-align: center;
}

.stMetric label {
    font-size: 1rem !important;
    color: #9CA3AF !important;
}

.stMetric [data-testid="stMetricValue"] {
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #00F5FF 0%, #7B2FF7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00F5FF 0%, #7B2FF7 50%, #FF00E5 100%);
    border-radius: 10px;
    height: 20px;
}

.stProgress > div > div {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

/* Alert boxes */
.stAlert {
    border-radius: 15px;
    border: none;
    backdrop-filter: blur(10px);
    font-weight: 500;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0, 245, 255, 0.3), transparent);
    margin: 2rem 0;
}

/* Insights section */
.insight-item {
    background: rgba(255, 255, 255, 0.02);
    padding: 1rem 1.5rem;
    border-left: 3px solid #00F5FF;
    border-radius: 8px;
    margin: 0.8rem 0;
    color: #D1D5DB;
    font-size: 0.95rem;
    transition: all 0.3s ease;
}

.insight-item:hover {
    background: rgba(0, 245, 255, 0.05);
    transform: translateX(5px);
}

/* Result container */
.result-container {
    background: rgba(255, 255, 255, 0.02);
    border-radius: 20px;
    padding: 2rem;
    border: 1px solid rgba(0, 245, 255, 0.2);
    margin-top: 2rem;
}

/* Columns */
.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# ---------------- ANIMATED HEADER ----------------
st.markdown('<h1>📡 Telecom Churn Intelligence</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">🔮 Advanced AI-Powered Customer Retention Analytics</p>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- INPUT SECTION WITH GLASS CARDS ----------------
st.markdown("### 📋 Customer Information")
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    age = st.number_input("👤 Customer Age", min_value=18, max_value=100, value=30, help="Age of the customer")
    tenure = st.number_input("📅 Tenure (Months)", min_value=0, max_value=120, value=12, help="How long they've been a customer")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    monthly_charges = st.number_input("💰 Monthly Charges (₹)", min_value=100, max_value=10000, value=1000, help="Monthly bill amount")
    contract = st.selectbox(
        "📝 Contract Type",
        ["Month-to-Month", "One Year", "Two Year"],
        help="Type of contract agreement"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    internet = st.selectbox(
        "🌐 Internet Service",
        ["DSL", "Fiber"],
        help="Type of internet service"
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- PREDICTION SECTION ----------------
if st.button("🚀 Analyze Customer Churn Risk"):

    input_data = {
        "Age": age,
        "Tenure": tenure,
        "Monthly_Charges": monthly_charges,
        "Contract_Type": contract,
        "Internet_Service": internet
    }

    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)

    model_columns = [
        'Age',
        'Tenure',
        'Monthly_Charges',
        'Contract_Type_One Year',
        'Contract_Type_Two Year',
        'Internet_Service_Fiber'
    ]

    for col in model_columns:
        if col not in input_df:
            input_df[col] = 0

    input_df = input_df[model_columns]

    # Apply preprocessing
    scaled = scaler.transform(input_df)
    reduced = pca.transform(scaled)

    prediction = model.predict(reduced)[0]
    probability = model.predict_proba(reduced)[0][1]

    # Results Section
    st.markdown('<div class="result-container">', unsafe_allow_html=True)
    
    st.markdown("## 📊 Risk Analysis Report")
    st.markdown("<br>", unsafe_allow_html=True)

    # Probability visualization
    st.markdown("### Churn Probability Score")
    st.progress(float(probability))
    
    st.markdown("<br>", unsafe_allow_html=True)

    colA, colB = st.columns(2, gap="large")

    with colA:
        st.metric("📈 Churn Probability", f"{probability*100:.1f}%", delta=f"{(probability*100 - 50):.1f}%" if probability > 0.5 else f"{(50 - probability*100):.1f}%")

    with colB:
        if prediction == 1:
            st.error("⚠️ **HIGH RISK** – Customer Likely to Churn")
        else:
            st.success("✅ **LOW RISK** – Customer Likely to Stay")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ---------------- ACTIONABLE INSIGHTS ----------------
    st.markdown("## 🧠 AI-Generated Insights")
    st.markdown("<br>", unsafe_allow_html=True)

    insights = []
    
    if tenure < 12:
        insights.append("📌 Short tenure detected: New customers have 2.5x higher churn risk. Consider offering loyalty incentives.")
    if contract == "Month-to-Month":
        insights.append("📌 Month-to-Month contracts show 3x higher churn rates. Promote annual contract upgrades with discounts.")
    if monthly_charges > 1500:
        insights.append("📌 Premium pricing tier: Higher charges correlate with increased churn. Review value proposition and service quality.")
    if internet == "Fiber":
        insights.append("📌 Fiber service users may churn due to competitive offers. Ensure service stability and competitive pricing.")
    if age < 25:
        insights.append("📌 Younger demographic detected: This age group is more price-sensitive and competitive-aware.")
    if probability > 0.7:
        insights.append("📌 CRITICAL ALERT: Extremely high churn risk! Immediate retention intervention recommended.")
    elif probability > 0.5:
        insights.append("📌 Moderate risk detected: Proactive engagement recommended within 7 days.")
    else:
        insights.append("📌 Customer is stable: Continue monitoring satisfaction metrics quarterly.")

    for insight in insights:
        st.markdown(f'<div class="insight-item">{insight}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
        <p>🤖 Powered by Machine Learning & PCA Feature Engineering | Built with Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)
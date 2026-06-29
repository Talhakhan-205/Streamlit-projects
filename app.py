import streamlit as st
import joblib
import pandas as pd
import datetime
import time
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Header ── */
.header-wrap {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
}
.main-title {
    color: #ffffff;
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
    background: linear-gradient(90deg, #fff 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-title {
    color: #6b7280;
    font-size: 0.85rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 6px;
}

/* ── Card ── */
.card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 18px;
    padding: 2rem;
    backdrop-filter: blur(12px);
}

/* ── Section label ── */
.section-label {
    color: #9ca3af;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 8px;
}

/* ── Inputs ── */
label, .stSelectbox label, .stNumberInput label,
.stSlider label, .stSelectSlider label {
    color: #9ca3af !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
}

/* ── Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
    color: white;
    font-weight: 700;
    font-size: 0.9rem;
    letter-spacing: 1.5px;
    border: none;
    border-radius: 12px;
    height: 3.4em;
    margin-top: 1.5rem;
    text-transform: uppercase;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(124, 58, 237, 0.6);
}

/* ── Result Card ── */
.result-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.15) 0%, rgba(167,139,250,0.1) 100%);
    border: 1px solid rgba(124, 58, 237, 0.4);
    border-radius: 18px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeIn 0.5s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-label {
    color: #9ca3af;
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.result-price {
    color: #ffffff;
    font-size: 3rem;
    font-weight: 800;
    margin: 8px 0;
    background: linear-gradient(90deg, #a78bfa 0%, #fff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.result-meta {
    color: #6b7280;
    font-size: 0.82rem;
    border-top: 1px solid rgba(255,255,255,0.08);
    padding-top: 12px;
    margin-top: 12px;
    letter-spacing: 1px;
}

/* ── Stats row ── */
.stat-box {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-value { color: #a78bfa; font-size: 1.3rem; font-weight: 700; }
.stat-label { color: #6b7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; }

/* ── Error ── */
.stAlert { border-radius: 12px !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #374151;
    font-size: 0.7rem;
    letter-spacing: 1px;
    padding: 2rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'car_price_model.pkl')
    return joblib.load(model_path)

model = load_model()

# ── Brand → Model Mapping (for smart dropdown filtering) ─────────────────────
BRAND_MODEL_MAP = {
    "Toyota":   ["Corolla", "Yaris", "Fortuner", "Vitz", "Aqua", "Passo",
                 "Raize", "Surf", "Hilux", "Prius"],
    "Suzuki":   ["Alto", "Cultus", "Wagon r", "Mehran", "Swift", "Bolan",
                 "Ravi", "Liana", "Baleno", "Khyber"],
    "Honda":    ["Civic", "City", "Vezel", "N wgn"],
    "Hyundai":  ["Tucson", "Sonata", "Elantra", "Santa fe"],
    "Kia":      ["Sportage", "Sorento"],
    "Nissan":   ["Dayz", "Moco"],
    "Daihatsu": ["Others"],
    "Others":   ["Others"],
}

ALL_MODELS = sorted(set(m for models in BRAND_MODEL_MAP.values() for m in models))

CURRENT_YEAR = datetime.datetime.now().year

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-wrap">
    <h1 class="main-title">🏎️ Car Price Predictor</h1>
    <p class="sub-title">OLX Pakistan · AI Valuation Engine · 2025 Data</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
left_gap, main_col, right_gap = st.columns([1, 3, 1])

with main_col:

    # ── Input Card ────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">Vehicle Details</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox(
            "Brand",
            list(BRAND_MODEL_MAP.keys()),
            help="Select the car manufacturer"
        )
    with col2:
        # Smart filter: only show models for selected brand + Others
        available_models = BRAND_MODEL_MAP.get(brand, ["Others"])
        if "Others" not in available_models:
            available_models = available_models + ["Others"]
        model_name = st.selectbox("Model", available_models)

    col3, col4 = st.columns(2)
    with col3:
        year = st.number_input(
            "Manufacturing Year",
            min_value=1990,
            max_value=CURRENT_YEAR,
            value=2020,
            step=1
        )
        fuel = st.selectbox("Fuel / Engine Type", ["Petrol", "Hybrid", "Others"])

    with col4:
        mileage = st.number_input(
            "Odometer Reading (km)",
            min_value=0,
            max_value=300000,
            value=50000,
            step=1000,
            help="Enter 0 for brand new car"
        )
        seats = st.selectbox(
            "Number of Seats",
            [2, 4, 5, 7, 8],
            index=2
        )

    st.markdown("<br>", unsafe_allow_html=True)
    condition = st.select_slider(
        "Vehicle Condition Score (1 = Poor  →  10 = Excellent)",
        options=list(range(1, 11)),
        value=7,
        help="Rate the overall physical and mechanical condition"
    )

    # Condition description helper
    condition_text = {
        1: "⚠️ Very Poor — Needs major repairs",
        2: "⚠️ Poor — Multiple issues",
        3: "🔧 Below Average — Visible wear",
        4: "🔧 Fair — Minor repairs needed",
        5: "✅ Average — Normal used car",
        6: "✅ Good — Clean and working",
        7: "⭐ Very Good — Well maintained",
        8: "⭐ Excellent — Near perfect",
        9: "🏆 Outstanding — Like new",
        10: "🏆 Perfect — Brand new condition",
    }
    st.caption(condition_text.get(condition, ""))

    predict_btn = st.button("🔍 Analyze & Predict Price", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Quick Stats (always visible) ──────────────────────────
    car_age_display = CURRENT_YEAR - year
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{car_age_display} yrs</div><div class="stat-label">Car Age</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{mileage:,} km</div><div class="stat-label">Mileage</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{condition}/10</div><div class="stat-label">Condition</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-box"><div class="stat-value">{seats}</div><div class="stat-label">Seats</div></div>', unsafe_allow_html=True)

    # ── Prediction Logic ──────────────────────────────────────
    if predict_btn:
        with st.spinner("Analyzing vehicle data..."):
            time.sleep(0.6)

        car_age = CURRENT_YEAR - year

        # Build input dict — one-hot style
        input_dict = {
            'Car_Age':   car_age,
            'Mileage':   mileage,
            'Seats':     seats,
            'Condition': condition,
            f'Brand_{brand}':      1,
            f'Model_{model_name}': 1,
            f'Fuel_{fuel}':        1,
        }

        df_input = pd.DataFrame([input_dict])

        try:
            actual_features = model.feature_names_in_
            df_final = df_input.reindex(columns=actual_features, fill_value=0)
            prediction = model.predict(df_final)[0]

            # ── Price bands for context ──
            low  = prediction * 0.92
            high = prediction * 1.08

            st.markdown(f"""
            <div class="result-card">
                <p class="result-label">Estimated Market Value</p>
                <p class="result-price">PKR {prediction:,.0f}</p>
                <p style="color:#6b7280; font-size:0.8rem; margin-top:-8px;">
                    Typical Range: PKR {low:,.0f} – PKR {high:,.0f}
                </p>
                <p class="result-meta">
                    {brand.upper()} {model_name.upper()} &nbsp;|&nbsp;
                    {year} Model &nbsp;|&nbsp;
                    {fuel.upper()} &nbsp;|&nbsp;
                    {mileage:,} km
                </p>
            </div>
            """, unsafe_allow_html=True)

            # ── Lacs display ──
            lacs = prediction / 100_000
            st.success(f"✅ Predicted Price: **PKR {lacs:.2f} Lacs** ({prediction:,.0f})")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Tip: Make sure `car_price_model.pkl` is in the same folder as this script.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    AI DRIVEN VALUATION · TRAINED ON OLX PAKISTAN 2025 DATA · FOR REFERENCE ONLY
</div>
""", unsafe_allow_html=True)

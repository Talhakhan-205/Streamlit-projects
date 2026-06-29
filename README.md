# 🏎️ Car Price Predictor — OLX Pakistan

AI-powered car valuation tool trained on OLX Pakistan 2025 data.

---

## 📁 Project Structure

```
car_price_predictor/
│
├── app.py                  ← Main Streamlit app (yeh run karo)
├── data_cleaning.py        ← Data cleaning pipeline (samajhne ke liye)
├── car_price_model.pkl     ← Trained ML model (Voting Regressor)
├── olx_car_dataset_2025.csv← Raw OLX dataset
├── requirements.txt        ← Python dependencies
└── README.md               ← Yeh file
```

---

## 🚀 App Chalane Ka Tarika (How to Run)

### Step 1 — Dependencies install karo
```bash
pip install -r requirements.txt
```

### Step 2 — Streamlit app run karo
```bash
streamlit run app.py
```

Browser automatically `http://localhost:8501` par khul jayega.

---

## 🧹 Data Cleaning Summary (`data_cleaning.py`)

Yeh script samjhata hai ke raw OLX data ko model-ready data mein kaise convert kiya:

| Step | Column | Problem | Solution |
|------|--------|---------|----------|
| 1 | Price | `"Rs 18.50 Lacs"` (string) | Numeric PKR mein convert |
| 2 | Mileage | `"90,000 km"` / `"New"` (string) | Numeric km mein convert |
| 3 | Car_Age | Year column se age nahi thi | `2026 - Year` se calculate |
| 4 | Condition | `"Used"`/`"New"`/NaN (mixed) | Binary 0/1 mein convert |
| 5 | Brand/Model | 326 rows mein missing tha | Title se keyword matching se extract |
| 6 | Seats | 421 rows missing | Default 5 se fill |
| 7 | Price outliers | Unrealistic prices | 5 Lac – 1.5 Crore range rakhi |
| 8 | Mileage outliers | 5 lakh+ km cars | Max 3 lakh km limit |
| 9 | Encoding | Categorical columns | One-Hot Encoding |

### Data cleaning script run karo (output dekhne ke liye):
```bash
python data_cleaning.py
```

---

## 🤖 ML Model Details

**Best Model Used: Voting Regressor** (Ensemble of 4 models)

| Model | R² Score |
|-------|----------|
| Linear Regression | 0.70 |
| Polynomial + Ridge | 0.73 |
| Random Forest | 0.72 |
| XGBoost | 0.74 |
| Gradient Boosting | 0.73 |
| **Voting Regressor** | **0.76** ← Best |

**Top Features affecting price:**
1. `Car_Age` — Jitna purana, utna sasta
2. `Brand_Suzuki` — Market mein sabse zyada common
3. `Fuel_Petrol` — Petrol cars zyada hain dataset mein

---

## 📊 Supported Brands & Models

| Brand | Models |
|-------|--------|
| Toyota | Corolla, Yaris, Fortuner, Vitz, Aqua, Passo, Raize, Surf, Hilux, Prius |
| Suzuki | Alto, Cultus, Wagon R, Mehran, Swift, Bolan, Ravi, Liana, Baleno, Khyber |
| Honda | Civic, City, Vezel, N Wgn |
| Hyundai | Tucson, Sonata, Elantra, Santa Fe |
| Kia | Sportage, Sorento |
| Nissan | Dayz, Moco |
| Others | Any other brand |

---

## ⚠️ Important Notes

- Predictions are **estimates only** — actual market price vary kar sakti hai
- Model Pakistan (PKR) prices ke liye trained hai
- Dataset: 696 OLX listings (2025)
- Price range: PKR 5 Lac – 1.5 Crore

---

*Built with Streamlit · Scikit-learn · XGBoost*

# 🏠 SmartHome AI: House Price Prediction & Real Estate Analytics

SmartHome AI is an end-to-end data science project. It cleans real Bengaluru property listings, trains and compares four regression models, and serves the best one in a **Streamlit** web app. Customers get a price estimate. An admin can sign in to a private dashboard to review the predictions made.

> ⚠️ Predictions are **estimates based on past asking prices**, not valuations or guaranteed selling prices.

## Features

**Customer site**
- Simple form: location, area type, availability, area (sq ft), bedrooms, bathrooms, balconies
- Estimated price in ₹ lakh / crore, price per sq ft, property summary and a clear disclaimer
- Input validation with readable error messages. The app never shows a made-up prediction when something fails.

**Admin dashboard** (login required)
- **Dashboard Overview:** total, average, highest and lowest predicted price, last-7-days activity, charts and the latest predictions
- **Prediction Analytics:** price distribution, predictions over time, area vs price, price by location, bedroom distribution
- **Prediction History:** search, date filter, sorting, pagination, details view, delete with confirmation
- **Admin Settings:** profile, change password, log out
- Indian number formatting (for example ₹61,08,848). All figures come from the real database, with empty states when there is no data.

## Screenshots

| Dashboard Overview | Prediction Analytics |
|---|---|
| ![Dashboard](assets/dashboard.png) | ![Analytics](assets/analytics.png) |

| Prediction History | Predict House Price |
|---|---|
| ![History](assets/history.png) | ![Predict](assets/predict.png) |

## Technologies
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit · Joblib · SQLite · Pytest

## Dataset
| Item | Details |
|---|---|
| Name | Bengaluru House Data |
| Original source | Kaggle: https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data |
| Download used by this project | GitHub mirror: https://raw.githubusercontent.com/GowriShaju/House-Price-Prediction-EDA/main/Bengaluru_House_Data.csv |
| Coverage | Bengaluru, India |
| Size | 13,320 raw rows, 9 columns (11,973 rows after cleaning) |
| Columns | area_type, availability, location, size, society, total_sqft, bath, balcony, price |
| Currency | Indian Rupees. `price` is in **lakh ₹** (1 lakh = ₹100,000) |

The dataset file is **not included** in this repository. The commands below download it. I could not independently verify the dataset's licence, so please check the Kaggle page before redistributing it. The dataset has no property age, furnishing, parking, floor or amenities, so the app does not ask for them.

## How it works
1. **Cleaning:** remove duplicates, parse area ranges and units, extract BHK, drop invalid rows, and remove implausible outliers (under 300 sq ft per bedroom, bathrooms above bedrooms + 2, price outside ₹1,000 to ₹60,000 per sq ft).
2. **Pipeline:** feature engineering, imputation, scaling and one-hot encoding all live **inside** one scikit-learn pipeline. They are fitted on training data only, which prevents data leakage.
3. **Models:** Linear Regression, Decision Tree, Random Forest and Gradient Boosting, trained on log(price).
4. **Selection:** 5-fold cross-validation on the training set picks the model. The 20% test set is used only to report MAE, RMSE and R².
5. **App:** the saved pipeline is loaded once (cached) and used for every prediction. Each prediction is saved to a local SQLite database.

## Project structure
```
SmartHome-AI/
├── app.py                    # Streamlit app (customer site + routing)
├── requirements.txt
├── .streamlit/config.toml    # theme
├── src/
│   ├── data_preprocessing.py # loading, cleaning, pipeline parts
│   ├── train_model.py        # train, compare, save model and reports
│   ├── evaluate_model.py     # metrics helpers
│   ├── predict.py            # validation + prediction
│   ├── history.py            # SQLite prediction history
│   ├── auth.py               # single-admin login (hashed password)
│   ├── services.py           # cached model / report loaders
│   └── admin_dashboard.py    # admin pages
├── notebooks/house_price_eda.ipynb
├── tests/                    # pytest tests
├── assets/                   # screenshots
├── data/                     # created when you run the commands below
├── models/                   # created by training
└── reports/                  # created by training
```

## Installation and run (Windows PowerShell)

```powershell
git clone https://github.com/mounika03122006/SmartHome-AI.git
cd SmartHome-AI
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Then, one command at a time:

```powershell
python -m src.data_preprocessing --download   # download the dataset
python -m src.train_model                     # train and evaluate (about 2 to 5 minutes)
python -m src.auth                            # create the admin account (asks for a username and password)
streamlit run app.py                          # open http://localhost:8501
```

- Click **🔒 Admin** on the home page to sign in to the dashboard.
- The admin password is chosen by you and stored only as a salted hash in `admin_credentials.json`, which is git-ignored. Use `python -m src.auth --reset` to replace it.
- This is simple local protection and is not production-grade security.

## Tests
```powershell
python -m pytest -q
```
The tests cover cleaning, validation, model loading, prediction reliability, the history database (including SQL-injection safety), and admin authentication. Some tests use the trained model, so run the training command first.

## Results
Gradient Boosting was selected by cross-validation. Results from one run (scikit-learn 1.8, seed 42):

| Model | CV RMSE (lakh) | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| **Gradient Boosting (selected)** | 77.9 | 31.9 | 83.4 | 0.631 |
| Random Forest | 83.6 | 32.8 | 84.9 | 0.618 |
| Decision Tree | 93.1 | 35.5 | 93.9 | 0.533 |
| Linear Regression | 92.8 | 34.1 | 97.8 | 0.493 |

Re-running on another machine (scikit-learn 1.9.1) gave a selected-model R² of 0.636 and MAE of 31.8 lakh. Small differences between library versions are expected. Your own numbers are in `reports/model_comparison.csv`.

## Limitations
- Covers **Bengaluru only**, and uses **asking prices**, not final sale prices
- No property age, furnishing, parking, floor or amenities, and no date column, so market changes over time are not modelled
- R² is about 0.63, so the model is a rough guide. Errors are larger for expensive properties.
- Most locations have few listings and are grouped as "rare", so estimates there are less reliable
- The admin login is basic local authentication

## Future improvements
- Hyperparameter tuning and prediction intervals ("likely between ₹X and ₹Y")
- More recent, multi-city data with richer features
- SHAP-based explanations
- Online deployment with stronger authentication

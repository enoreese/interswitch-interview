from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

DB = pd.read_csv('data/credit_hold_out_data.csv', index_col=0)

RF_MODEL = joblib.load('models/rf_credit_pipeline.sav')
CAT_COLS = ['age_group', 'Payment_Behaviour', 'Month', 'Credit_Mix', 'Occupation']
NUM_COLS = ['Monthly_Inhand_Salary',
            'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
            'Delay_from_due_date', 'Outstanding_Debt',
            'Credit_Utilization_Ratio', 'imputed_age', 'imputed_ccl',
            'imputed_monthly_balance', 'total_emi_per_month_transform',
            'num_loan_transform', 'debt_to_income_ratio', 'loan_to_income_ratio',
            'total_financial_obligations', 'total_credit_utilization',
            'delayed_payment_impact', 'salary_deviation',
            'banking_to_credit_ratio', 'annual_savings_estimate',
            'credit_card_limit_utilization', 'debt_utilization_interaction',
            'interest_bank_accounts_interaction']
SELECTED_COLUMNS = ['Interest_Rate', 'Delay_from_due_date', 'Outstanding_Debt', 'debt_to_income_ratio',
                    'total_financial_obligations', 'debt_utilization_interaction', 'interest_bank_accounts_interaction',
                    'age_group_gt50', 'Credit_Mix_Good', 'Credit_Mix_Standard']


def preprocess_data(x_data):
    x_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    x_data.fillna(0, inplace=True)

    x_data = pd.get_dummies(x_data[CAT_COLS+NUM_COLS], drop_first=True)

    scaler = MinMaxScaler()
    x_data = pd.DataFrame(scaler.fit_transform(x_data), columns=x_data.columns)

    x_data = x_data[SELECTED_COLUMNS]
    return x_data


def get_prediction(customer_id):
    customer_df = DB[DB.Customer_ID == customer_id]
    preprocessed = preprocess_data(customer_df)
    if preprocessed.shape[0] == 0:
        return None

    prediction = RF_MODEL.predict(preprocessed)
    proba = RF_MODEL.predict(preprocessed)
    return prediction, proba


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/credit-score")
def predict_credit_score(customer_id: str):
    if not customer_id:
        return {
            "status": "error", "message": "No institution passed"
        }

    if DB[DB.Customer_ID == customer_id].shape[0] == 0:
        return {
            "status": "error", "message": "No DB entry found"
        }

    prediction, proba = get_prediction(customer_id)
    if not prediction:
        return {
            "status": "error", "message": "Cannot get prediction"
        }

    return {
        "status": "success", "customer_id": customer_id, "prediction": prediction, "proba": proba
    }

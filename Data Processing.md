# Data Preprocessing Documentation
## 1. Overview
This document outlines the preprocessing steps applied to the raw credit risk dataset for the purpose of developing predictive classification models for credit risk.

## 2. Raw Data
The raw data consists of two primary datasets:

- Patient Data (`credit_score.csv`):
    - Columns: `Customer_ID`, `Month`, `Name`, `Annual_Income`, `Interest_Rate`...

## 3. Credit Score Data Processing
#### Observations
- Noticed that customers were missing data for some loans, to handle the inconsistency, I am filling the missing value 
with the majority for that customer. This applied to the following columns: occupation and Monthly_Inhand_Salary.
  ```python
  def fill_customer_majority(cust_df: pd.DataFrame, column: str, nan_str: str = 'nan'):
      cust_df[column] = cust_df[column].apply(lambda x: str(x))
      column_list = [col for col in cust_df[column].to_list() if col != nan_str]
      column_list = list(map(str, column_list))
  
      return max(set(column_list), key=column_list.count)
  ```

#### Handling Inconsistencies in Other columns
- Used regex to clean number features with alphabet or special characters (`Annual income`, `Changed_Credit_Limit`, `Outstanding_Debt`, etc.).
- Replaced feature values that have low count with new value (`Interest rate`, `Monthly_Balance`).
- Imputed missing categories for some categorical features with the label "unknown" for records with empty information (`Credit Mix`).

## 4. Output
The final cleaned dataset is saved and ready for further analysis and model development.

## 5. Conclusion
The preprocessing steps ensure data consistency, handle missing values, and create a unified dataset for developing predictive models.
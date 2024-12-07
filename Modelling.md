# Credit Score Classification Documentation
## 1. Overview
This document outlines the steps taken to build predictive models for classifying credit score. The models are designed to predict credit score for each customer based on historical patterns and relevant features.

## 2. Feature Engineering
### 2.1 Binning Features
- `Age Group` - Categorize age into bins

### 2.2 Interaction Features
- `debt_utilization_interaction` - Combines utilization ratio with debt
- `interest_bank_accounts_interaction` - Interaction between interest rates and banking activity.

### 2.3 Other Financial Features
- `debt_to_income_ratio` - Debt relative to customer's income
- `loan_to_income_ratio` - Ratio between monthly loan payment and salary.
- `total_financial_obligations` - Total amount owed.

## 3. Data Splitting
- Created Hold out set for the month of `August`.
- Split the remaining data into training and validation sets
- Maintain stratification based on the 'credit_score' target column
- Use 80% of the data for training

## 4. Model Training and Evaluation
### 4.1 Classification
- Create a baseline model and check performance.
- Apply various feature selection techniques and selected the best 10 features for each technique.
- Define a list of classification models including Logistic Regression, XGBoost, and RandomForest
- For each model and each set of selected features, preprocess the data using:
  - One-hot encode categorical columns
  - Standard scale numerical columns
- Train the model on the training set and evaluate on the testing set

### 4.2 Evaluation
- Classification report showing metrics such as Precision, Recall and F1 Score are reported for both training and testing sets

## 5. Hyperparameter Tuning
- Define hyperparameter space for XGBoost/Random Forest using hyperopt library
- Implement objective function for hyperparameter tuning
- Use hyperopt's Tree Parzen Estimator (TPE) algorithm for optimization

## 6. Model Explainability and Interpretation
### 6.1 SHAP Values Overview
SHAP (SHapley Additive exPlanations) values provide a way to explain the output of machine learning models. 
They allocate contributions of each feature to the prediction for each instance, providing insights into how each feature influences the model's output. 
Here, we use SHAP values to interpret the prediction model.

### 6.2 SHAP Summary Plot
The SHAP summary plot serves as a replacement for the traditional bar chart of feature importance. 
It offers a comprehensive view of each feature's importance and the range of effects over the dataset. 
The plot indicates which features are most influential and how changes in their values impact the predictions.

In our model:

- The primary factor influencing credit score is the `Credit Mix (Good and Standard)`.
- The next most powerful indicator is the `interest rate`.

### 6.3 SHAP Dependence Plot
SHAP dependence plots illustrate how the model output varies with changes in a specific feature's value. 
Each dot in the plot represents an individual instance in the dataset, allowing for the observation of trends and interactions.

### 6.4 Interpretation and Insights
- **Interest rate Patterns**: Customers with lower interest rates are more likely to have better credit scores.

## 7. Conclusion
This documentation provides a comprehensive overview of the process involved in developing predictive credit score models. It covers feature engineering, feature selection, model selection, and hyperparameter tuning. The resulting models are ready for deployment and can be integrated into the system for real-time credit score predictions.


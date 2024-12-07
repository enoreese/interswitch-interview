# Credit Scoring Model Using Alternative Data

- Presenter: Osasu Usen
- Organization: FinDataTech
- Role: Head, Data Science

## 1. Overview of the Case Study

### Objective

Develop a credit scoring model leveraging alternative data for underbanked populations, particularly in rural areas,
where traditional credit data is unavailable.

### Data Sources

- Transaction Histories: Mobile money wallets.
- Telco Data: Airtime purchases and mobile usage patterns.
- Social Media Activity: Whatsapp, Facebook, tweets, etc.
- Repayment Behavior: Past loan repayment trends (this can be extracted from bank account statement or 
 open banking sources).
- Demographic Data: Age, location, gender, profession.

## 2. Approach

### Key Steps

1. Data Preparation: Clean and preprocess data.
2. Feature Engineering: Extract meaningful features.
3. Model Selection: Choose appropriate algorithms.
4. Evaluation Metrics: Measure performance rigorously.
5. Bias and Fairness: Ensure model impartiality.
6. Production Deployment: Deploy the solution via APIs.
7. Regulatory Compliance: Adhere to local standards.

## 3. Data Preparation

### Challenges

- Missing and inconsistent data.
- High dimensionality and noise.

### Steps Taken

- Handling Missing Data: Imputation (mean, median, or mode).
    ```python
    demograh_data.profession.loc[demograh_data.profession.isna()] = "missing"
    demograh_data.profession = demograh_data.profession.apply(lambda x: x.lower())
    ```
    ```python
    repayment_amount_imputer = KNNImputer(n_neighbors=2, weights="uniform")
    
    repayment_amount_imputer.fit(imp_data)
    repayment_history["imputed_repayment_amount"] = repayment_amount_imputer.transform(imp_data)[:, 0]
    ```
- Dealing with Noise: Filtering and outlier detection.
- Data Formatting: Standardizing formats across sources.
    ```
  transactions.timestamp = pd.to_datetime(transactions.timestamp)
  repayment_history.status = repayment_history.status.apply(lambda x: x.lower().replace(" ", "_"))
    ```
- Data Integration: Aggregate and merge disparate datasets.
  ```python
  agg_trans_data = []
  customer_group = transactions.groupby(['customer_id'])
  for i, x in enumerate(customer_group.groups):
      group = customer_group.get_group(x)
      group.sort_values(by=["timestamp"], inplace=True)
  
      deposits = group[group.transaction_type=='Deposit']
      purchases = group[group.transaction_type=='Purchase']
      withdrawal = group[group.transaction_type=='Withdrawal']
  
      min_date = group.timestamp.min()
      max_date = group.timestamp.max()
  
      obj = {
          'customer': x,
          'min_date': min_date,
          'max_date': max_date,
          'total_count': len(group),
          'no_deposits': deposits.shape[0],
          'no_purchases': purchases.shape[0],
          'no_withdrawal': withdrawal.shape[0],
          'avg_amount': group['imputed_amount'].mean(),
          'avg_amount_deposits': deposits['imputed_amount'].mean(),
          'avg_amount_purchases': purchases['imputed_amount'].mean(),
          'avg_amount_withdrawal': withdrawal['imputed_amount'].mean(),
      }
      agg_trans_data.append(obj)
  ```

## 4. Feature Engineering

### Example Features

1. Transaction Histories: Spending patterns, transaction frequency.
2. Telco Data: Call/SMS volume, airtime recharges.
3. Social Media Activity: Sentiment analysis, activity patterns, network size.
4. Repayment Behavior: Payment consistency, delays, loan institutions.
5. Demographics: Age buckets, geolocation insights.

### Feature Importance

- Correlation analysis.
- Model based techniques
- Recursive feature elimination techniques.

## 5. Model Selection

### Algorithms Considered

- Logistic Regression (baseline).
- Random Forest (interpretability, handling missing data).
- XGBoost (handling large datasets, accuracy).

### Tuning Strategies

- Hyperparameter tuning with GridSearch/RandomSearch/hyperOpt.
- Cross-validation to avoid overfitting.

## 6. Evaluation Metrics

### Key Metrics

- ROC-AUC: Overall performance, Useful for assessing the trade-off between true positive and false positive rates.
- Precision: The proportion of predicted customers classified as “creditworthy” that are actually “creditworthy”.
- Recall: The proportion of actual positive cases that are correctly identified by the model.
- F1-Score: Harmonic mean of precision and recall.
- Bias Metrics: Difference in Positive Proportions of True Labels, Conditional Demographic Disparity in Labels, Recall, precision, and accuracy differences, etc.

### Addressing Bias and Fairness

1. Data Audits: Identify underrepresented groups.
2. Bias Mitigation: Fair/balanced resampling.
3. Fairness Testing: Evaluate predictions across demographics.

## 7. Production Deployment

### Deployment Strategy

* **Integration Steps**:
  * ***Model Serialization***: Serialize the trained model (e.g., using joblib or pickle) for easy deployment. A model registry can also be created using tools like MlFlow to house different models and their versions.
  * ***API Development***: Create an API endpoint for the Client's system to interact with the model. This can be achieved using frameworks like Flask or FastAPI.
  * ***Data Preprocessing***: Implement the same preprocessing steps used during model training to ensure consistency.
  * ***Real-time Data Feed***: Establish a mechanism for the Client's system to provide real-time data to the deployed model.
* **Scalability Considerations**:
  * ***Containerization***: Consider containerizing the API using Docker for scalability and portability.
  * ***SageMaker Endpoints***: Deploy machine learning models as endpoints on Amazon SageMaker.

### Monitoring Plan

* **Model Performance Metrics**:
  * ***Define Metrics***: Clearly define relevant metrics for monitoring model performance, in this case, precision, recall, AUC.
  * ***Collect Logs***: Using tools like prometheus, collect model outputs and inputs
  * ***Dashboard Integration***: Integrate a monitoring dashboard (e.g., using tools like Grafana or Kibana) to visualize real-time performance metrics.
* **Anomaly Detection**:
  * ***Thresholds***: Set thresholds for each performance metric to trigger alerts if they deviate significantly.
  * ***Automated Alerts***: Implement automated alerting systems (e.g., through email or messaging services) to notify stakeholders when anomalies are detected.
* **Data Drift Detection**:
  * ***Define Baselines***: Establish baselines for input data distribution during training.
  * ***Monitoring Tools***: Utilize tools like AWS data monitor or custom scripts to detect data drift.
  * ***Scheduled Checks***: Regularly check for data drift and retrain the model if significant drift is detected.
* **Model Updates**:
  * ***Scheduled Retraining***: Plan periodic model updates based on the availability of new data.
  * ***Automated Deployment***: Automate the deployment process to streamline updates without disrupting the system.

## 8. Regulatory Compliance

### Key Considerations

- Data Privacy: Anonymization, encryption.
- Audit Trails: Logging for transparency.

## 9. Conclusion

### Next Steps

- Prototype development.
- Stakeholder feedback.

## 10. Q&A

Questions?
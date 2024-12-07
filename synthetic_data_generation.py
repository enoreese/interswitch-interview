import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta

faker = Faker()
Faker.seed(42)
np.random.seed(42)

# Parameters
n_customers = 10000
n_transactions = 68000
n_telco_events = 33000
n_social_events = 27200
n_repayments = 18350

if __name__ == '__main__':
    # Helper function for timestamps
    def random_date(start, end, n, bias=None):
        start_u = int(start.timestamp())
        end_u = int(end.timestamp())
        dates = []
        for _ in range(n):
            date = datetime.fromtimestamp(random.randint(start_u, end_u))
            if bias and random.random() < 0.7:
                date += timedelta(days=random.randint(-bias, bias))
            dates.append(date)
        return dates


    # Assign customer profiles with demographic patterns and age-profession dependency
    customer_profiles = []
    for i in range(n_customers):
        is_rural = random.choice([True, False])
        age = random.randint(18, 60)
        gender = random.choice(["Male", "Female"])
        location = faker.state()

        # Assign profession based on age
        if age < 25:
            profession = np.random.choice(["Trader", "Teacher", "Engineer"], p=[0.5, 0.3, 0.2])
        elif 25 <= age < 40:
            profession = np.random.choice(["Farmer", "Trader", "Engineer", "Healthcare Worker"], p=[0.2, 0.4, 0.3, 0.1])
        else:
            profession = np.random.choice(["Farmer", "Teacher", "Trader"], p=[0.5, 0.3, 0.2])

        # Determine credit score based on age, profession, and rural status
        if is_rural:
            credit_score = random.randint(300, 700)  # Rural customers have generally lower scores
        else:
            credit_score = random.randint(400, 850)

        if profession in ["Engineer", "Trader"]:
            credit_score += random.randint(-20, 50)  # Higher potential for urban traders and engineers
        elif profession == "Farmer":
            credit_score += random.randint(-50, 20)  # Farmers in rural areas typically have lower scores

        credit_score = min(max(credit_score, 300), 850)  # Ensure score is within valid range

        # Determine high activity based on profession and age
        if profession in ["Trader", "Engineer"]:
            high_activity = age < 40 and random.random() < 0.7  # Younger traders and engineers likely to be active
        elif profession == "Farmer":
            high_activity = random.random() < 0.3  # Farmers less likely to show high activity
        else:
            high_activity = random.random() < 0.5  # Moderate activity for other professions

        customer_profiles.append({
            "customer_id": f"ID_{i}",
            "age": age,
            "gender": gender,
            "location": location,
            "profession": profession,
            "is_rural": is_rural,
            "credit_score": credit_score,
            "high_activity": high_activity,
        })

    # Save demographic data
    demographics_df = pd.DataFrame(customer_profiles)
    demographics_df.loc[demographics_df.sample(frac=0.05).index, "profession"] = None  # Missing data
    demographics_df.loc[demographics_df.sample(frac=0.074).index, "age"] = None  # Missing data
    demographics_df.to_csv("data/demographics.csv", index=False)

    # ----- Generate Transactions Data -----
    transactions = []
    for _ in range(n_transactions):
        customer = random.choice(customer_profiles)
        amount = np.random.normal(240000 if customer["high_activity"] else 106000, 47900)
        while amount < 0:
            amount = np.random.normal(240000 if customer["high_activity"] else 106000, 47900)  # Avoid negative amounts

        transactions.append({
            "transaction_id": f"TX_{len(transactions)}",
            "customer_id": customer["customer_id"],
            "timestamp": random_date(datetime(2023, 1, 1), datetime(2024, 1, 1), 1)[0],
            "amount": round(amount, 2),
            "transaction_type": np.random.choice(["Deposit", "Withdrawal", "Purchase"]),
        })
    transactions_df = pd.DataFrame(transactions)
    transactions_df.loc[transactions_df.sample(frac=0.0523).index, "amount"] = None  # Missing amounts
    transactions_df.to_csv("data/transactions.csv", index=False)

    # ----- Generate Telco Data -----
    telco_data = []
    for _ in range(n_telco_events):
        customer = random.choice(customer_profiles)
        value = np.random.normal(700 if customer["high_activity"] else 300, 100)
        while value < 0:
            value = np.random.normal(700 if customer["high_activity"] else 300, 100)

        telco_data.append({
            "event_id": f"TELCO_{len(telco_data)}",
            "customer_id": customer["customer_id"],
            "timestamp": random_date(datetime(2023, 1, 1), datetime(2024, 1, 1), 1)[0],
            "event_type": np.random.choice(["Airtime Purchase", "Data Usage", "Call"]),
            "value": round(value, 2),
        })
    telco_df = pd.DataFrame(telco_data)
    telco_df.loc[telco_df.sample(frac=0.0334).index, "value"] = None
    telco_df.to_csv("data/telco.csv", index=False)

    # ----- Generate Social Media Data -----
    social_media = []
    for _ in range(n_social_events):
        customer = random.choice(customer_profiles)
        sentiment = np.random.normal(0.7 if customer["high_activity"] else -0.2, 0.5)
        social_media.append({
            "event_id": f"SOCIAL_{len(social_media)}",
            "customer_id": customer["customer_id"],
            "timestamp": random_date(datetime(2023, 1, 1), datetime(2024, 1, 1), 1)[0],
            "platform": np.random.choice(["Whatsapp", "Facebook", "Twitter"]),
            "event_type": np.random.choice(["Message Sent", "Post Created", "Comment"]),
            "sentiment_score": round(sentiment, 2),
        })
    social_media_df = pd.DataFrame(social_media)
    social_media_df.loc[social_media_df.sample(frac=0.07).index, "sentiment_score"] = None
    social_media_df.to_csv("data/social_media.csv", index=False)

    # ----- Generate Repayment History Data -----
    repayments = []
    for _ in range(n_repayments):
        customer = random.choice(customer_profiles)
        repayment_amount = np.random.normal(500000 if customer["credit_score"] > 600 else 50000, 10000)
        repayments.append({
            "repayment_id": f"REPAY_{len(repayments)}",
            "customer_id": customer["customer_id"],
            "loan_id": f"LOAN_{random.randint(1, n_repayments+1000)}",
            "timestamp": random_date(datetime(2023, 1, 1), datetime(2024, 1, 1), 1)[0],
            "repayment_amount": round(max(0., repayment_amount), 2),
            "status": np.random.choice(["On Time", "Late", "Missed"], p=[0.7, 0.2, 0.1]),
        })
    repayments_df = pd.DataFrame(repayments)
    repayments_df.loc[repayments_df.sample(frac=0.1).index, "repayment_amount"] = None
    repayments_df.to_csv("data/repayment_history.csv", index=False)

    # ----- Summary -----
    print("Datasets created with patterns:")
    print("- demographics.csv")
    print("- transactions.csv")
    print("- telco.csv")
    print("- social_media.csv")
    print("- repayment_history.csv")

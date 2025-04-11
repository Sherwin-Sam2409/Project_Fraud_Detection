import pandas as pd
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["fraud_detection"]
transactions_col = db["transactions"]

# Load CSV file
file_path = "financial_transactions.csv"  
df = pd.read_csv(file_path)

# Convert timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Store data in MongoDB
transactions_col.insert_many(df.to_dict(orient="records"))

print("CSV data loaded successfully into MongoDB!")

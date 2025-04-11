import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv("financial_transactions.csv")

# Print total number of transactions
total_transactions = len(df)
print(f"Total number of transactions: {total_transactions}")

# Plot histogram of transaction types
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Type', hue='Type', palette='viridis', legend=False)
plt.title("Transaction Type Distribution - Histogram")
plt.xlabel("Transaction Type")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot pie chart of transaction types
type_counts = df['Type'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(type_counts, labels=type_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(type_counts)))
plt.title("Transaction Type Distribution - Pie Chart")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()
plt.show()

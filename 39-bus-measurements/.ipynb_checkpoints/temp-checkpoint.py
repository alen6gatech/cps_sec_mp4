import pandas as pd

# Load the CSV file into a DataFrame
df_train_features = pd.read_csv("train_features.csv")

# Display the first few rows to verify
print(df_train_features.head())

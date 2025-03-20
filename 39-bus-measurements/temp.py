#Part1A
import pandas as pd

# Load the CSV file into a DataFrame
df_train_features = pd.read_csv("train_features.csv")

print(df_train_features.shape)

#Part1B
import pandas as pd

# Load the CSV file into a DataFrame
df_train_features = pd.read_csv("train_features.csv")

# Get statistics
desc = df_train_features.describe()

zero_columns_count = ((desc.loc["mean"] == 0) & 
                      (desc.loc["std"] == 0) & 
                      (desc.loc["max"] == 0)).sum()

print(f"Number of columns that are all zeros: {zero_columns_count}")

#Part1C
import pandas as pd

# Load the CSV file into a DataFrame
df_train_features = pd.read_csv("train_features.csv")

# Identify columns with constant values
constant_columns = df_train_features.nunique() == 1  # Checks if a column has only one unique value

# Exclude all-zero columns
non_zero_constant_columns = constant_columns & (df_train_features.sum() != 0)

# Count number of non-zero constant columns
constant_column_count = non_zero_constant_columns.sum()

print(f"Number of non-zero constant columns: {constant_column_count}")

#Part1D
import pandas as pd

# Load the CSV file into a DataFrame
df_train_features = pd.read_csv("train_features.csv")

# Get standard deviations from describe()
desc_stats = df_train_features.describe()

# Find the column with the highest std deviation
max_std_column = desc_stats.loc["std"].idxmax()
max_std_value = desc_stats.loc["std"].max()

print(f"Column with highest std deviation: {max_std_column} (Std Dev = {max_std_value})")

#Part2A
import pandas as pd

# Load the CSV file into a DataFrame
df_train_labels = pd.read_csv("train_labels.csv")

# Get statistics
desc = df_train_labels.describe()

# Find the column with the highest mean
max_mean_column = desc.loc["mean"].idxmax()
max_mean_value = desc.loc["mean"].max()

# Find the column with the lowest mean
min_mean_column = desc.loc["mean"].idxmin()
min_mean_value = desc.loc["mean"].min()

print(f"Column with highest mean: {max_mean_column} (Mean = {max_mean_value})")
print(f"Column with lowest mean: {min_mean_column} (Mean = {min_mean_value})")


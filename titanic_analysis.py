# Import the pandas library for data handling 

import pandas as pd

# Step 1: Load the dataset from a CSV file
df = pd.read_csv("train.csv")  # Make sure 'train.csv' is in the same folder

# Step 2: View the first 5 rows of the dataset
print("1. First 5 rows of the dataset:")
print(df.head())  # Shows a preview of how the data looks

# Step 3: Check the shape (number of rows and columns)
print("\n2. Shape of the dataset (rows, columns):")
print(df.shape)  # Tells how many entries and features there are

# Step 4: Check for missing (null) values in each column
print("\n3. Missing values in each column:")
print(df.isnull().sum())  # Helps us know where data is incomplete

# Step 5: Show summary statistics for numerical columns
print("\n4. Summary statistics of numerical data:")
print(df.describe())  # Mean, min, max, etc., for Age, Fare, etc.

# Step 6: Get information about each column (type and non-null count)
print("\n5. Info about columns (data types and nulls):")
print(df.info())  # Shows which columns are text, numbers, etc.

# Step 7: List all column names
print("\n6. Column names in the dataset:")
print(df.columns.tolist())  # Lists all column headers

# Step 8: Quick check – how many survived vs not survived

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
"""
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title('Survival Count (0 = Died, 1 = Survived)')
plt.show()
"""
#9. Survival by gender
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(['Died', 'Survived'])
plt.show()
"""

# 10. Histogram - Age distribution
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='Age', bins=30, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 11. Heatmap of missing values
plt.figure(figsize=(10,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()
"""
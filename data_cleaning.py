import pandas as pd

# Load the data from the correct file
df = pd.read_csv("train.csv")

# Step 1: Check missing values
print("Missing values before cleaning:\n", df.isnull().sum())

# Fill missing 'Age' with median value
df['Age'].fillna(df['Age'].median(), inplace=True)

# Drop 'Cabin' column (too many missing values)
df.drop('Cabin', axis=1, inplace=True)

# Drop rows where 'Embarked' is missing
df.dropna(subset=['Embarked'], inplace=True)

# Confirm all missing values are handled
print("\nMissing values after cleaning:\n", df.isnull().sum())
# Step 2: Encode 'Sex' column
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Step 3: Encode 'Embarked' column
embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
df['Embarked'] = df['Embarked'].map(embarked_mapping)

# Display first 5 rows to confirm
print("\nAfter encoding:\n", df[['Sex', 'Embarked']].head())

# Step 4: Drop unnecessary columns
df.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

# Show the final cleaned data
print("\nFinal cleaned dataset preview:\n", df.head())
# Save the cleaned data to a new CSV file
df.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned data saved to 'titanic_cleaned.csv'")

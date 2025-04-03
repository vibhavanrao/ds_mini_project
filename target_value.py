import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
 
# Load the dataset
file_path = 'final_merged_data.csv'  # Update with your file path
df = pd.read_csv(file_path)
 
# Display the first few rows of the dataset
print("Dataset Preview:")
print(df.head())
 
# Data Cleaning
# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
 
# Fill missing values (example: filling numerical columns with 0 and categorical with mode)
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = df.select_dtypes(include=['object']).columns
 
df[numerical_columns] = df[numerical_columns].fillna(0)
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])
# Feature Engineering: Create churn_utilization based on seat_utilization and agent_utilization
df['churn_utilization'] = (
    (df['seat_utilization'] < 0.2) |
    (df['agent_utilization'] < 0.2)
).astype(int)
 
# Group by 'id' and calculate the average of current_arr and future_arr for each id
grouped_df = df.groupby('id').agg({
    'current_arr': 'mean',
    'future_arr': 'mean',
    'churn_utilization': 'max'  # If any row for the id has churn_utilization = 1, set it to 1
}).reset_index()
 
# Add a churn column based on the condition avg(future_arr) < avg(current_arr)
grouped_df['churn'] = (grouped_df['future_arr'] < grouped_df['current_arr']).astype(int)
 
# Merge the churn column back into the original dataset
df = df.merge(grouped_df[['id', 'churn']], on='id', how='left')
 
# Drop intermediate columns if not needed
df = df.drop(columns=['churn_utilization'])
 
# Display the updated dataset with the churn column
print("\nUpdated Dataset with Churn Column:")
print(df.head())
 
# Save the updated dataset to a new CSV file
output_file_path = 'updated_dataset_with_churn.csv'  # Update with your desired file path
df.to_csv(output_file_path, index=False)
 
print(f"\nUpdated dataset with 'churn' column has been saved to: {output_file_path}")
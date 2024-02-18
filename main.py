#Team 12 1st Deliverable
#1st Member: Zain Baig
#PSID: 1919288

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load COVID-19 data from your CSV file
covid_data = pd.read_csv('country_wise_latest.csv')

# Display the first few rows of the dataframe
print("Original COVID-19 Data:")
print(covid_data.head())

# Data Cleaning (replace 'your_column_names' with actual column names)
# For example, dropping rows with missing values
covid_data_cleaned = covid_data.dropna(subset=['New deaths'])

# Feature Selection using Recursive Feature Elimination (RFE)
X = covid_data_cleaned[['your_feature_columns']]
y = covid_data_cleaned['your_target_column']

# Using a RandomForestClassifier as an example estimator
estimator = RandomForestClassifier()
rfe = RFE(estimator, n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)

print("\nSelected Features after RFE:")
print(X_rfe[:5])  # Displaying the first few rows of the selected features

# Principal Component Analysis (PCA)
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nPrincipal Components after PCA:")
print(X_pca[:5])  # Displaying the first few rows of the principal components

# Plotting the original data and the PCA-transformed data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.scatterplot(x='your_feature_column1', y='your_feature_column2', hue='your_target_column', data=covid_data_cleaned)
plt.title('Original COVID-19 Data')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y)
plt.title('PCA-transformed COVID-19 Data')

plt.show()

#%%
# Project: Analyzing Used Car Listings on eBay Kleinanzeigen
# This project focuses on cleaning, exploring, and analyzing data from used car listings 
# on the German eBay website to derive insights and apply machine learning techniques.

#%%
# 1. Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# 2. Loading the Dataset
# Reading the CSV file containing used car listings into a Pandas DataFrame
autos = pd.read_csv("autos.csv", encoding="Latin-1")
# Displaying the first few rows to understand the dataset structure
autos.head()

#%%
# 3. Overview of the Dataset
# Getting a concise summary of the DataFrame
autos.info()

# Checking the shape of the dataset
autos_shape = autos.shape
print(f"The dataset contains {autos_shape[0]} rows and {autos_shape[1]} columns.")

#%%
# 4. Renaming Columns for Clarity
# Converting camelCase column names to snake_case for better readability
autos.columns = [
    'date_crawled', 'name', 'seller', 'offer_type', 'price', 'abtest',
    'vehicle_type', 'registration_year', 'gearbox', 'power_ps', 'model',
    'odometer', 'registration_month', 'fuel_type', 'brand',
    'unrepaired_damage', 'ad_created', 'nr_of_pictures', 'postal_code',
    'last_seen'
]
# Displaying the updated column names
autos.columns

#%%
# 5. Initial Data Exploration
# Describing the dataset to understand distributions and data types
autos.describe(include='all')

# Checking unique values in 'odometer' to understand its distribution
print(autos['odometer'].value_counts())

# Dropping columns that are not useful for analysis
autos.drop(columns=['seller', 'offer_type', 'nr_of_pictures'], inplace=True)

#%%
# 6. Data Cleaning: Converting Text Columns to Numeric
# Cleaning 'price' and 'odometer' columns by removing text and converting to numeric
autos['price'] = autos['price'].str.replace("$", "").str.replace(",", "").astype(int)
autos['odometer'] = autos['odometer'].str.replace("km", "").str.replace(",", "").astype(int)
autos.rename({"odometer": "odometer_km"}, axis=1, inplace=True)

# Displaying the cleaned data
autos.head()

#%%
# 7. Analyzing 'price' and 'odometer_km' Distributions
# Exploring basic statistics of price and odometer
print(autos['price'].describe())
print(autos['odometer_km'].describe())

# Visualizing price distribution
plt.figure(figsize=(10, 5))
sns.histplot(autos['price'], bins=30, kde=True)
plt.title("Price Distribution of Used Cars")
plt.xlabel("Price (in Euros)")
plt.ylabel("Frequency")
plt.show()

# Visualizing odometer_km distribution
plt.figure(figsize=(10, 5))
sns.histplot(autos['odometer_km'], bins=30, kde=True)
plt.title("Odometer Reading Distribution of Used Cars")
plt.xlabel("Odometer (in km)")
plt.ylabel("Frequency")
plt.show()

#%%
# 8. Filtering Out Unrealistic Values
# Removing cars with zero or unrealistic prices
autos = autos[autos["price"].between(1000, 1000000)]

# Filtering out unrealistic odometer values (if any)
autos['odometer_km'].value_counts().sort_index(ascending=False)

#%%
# 9. Cleaning Registration Year Data
# Investigating registration year to remove outliers
autos["registration_year"].describe()

# Filtering unrealistic registration years
autos = autos[autos['registration_year'].between(1900, 2016)]
print(autos["registration_year"].value_counts(normalize=True).sort_index())

#%%
# 10. Analyzing Prices by Brand
# Getting the top brands with a market share of over 5%
autos_brand = autos['brand'].value_counts(normalize=True)
brand_top20 = autos_brand[autos_brand > 0.05].index

# Aggregating average price for each of the top brands
agg_data = {}
for brand in brand_top20:
    brand_data = autos[autos['brand'] == brand]
    avg_price = brand_data['price'].mean()
    agg_data[brand] = int(avg_price)

# Displaying aggregated data
agg_data

#%%
# 11. Visualizing Price by Brand
# Creating a DataFrame for easier visualization
price_brand_df = pd.DataFrame(agg_data.items(), columns=['Brand', 'Average_Price'])

# Plotting the average price by brand
plt.figure(figsize=(12, 6))
sns.barplot(x='Average_Price', y='Brand', data=price_brand_df.sort_values('Average_Price', ascending=False))
plt.title("Average Price of Used Cars by Brand")
plt.xlabel("Average Price (in Euros)")
plt.ylabel("Brand")
plt.show()

#%%
# 12. Analyzing Mileage by Brand
# Aggregating average mileage for each of the top brands
mileage_top = {}
for brand in brand_top20:
    brand_data = autos[autos['brand'] == brand]
    avg_mileage = brand_data['odometer_km'].mean()
    mileage_top[brand] = int(avg_mileage)

# Creating a DataFrame for average mileage
mileage_brand_df = pd.DataFrame(mileage_top.items(), columns=['Brand', 'Average_Mileage'])

# Merging average price and mileage data
combined_df = price_brand_df.merge(mileage_brand_df, on='Brand')

# Displaying the combined DataFrame
combined_df

#%%
# 13. Correlation Analysis
# Analyzing the correlation between price and mileage
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average_Mileage', y='Average_Price', data=combined_df)
plt.title("Correlation between Average Price and Mileage by Brand")
plt.xlabel("Average Mileage (in km)")
plt.ylabel("Average Price (in Euros)")
plt.show()

# Checking the correlation coefficient
correlation = combined_df['Average_Price'].corr(combined_df['Average_Mileage'])
print(f"The correlation coefficient between Average Price and Average Mileage is: {correlation:.2f}")

#%%
# 14. Machine Learning: Predicting Car Prices
# Preparing data for machine learning
# Encoding categorical variables
autos_encoded = pd.get_dummies(autos, columns=['brand', 'fuel_type', 'gearbox'], drop_first=True)

# Defining features and target variable
X = autos_encoded.drop(['price'], axis=1)
y = autos_encoded['price']

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%%
# 15. Training a Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")





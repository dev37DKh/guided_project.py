#%%
# Project: Storytelling Data Visualization on Exchange Rates
# This project aims to visualize and analyze exchange rate data between the Euro and the US Dollar from 1999 to 2022.
# We will explore trends, rolling averages, and significant events that may have influenced exchange rates.

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

# Setting the plotting style
style.use('seaborn')

#%%
# Step 1: Load the dataset
exchange_rates = pd.read_csv("euro-daily-hist_1999_2022.csv")

# Display the first few rows of the dataset
print(exchange_rates.head())

# Display the last few rows of the dataset
print(exchange_rates.tail())

# Get a summary of the dataset
print(exchange_rates.info())

# Analyzing the dataset
# We have 6177 rows and 41 columns, with some null values present in the dataset.
# The dataset includes various types of data, including object types and float64.

#%%
# Step 2: Data Cleaning
# Renaming columns for better accessibility and understanding
exchange_rates.rename(columns={
    '[US dollar ]': 'US_dollar',
    'Period\\Unit:': 'Time'
}, inplace=True)

# Converting 'Time' column to datetime format
exchange_rates['Time'] = pd.to_datetime(exchange_rates['Time'])

# Sorting values by 'Time'
exchange_rates.sort_values('Time', inplace=True)

# Resetting the index for easier access
exchange_rates.reset_index(drop=True, inplace=True)

# Creating a new DataFrame with relevant columns
euro_to_dollar = exchange_rates[['Time', 'US_dollar']]

# Filtering out any rows where 'US_dollar' is not a valid number
euro_to_dollar = euro_to_dollar[euro_to_dollar["US_dollar"] != "-"]

# Converting 'US_dollar' to float for numerical analysis
euro_to_dollar['US_dollar'] = euro_to_dollar['US_dollar'].astype(float)

# Displaying the cleaned DataFrame
print(euro_to_dollar.head())

#%%
# Step 3: Adding a Rolling Mean
# Calculating a 30-day rolling mean for smoothing the data
euro_to_dollar['rolling_mean'] = euro_to_dollar['US_dollar'].rolling(30).mean()

# Displaying the DataFrame with the rolling mean
print(euro_to_dollar.head())

#%%
# Step 4: Data Visualization
# Plotting the exchange rate over time
plt.figure(figsize=(14, 7))
plt.plot(euro_to_dollar['Time'], euro_to_dollar['US_dollar'], label='Daily Exchange Rate', color='blue', alpha=0.5)
plt.plot(euro_to_dollar['Time'], euro_to_dollar['rolling_mean'], label='30-Day Rolling Mean', color='orange', linewidth=2)
plt.title('Euro to US Dollar Exchange Rate (1999 - 2022)')
plt.xlabel('Year')
plt.ylabel('Exchange Rate (US Dollar)')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%%
# Step 5: Insights and Analysis
# Analyzing trends over the years and notable events that might have influenced exchange rates
# - Identify major economic events such as the 2008 financial crisis, the COVID-19 pandemic, etc.
# - Discuss fluctuations in exchange rates and potential causes.

# Conclusion: The visualization highlights the overall trend of the Euro against the US Dollar over time, revealing key patterns and fluctuations that could be correlated with major global events.

# Future steps: 
# - Investigate more granular data points such as specific events, economic policies, or geopolitical events affecting exchange rates.
# - Consider incorporating machine learning techniques for predictive analysis on future exchange rates.




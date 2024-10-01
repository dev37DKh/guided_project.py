#%% 
# Project: Heavy Traffic Indicators Analysis on I-94
# This project aims to analyze factors contributing to heavy congestion on the I-94 highway, specifically focusing on the westbound traffic. 
# We will explore various indicators, including time of day, day of the week, and weather conditions, to determine their impact on traffic volume.

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

#%% 
# Step 1: Load the dataset into a Pandas DataFrame
traffic = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
# Display the first few rows of the dataset
traffic.head()

# Display the last few rows of the dataset
traffic.tail()

# Get a summary of the dataset
traffic.info()

#%% 
# Step 2: Analyze Traffic Volume
# Visualizing the distribution of traffic volume using a histogram
plt.figure(figsize=(10, 5))
plt.hist(traffic['traffic_volume'], bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Traffic Volume')
plt.xlabel('Traffic Volume')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# Statistical description of traffic volume
traffic_volume_description = traffic['traffic_volume'].describe()
print(traffic_volume_description)

#%% 
# Step 3: Convert 'date_time' to datetime format for further analysis
traffic['date_time'] = pd.to_datetime(traffic['date_time'])

# Visualize traffic volume over time
plt.figure(figsize=(10, 5))
traffic.plot(x='date_time', y='traffic_volume', title='Traffic Volume Over Time', legend=False)
plt.xlabel('Date Time')
plt.ylabel('Traffic Volume')
plt.show()

#%% 
# Step 4: Segment data into daytime and nighttime
# Daytime data (7 AM to 7 PM)
day_data = traffic[(traffic['date_time'].dt.hour >= 7) & (traffic['date_time'].dt.hour < 19)]
# Nighttime data (7 PM to 7 AM)
night_data = traffic[(traffic['date_time'].dt.hour < 7) | (traffic['date_time'].dt.hour >= 19)]

# Visualizing traffic volume during day and night
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(day_data['traffic_volume'], bins=50, color='orange', alpha=0.7)
plt.title('Traffic Volume During Day')
plt.xlabel('Traffic Volume')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(night_data['traffic_volume'], bins=50, color='purple', alpha=0.7)
plt.title('Traffic Volume During Night')
plt.xlabel('Traffic Volume')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

#%% 
# Step 5: Analyze traffic volume by month
day_data['month'] = day_data['date_time'].dt.month
monthly_traffic = day_data.groupby('month').mean()

# Plotting average traffic volume by month
plt.figure(figsize=(10, 5))
monthly_traffic['traffic_volume'].plot(kind='bar', color='skyblue')
plt.title('Average Traffic Volume by Month')
plt.xlabel('Month')
plt.ylabel('Average Traffic Volume')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
plt.show()

#%% 
# Conclusion: Monthly traffic analysis indicates a steady increase from January to March, followed by a plateau until June. 
# A significant drop occurs in July, with a consistent volume from August to October, and a decline observed from November to December.

#%% 
# Step 6: Analyze traffic volume by day of the week
day_data['dayofweek'] = day_data['date_time'].dt.dayofweek
weekly_traffic = day_data.groupby('dayofweek').mean()

# Plotting average traffic volume by day of the week
plt.figure(figsize=(10, 5))
weekly_traffic['traffic_volume'].plot(kind='bar', color='lightgreen')
plt.title('Average Traffic Volume by Day of the Week')
plt.xlabel('Day of the Week (0=Mon, 6=Sun)')
plt.ylabel('Average Traffic Volume')
plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], rotation=45)
plt.show()

#%% 
# Conclusion: Traffic volume is relatively stable during weekdays, with higher values ranging from 4800 to 5250. 
# Conversely, weekends exhibit lower traffic volume, averaging between 3500 and 4000.

#%% 
# Step 7: Analyze traffic volume by hour on weekdays and weekends
day_data['hour'] = day_data['date_time'].dt.hour
business_days = day_data[day_data['dayofweek'] <= 4]  # Weekdays
weekend = day_data[day_data['dayofweek'] >= 5]  # Weekend

# Grouping by hour to analyze traffic volume
traffic_by_hour_weekdays = business_days.groupby('hour').mean()
traffic_by_hour_weekend = weekend.groupby('hour').mean()

# Plotting average traffic volume by hour
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(traffic_by_hour_weekdays['traffic_volume'], marker='o', color='coral')
plt.title("Traffic Volume on Weekdays")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Traffic Volume")
plt.xticks(range(24))
plt.ylim([1500, 6500])

plt.subplot(1, 2, 2)
plt.plot(traffic_by_hour_weekend['traffic_volume'], marker='o', color='gold')
plt.title("Traffic Volume on Weekends")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Traffic Volume")
plt.xticks(range(24))
plt.ylim([1500, 6500])

plt.tight_layout()
plt.show()

#%% 
# Conclusion: Heavy traffic peaks around 7 AM and 4 PM on weekdays. 
# During weekends, a noticeable increase occurs from 12 PM to 4 PM.

#%% 
# Step 8: Explore weather conditions affecting traffic volume
# Correlation analysis of weather conditions and traffic volume
correlation_matrix = traffic.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Traffic Volume and Weather Variables')
plt.show()

# Visualizing traffic volume against temperature
plt.figure(figsize=(8, 5))
plt.scatter(data=traffic, x='temp', y='traffic_volume', alpha=0.5, color='blue')
plt.title('Traffic Volume vs. Temperature')
plt.xlabel('Temperature (F)')
plt.ylabel('Traffic Volume')
plt.grid()
plt.show()

# Grouping traffic volume by main weather condition
traffic_weather_main = day_data.groupby('weather_main').mean()
plt.figure(figsize=(10, 5))
traffic_weather_main['traffic_volume'].plot(kind='barh', color='salmon')
plt.title('Average Traffic Volume by Main Weather Condition')
plt.xlabel('Average Traffic Volume')
plt.ylabel('Weather Condition')
plt.show()

# Grouping traffic volume by weather description
traffic_weather_desc = day_data.groupby('weather_description').mean()
plt.figure(figsize=(10, 8))
traffic_weather_desc['traffic_volume'].plot(kind='barh', color='teal')
plt





#!/usr/bin/env python
# coding: utf-8

# ###  Python Task 2

# ### Q1

# In[1]:


import pandas as pd

def calculate_distance_matrix(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Create an empty matrix with unique IDs as both row and column indices
    unique_ids = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids)
    distance_matrix = distance_matrix.fillna(0)

    # Fill the matrix with cumulative distances along known routes
    for _, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] += row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] += row['distance']

    return distance_matrix

# Example usage with your CSV file
#resulting_matrix = calculate_distance_matrix('C:/Users/LENOVO/Downloads/dataset-3.csv')
#print(resulting_matrix)


# In[ ]:





# ### Q2

# In[2]:


import pandas as pd

def unroll_distance_matrix(distance_matrix):
    # Initialize an empty list to store the unrolled data
    unrolled_data = []

    # Iterate through the rows of the distance matrix
    for id_start, row in distance_matrix.iterrows():
        # Iterate through the columns
        for id_end, distance in row.items():
            # Exclude same id_start to id_end combinations
            if id_start != id_end:
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

# Example usage with the resulting_matrix from Question 1
#unrolled_distance_df = unroll_distance_matrix(resulting_matrix)
#print(unrolled_distance_df)


# In[ ]:





# ### Q3

# In[3]:


import pandas as pd

def calculate_distance_matrix(csv_file):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Create an empty matrix with unique IDs as both row and column indices
    unique_ids = sorted(set(df['id_start'].unique()) | set(df['id_end'].unique()))
    distance_matrix = pd.DataFrame(index=unique_ids, columns=unique_ids)
    distance_matrix = distance_matrix.fillna(0)

    # Fill the matrix with cumulative distances along known routes
    for _, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] += row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] += row['distance']

    return distance_matrix

# Example usage with your CSV file path
#resulting_matrix = calculate_distance_matrix('C:/Users/LENOVO/Downloads/dataset-3.csv')
#print(resulting_matrix)


# In[ ]:





# ### Q4

# In[4]:


import pandas as pd
import numpy as np

def calculate_toll_rate(distance_matrix):
    # Copy the input DataFrame to avoid modifying the original
    toll_df = distance_matrix.copy()

    # Flatten the values in the distance matrix
    flat_values = toll_df.values.flatten()

    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Create a new DataFrame for toll rates
    toll_rates_df = pd.DataFrame()

    # Iterate through rate_coefficients and calculate toll rates
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        toll_rates_df[vehicle_type] = flat_values * rate_coefficient

    # Concatenate the toll rates DataFrame with the original DataFrame
    resulting_toll_df = pd.concat([toll_df, toll_rates_df], axis=1)

    return resulting_toll_df

# Example usage with the resulting_distance_matrix from Question 2
#resulting_distance_matrix = calculate_distance_matrix('C:/Users/LENOVO/Downloads/dataset-3.csv')
#resulting_toll_df = calculate_toll_rate(resulting_distance_matrix)
#print(resulting_toll_df)


# In[ ]:





# ### Q5

# In[5]:


import pandas as pd
import numpy as np
import datetime

def calculate_time_based_toll_rates(df):
    # Make a copy of the input DataFrame to avoid modifying the original
    result_df = df.copy()

    # Define time ranges for weekdays
    weekday_morning_range = pd.to_datetime("00:00:00").time(), pd.to_datetime("10:00:00").time()
    weekday_afternoon_range = pd.to_datetime("10:00:00").time(), pd.to_datetime("18:00:00").time()
    weekday_evening_range = pd.to_datetime("18:00:00").time(), pd.to_datetime("23:59:59").time()

    # Define discount factors for different time intervals
    discount_factors = {
        "weekday_morning": 0.8,
        "weekday_afternoon": 1.2,
        "weekday_evening": 0.8,
        "weekend_all_day": 0.7
    }

    # Create new columns for start_day, start_time, end_day, and end_time
    result_df["start_day"] = result_df["start_time"].dt.day_name()
    result_df["start_time"] = result_df["start_time"].dt.time
    result_df["end_day"] = result_df["end_time"].dt.day_name()
    result_df["end_time"] = result_df["end_time"].dt.time

    # Iterate over each row and apply discount factors based on time ranges
    for index, row in result_df.iterrows():
        if row["start_day"] in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
            if weekday_morning_range[0] <= row["start_time"] <= weekday_morning_range[1]:
                result_df.loc[index, result_df.columns[5:]] *= discount_factors["weekday_morning"]
            elif weekday_afternoon_range[0] <= row["start_time"] <= weekday_afternoon_range[1]:
                result_df.loc[index, result_df.columns[5:]] *= discount_factors["weekday_afternoon"]
            elif weekday_evening_range[0] <= row["start_time"] <= weekday_evening_range[1]:
                result_df.loc[index, result_df.columns[5:]] *= discount_factors["weekday_evening"]
        elif row["start_day"] in ["Saturday", "Sunday"]:
            # Apply constant discount factor for weekends
            result_df.loc[index, result_df.columns[5:]] *= discount_factors["weekend_all_day"]

    return result_df


# In[ ]:





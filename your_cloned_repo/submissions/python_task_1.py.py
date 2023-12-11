#!/usr/bin/env python
# coding: utf-8

# ### Python Task 1

# # Q1

# In[44]:


import pandas as pd
import numpy as np

def generate_car_matrix(dataset='C:/Users/LENOVO/Downloads/dataset-1.csv'):
    
    # Read the CSV file as DataFrame
    df = pd.read_csv(dataset)

    # Creating  a pivot table using id_1 as index, id_2 as columns, and car as values
    car_matrix = pd.pivot_table(df, values='car', index='id_1', columns='id_2', fill_value=0)

    # Setting diagonal values as 0
    np.fill_diagonal(car_matrix.values, 0)

    return car_matrix

# Example usage
#result_matrix = generate_car_matrix()
#print(result_matrix)


# In[ ]:





# ### Q2

# In[7]:


import pandas as pd

def get_type_count(df):
    # Add a new categorical column 'car_type' based on values of the column 'car'
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts

# Example usage
#dataset_path = 'C:/Users/LENOVO/Downloads/dataset-1.csv'  # Replace with the actual path to your dataset
#df = pd.read_csv(dataset_path)

#result = get_type_count(df)
#print(result)


# ### Q3

# In[8]:


import pandas as pd

def get_bus_indexes(df):
    # Calculate the mean value of the 'bus' column
    mean_bus = df['bus'].mean()

    # Identify indices where 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

# Example usage
#dataset_path = 'C:/Users/LENOVO/Downloads/dataset-1.csv'  # Replace with the actual path to your dataset
#df = pd.read_csv(dataset_path)

#result = get_bus_indexes(df)
#print(result)


# ### Q4

# In[9]:


import pandas as pd

def filter_routes(df):
    # Calculate the average of the 'truck' column for each route
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' values is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes
    selected_routes.sort()

    return selected_routes

# Example usage
#dataset_path = 'C:/Users/LENOVO/Downloads/dataset-1.csv'  # Replace with the actual path to your dataset
#df = pd.read_csv(dataset_path)

#result = filter_routes(df)
#print(result)


# ### Q5

# In[10]:


import pandas as pd

def multiply_matrix(input_matrix):
    # Deep copy the input matrix to avoid modifying the original DataFrame
    modified_matrix = input_matrix.copy()

    # Apply the logic to modify each value in the DataFrame
    modified_matrix = modified_matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

# Example usage
# Assuming 'result_matrix' is the DataFrame from Question 1
# result_matrix = generate_car_matrix()  # Replace with the actual DataFrame

#modified_result = multiply_matrix(result_matrix)
#print(modified_result)


# ### Q6

# In[1]:


import pandas as pd
from datetime import datetime, timedelta

def check_timestamp_completeness(df):
    # Combine startDay and startTime into a single datetime column
    df['start_datetime'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
    # Combine endDay and endTime into a single datetime column
    df['end_datetime'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')

    # Drop rows with NaT values
    df = df.dropna(subset=['start_datetime', 'end_datetime'])

    # Create a timedelta representing a full 24-hour period
    full_day_duration = timedelta(hours=24)

    # Create a DataFrame to store the results
    result_df = pd.DataFrame(index=df.set_index(['id', 'id_2']).index.unique())

    # Check if each (id, id_2) pair has incorrect timestamps
    result_df['incorrect_timestamps'] = df.groupby(['id', 'id_2']).apply(lambda group:
        not (group['start_datetime'].min() <= group['end_datetime'].max() and
             group['end_datetime'].max() - group['start_datetime'].min() >= full_day_duration)
    )

    return result_df['incorrect_timestamps']



# In[ ]:





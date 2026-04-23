import pandas as pd
import numpy as np
import os
import time

import matplotlib.pyplot as plt

from datetime import datetime


def get_dataframe_memory_usage(df):
    """
    Returns the total memory consumption of a Pandas DataFrame in megabytes (MB).
    """
    memory_usage = df.memory_usage(deep=True)
    """ The deep=True parameter is used to include the memory usage of any objects that are referenced by the DataFrame's columns, such as strings or other nested data structures. If you set deep=False, the method will return only the memory usage of the column itself, which may not be the total memory consumed by the DataFrame. """

    # sum the memory usage to get the total memory consumed
    total_memory = memory_usage.sum()

    # convert to megabytes
    total_memory_mb = total_memory / 1048576

    print(f"Total memory consumed by the DataFrame: {total_memory_mb:.2f} MB")


######################################################################

def check_for_missing_timestamp(df):
    # calculate the time differences between consecutive rows
    time_diffs = df.index[1:] - df.index[:-1]

    # count the frequency of each unique time difference
    counts_df = time_diffs.value_counts()

    # return the counts
    return counts_df.head()


def impute_missing_data(df):
    # sort the index in increasing order
    df = df.sort_index()

    imputed_df = df.reindex(range(df.index[0], df.index[-1]+60, 60),  method='pad')

    return imputed_df


def impute_missing_data(btc_df: pd.DataFrame) -> pd.DataFrame:
   
    new_index = range(btc_df.index[0], btc_df.index[-1] + 60, 60)
    """ range(btc_df.index[0],btc_df.index[-1]+60,60) creates a new range of values starting from the first timestamp in the index of btc_df and ending at the last timestamp plus 60 seconds, with a step of 60 seconds. This creates a new index with a minute-by-minute frequency. """

    imputed_btc_df = btc_df.reindex(new_index, method='pad')
    """ The missing values (i.e., the timestamps that are not present in the original index) are filled in with the pad method, which fills the missing values with the most recent available value. """

    return imputed_btc_df





def get_optimal_numeric_type(c_min: float, c_max: float, col_type: str) -> str:

    type_info = np.iinfo if col_type == 'int' else np.finfo
    for dtype in [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
        if col_type in str(dtype):
            if c_min > type_info(dtype).min and c_max < type_info(dtype).max:
                return dtype
    return None

""" Based on the data type and the range of values, the function determines the smallest possible data type that can accommodate the data without losing information. For example, if the data type is an integer and the range of values fits within the bounds of an int8 data type, the function converts the column data type to int8: """

def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:

    # Iterate through each column in the DataFrame
    df_copy = df.copy()
    for col in df_copy.columns:
        col_type = df_copy[col].dtype

        # Check if the data type is not an object (i.e., numeric type)
        if col_type != object:
            c_min, c_max = df_copy[col].min(), df_copy[col].max()
            col_type_str = 'int' if 'int' in str(col_type) else 'float'
            optimal_type = get_optimal_numeric_type(c_min, c_max, col_type_str)
            if optimal_type:
                df_copy[col] = df_copy[col].astype(optimal_type)
        # If the data type is an object, convert the column to a 'category' data type
        else:
            df_copy[col] = df_copy[col].astype('category')

    # Return the optimized DataFrame with reduced memory usage
    return df_copy


#################################################

def create_lagged_dataframe(df: pd.DataFrame, lag_values: list = [3, 2, 1]) -> pd.DataFrame:
    shifted_dfs = [df.shift(lag) for lag in lag_values]  # List comprehension to create a list of shifted DataFrames
    shifted_dfs.append(df)  # Append the original DataFrame to the list

    lag_df = pd.concat(shifted_dfs, axis=1)  # Concatenate the list of shifted DataFrames and the original DataFrame along the columns axis

    return lag_df

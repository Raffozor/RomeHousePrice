import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_sqlite_table(table_name: str, db_path: str) -> pd.DataFrame:
    """
    Reads data from a specified table in a SQLite database and returns the data as a pandas DataFrame.

    Args:
    table_name (str): The name of the table to read.
    db_path (str): The path of the SQLite database.

    Returns:
    pandas.DataFrame: A DataFrame containing all the data from the specified table.

    Example:
    db_path = "C:/Users/User/Database.db"
    table_name = "Customers"
    df = read_sqlite_table(table_name, db_path)
    """
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)

    # SQL query to select all data from the desired table
    query = f"SELECT * FROM {table_name}"

    # Use pandas to read data from the query and create a DataFrame
    df = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()

    # Return the DataFrame
    return df


def plot_value_percentages(df: pd.DataFrame, value: str) -> None:
    """
    Creates a horizontal bar chart that displays the percentage of values equal to a specified value
    in each column of a DataFrame.

    Args:
    df (pandas.DataFrame): The DataFrame to operate on.
    value (any data type): The value of which to display the percentages.

    Returns:
    None. Shows the horizontal bar chart of percentages of values equal to 'value' for each
    column of the DataFrame.
    :rtype: None
    """
    # Calculate the percentage of values equal to 'value' for each column of the DataFrame
    value_percentages = df.apply(lambda x: (x == value).sum() / len(df))

    # Create a new figure with size (10,6)
    plt.figure(figsize=(10, 6))

    # Create a bar chart of the percentages of values equal to 'value'
    value_percentages.plot(kind='barh')

    # Customize the chart
    plt.title(f"Percentage of '{value}' values per column")
    plt.xlabel('Column')
    plt.ylabel(f"Percentage of '{value}'")

    # Show the chart
    plt.show()

    return None


def replace_value_with_nan(df: pd.DataFrame, value):
    """
    Replaces a specified value with np.Nan for each column in a DataFrame.

    Args:
    df (pandas.DataFrame): the DataFrame to operate on.
    value (str or any other data type): the value to replace with np.nan.

    Returns:
    pandas.DataFrame: the DataFrame with the specified values replaced with np.nan.

    :param value: (str or any other data type): the value to replace with np.nan.
    :type df: pd.DataFrame
    """
    for col in df:
        df[col] = df[col].replace(value, np.nan)

    return df


def plot_nan_percentages(df):
    """
    Calculates the percentage of NaN values for each column in a DataFrame and plots it as a horizontal bar chart.

    Args:
    df (pandas.DataFrame): The DataFrame to operate on.

    Returns: None. Shows the horizontal bar chart of NaN percentages for each column in the DataFrame.

    """
    # Calculate the percentage of NaN values for each column in the DataFrame
    nan_percentages = df.isna().mean()

    # Create a new figure with size (10,6)
    plt.figure(figsize=(10, 6))

    # Create a horizontal bar chart of the NaN percentages
    nan_percentages.plot(kind='barh')

    # Customize the chart
    plt.title('Percentage of NaN values per column')
    plt.xlabel('Column')
    plt.ylabel('Percentage of NaN')

    # Show the chart
    plt.show()

    return None


def save_to_sql(path, df, table_name):
    # Connessione al database SQLite
    conn = sqlite3.connect(path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    # Chiudere la connessione al database
    conn.close()
    return None

def plot_bar(df,missing_value):
    nones = []
    for col in df.columns:
        nones.append((df[col]==missing_value).sum())
    nones = pd.DataFrame(nones,df.columns)
    nones = nones.sort_values(0, ascending=False)

    fig, ax = plt.subplots()

    columns = list(nones.index)
    counts = nones[0]

    ax.bar(columns, counts, label=columns)

    ax.set_ylabel(f'Number of {missing_value}')
    ax.set_title('Columns')
    x = np.arange(len(columns))
    width = 0.25
    ax.set_xticks(x + width, columns, rotation=-60)

    plt.show()
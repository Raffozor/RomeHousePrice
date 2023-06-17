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
    #plt.figure(figsize=(10, 6))

    # Create a horizontal bar chart of the NaN percentages
    nan_percentages.sort_values(ascending=False).plot(kind='bar')

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

def replace_missing_values(df,missing_value):

    for col in df.columns:
        df[col] = df[col].apply(lambda x: np.nan if x == missing_value else x)
    return df

def value_encoding(value,diz,train,sigma_proportion):
    if value not in diz.keys():
        mu = diz['missing_values'][0]
        sigma = diz['missing_values'][1]
        value_encoded = np.random.normal(mu, sigma*sigma_proportion, 1)[0]
    else:
        mu = diz[value][0]
        sigma = diz[value][1]
        value_encoded = np.random.normal(mu, sigma*sigma_proportion, 1)[0]
    return value_encoded

def mean_std_encoding(train, test, cols_to_encode, target, sigma_proportion, min_cat_n):

    encoding_diz = {}
    for col in cols_to_encode:
        train.loc[train.groupby(col)[col].transform('count').lt(min_cat_n), col] = np.nan
        diz = train.groupby(col).apply(lambda row: [row[target].mean(),row[target].std()]).to_dict()
        diz['missing_values'] = [train[target].mean(), train[target].std()]
        encoding_diz[col] = diz
        train[col] = train[col].apply(lambda row: value_encoding(row, diz, train, sigma_proportion))
        test[col] = test[col].apply(lambda row: value_encoding(row, diz, train, sigma_proportion))
    return  train, test, encoding_diz


class DataPreprocessor:

    def __init__(self, df):
        self.df = df

    def replace_with_nan(self, missing_values):
        self.df = self.df.replace(missing_values, np.nan)

    def train_test(self, train_size=0.8, seed=42):
        self.train = self.df.sample(frac=train_size, random_state=seed)
        self.test = self.df.drop(index=self.train.index)

    def plot_missing(self):

        cols_with_missing_train = pd.DataFrame()
        cols_with_missing_test = pd.DataFrame()

        for col in self.train.columns:
            if self.train[col].isna().sum() > 0:
                cols_with_missing_train[col] = self.train[col]

        for col in self.test.columns:
            if self.test[col].isna().sum() > 0:
                cols_with_missing_test[col] = self.test[col]
        if len(cols_with_missing_train) == 0 and len(cols_with_missing_test) == 0:
            print('No missing values')

        else:
            percentage_train = cols_with_missing_train.isna().sum() / len(cols_with_missing_train)
            percentage_train = percentage_train.sort_values(ascending=False)

            percentage_test = cols_with_missing_test.isna().sum() / len(cols_with_missing_test)
            percentage_test = percentage_test.sort_values(ascending=False)
            X = cols_with_missing_test.columns
            X_axis = np.arange(len(X))
            fig, ax = plt.subplots()
            plt.bar(X_axis - 0.2, percentage_train, 0.4, label='Train')
            plt.bar(X_axis + 0.2, percentage_test, 0.4, label='Test')

            ax.set_xticks(X_axis, X, rotation=-45)
            ax.set_xlabel("Variables")
            ax.set_ylim([0, 1])
            ax.set_ylabel('% of Missing Values')
            ax.set_title('Missing Values')
            ax.legend()
            plt.show()

        plt.show()

    def replace_missing(self):

        for col in self.train_nomiss.columns:
            if self.train_nomiss[col].isna().sum() > 0:

                if self.train[col].dtypes == 'object':
                    self.train_nomiss[col] = self.train_nomiss[col].replace(np.nan, self.train[col].mode()[0])
                    self.test_nomiss[col] = self.test_nomiss[col].replace(np.nan, self.train[col].mode()[0])

                if is_numeric_dtype(self.train[col]):
                    self.train_nomiss[col] = self.train_nomiss[col].replace(np.nan, round(self.train[col].mean(), 2))
                    self.test_nomiss[col] = self.test_nomiss[col].replace(np.nan, round(self.train[col].mean(), 2))

    def run_preprocessor(self, missing_values, train_size=0.8, seed=42):

        self.replace_with_nan(missing_values)
        self.split_train_test(train_size, seed)
        self.plot_missing()
        self.replace_missing()

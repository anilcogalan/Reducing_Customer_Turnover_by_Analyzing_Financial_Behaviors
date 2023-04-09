import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from EDA import num_cols,df
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df.isnull().sum() # NO MISSING VALUE!

##################################
# OUTLIERS ANALYSIS
##################################

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    """
       Calculate the upper and lower outlier thresholds for a given column in a Pandas DataFrame.

       Args:
           dataframe (pandas.DataFrame): The DataFrame containing the column to calculate outlier thresholds for.
           col_name (str): The name of the column to calculate outlier thresholds for.
           q1 (float, optional): The quantile to use for the lower threshold calculation. Defaults to 0.05.
           q3 (float, optional): The quantile to use for the upper threshold calculation. Defaults to 0.95.

       Returns:
           tuple: A tuple containing the lower and upper outlier thresholds for the specified column.

       Example:
           >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
           >>> outlier_thresholds(df, 'A')
           (-1.5, 7.5)
       """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    """
       Check if there are any outliers in a given column of a Pandas DataFrame.

       Args:
           dataframe (pandas.DataFrame): The DataFrame to check for outliers.
           col_name (str): The name of the column to check for outliers.

       Returns:
           bool: True if there are any outliers in the specified column, False otherwise.

       Example:
           >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 20]})
           >>> check_outlier(df, 'A')
           True
       """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    """
        Replace the outliers in a given column of a Pandas DataFrame with their corresponding outlier thresholds.

        Args:
            dataframe (pandas.DataFrame): The DataFrame containing the column to replace outliers in.
            variable (str): The name of the column to replace outliers in.
            q1 (float, optional): The quantile to use for the lower threshold calculation. Defaults to 0.05.
            q3 (float, optional): The quantile to use for the upper threshold calculation. Defaults to 0.95.

        Returns:
            None

        Example:
            >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 20]})
            >>> replace_with_thresholds(df, 'A')
            >>> df
               A
            0  1.00
            1  2.00
            2  3.00
            3  4.00
            4  6.75
        """
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col)) # 4 OUTLIER COLUMNS
    if check_outlier(df, col):
        replace_with_thresholds(df, col) # 0 OUTLIER COLUMNS

for col in num_cols:
    print(col, check_outlier(df, col))
##################################
# FEATURE INFERENCE
##################################

df = df.drop(columns = ['months_employed'])
df['personal_account_months'] = (df.personal_account_m + (df.personal_account_y * 12))
df[['personal_account_m', 'personal_account_y', 'personal_account_months']].head()
df = df.drop(columns = ['personal_account_m', 'personal_account_y'])

##################################
# ENCODING
##################################

# One Hot Encoding
df = pd.get_dummies(df)
df.columns
df = df.drop(columns = ['pay_schedule_semi-monthly'])

y = df["e_signed"]
X = df.drop("e_signed", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)


##################################
# Feature Scaling
##################################

sc_X = RobustScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2



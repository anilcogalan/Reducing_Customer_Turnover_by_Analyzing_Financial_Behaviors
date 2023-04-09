import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' %x)
pd.set_option('display.width', 500)

df = pd.read_csv('Financial-Data.csv')


##################################
# EFFICIENCY
##################################

def reduce_mem_usage(df, verbose=True):
    """
    Reduces memory usage of a pandas dataframe by downcasting numerical data types.

    Args:
    df (pandas dataframe): Input dataframe to reduce memory usage.
    verbose (bool): Optional parameter, if True prints the amount of memory saved.

    Returns:
    pandas dataframe: Modified dataframe with downcasted numerical data types.

    Example Usage:
    df = pd.read_csv('data.csv')
    reduced_df = reduce_mem_usage(df)

    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
reduce_mem_usage(df)

##################################
# GENERAL PICTURE
##################################

def check_df(dataframe, head=5, shape=True, types=True, head_tail=True, na=True, quantiles=True, duplicates=True):
    '''
    Prints various summary statistics and information about a given dataframe.

    Parameters:
    dataframe (pandas.DataFrame): The dataframe to analyze.
    head (int, optional): The number of rows to display for the head and tail of the dataframe.
    shape (bool, optional): Whether to print the shape of the dataframe.
    types (bool, optional): Whether to print the data types of each column.
    head_tail (bool, optional): Whether to print the head and tail of the dataframe.
    na (bool, optional): Whether to print the number of missing values in each column.
    quantiles (bool, optional): Whether to print the quantiles of each numeric column.
    duplicates (bool, optional): Whether to print the number of duplicate rows in the dataframe.

    Returns:
    None
    '''

    if shape:
        print("#" * 21, "Shape", "#" * 21)
        print(dataframe.shape)

    if types:
        print("#" * 21, "Types", "#" * 21)
        print(dataframe.dtypes)

    if head_tail:
        print("#" * 21, "Head", "#" * 21)
        print(dataframe.head(head))
        print("#" * 21, "Tail", "#" * 21)
        print(dataframe.tail(head))

    if na:
        print("#" * 21, "Missing Values", "#" * 21)
        print(dataframe.isnull().sum())

    if quantiles:
        print("#" * 21, "Quantiles", "#" * 21)
        num_cols = dataframe.select_dtypes(include=["int", "float"]).columns.tolist()
        print(dataframe[num_cols].describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    if duplicates:
        print("#" * 21, "Duplicates", "#" * 21)
        print(dataframe.duplicated().sum())

check_df(df, head=10)

##################################
# CAPTURE OF NUMERICAL AND CATEGORY VARIABLES
##################################

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

   It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
   Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                Class threshold value for numeric but categorical variables
        car_th: int, optinal
                Class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                Categorical view cardinal variable list

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of the 3 returned lists equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

##################################
# ANALYSIS OF CATEGORICAL VARIABLES
##################################

def cat_summary(dataframe, col_name, plot=False):
    """
        Summarizes a categorical variable in a Pandas DataFrame by computing the frequency count and ratio
        of each unique value in the specified column, and optionally plots a bar chart.

        Args:
            dataframe (pandas.DataFrame): The input DataFrame containing the categorical column to be summarized.
            col_name (str): The name of the categorical column to be summarized.
            plot (bool, optional): Whether to plot a bar chart of the frequency count. Defaults to False.

        Returns:
            None. The function prints the frequency count and ratio table to the console and, optionally,
            plots a bar chart.

        Raises:
            ValueError: If the specified column name is not present in the input DataFrame.
        """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df,col,plot=True)

##################################
# ANALYSIS OF NUMERICAL VARIABLES
##################################

def num_summary(dataframe, numerical_col, plot=False):
    """
    Generates a summary of a numerical column in a Pandas DataFrame by computing descriptive statistics
    such as count, mean, standard deviation, minimum, maximum, and selected percentiles. Optionally,
    plots a histogram of the column's distribution.

    Args:
        dataframe (pandas.DataFrame): The input DataFrame containing the numerical column to be summarized.
        numerical_col (str): The name of the numerical column to be summarized.
        plot (bool, optional): Whether to plot a histogram of the column's distribution. Defaults to False.

    Returns:
        None. The function prints the summary statistics table to the console and, optionally,
        plots a histogram.

    Raises:
        ValueError: If the specified column name is not present in the input DataFrame.
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)

##################################
# ANALYSIS OF NUMERICAL VARIABLES ACCORDING TO TARGET
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "e_signed", col)



##################################
# CORRELATION
##################################

num_df = df[num_cols]
num_df.corrwith(df.e_signed).plot.bar(
        figsize = (20, 10), title = "Correlation with E Signed", fontsize = 15,
        rot = 45, grid = True)
plt.tight_layout()
plt.show(block=True)

# Compute the correlation matrix
corr = num_df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.tight_layout()
plt.show(block=True)


from typing import Optional, List, Literal
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from scipy.stats import chi2_contingency, kstest


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import (
    SimpleImputer,
    KNNImputer,
)
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
)


#====================================================================#
#                               Classes                              #
#====================================================================#


class DropColumnTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop specified columns
        X_transformed = X.drop(columns=self.columns, axis=1)
        return X_transformed


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy: Literal['mean', 'median', 'most_frequent', 'fill_value'] = 'mean', columns: Optional[List[str]] = None):
        self.strategy = strategy
        self.columns = columns if columns is not None else []
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X, y=None):
        self.imputer.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.imputer.transform(X[self.columns])
        return X_transformed


class NaNIndicator(BaseEstimator, TransformerMixin):

    """
    NaNIndicator
    ------------
    This transformer identifies missing values in specified columns and creates new binary columns indicating the presence of missing values.

    Parameters
    ----------
    columns : list of strings
        The list of columns to check for missing values.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, None],
    ...     'B': [4, None, 6]
    ... })
    >>> nan_indicator = NaNIndicator(columns=['A', 'B'])
    >>> transformed_data = nan_indicator.fit_transform(data)
    >>> print(transformed_data)
         A    B  A_missing  B_missing
    0  1.0  4.0          0          0
    1  2.0  NaN          0          1
    2  NaN  6.0          1          0
    """

    def __init__(self, columns, default_suffix: str = '_missing'):
        self.columns = columns
        self.suffix = default_suffix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            new_column_name = column + self.suffix
            X_transformed[new_column_name] = X_transformed[column].isna().astype(int)
        return X_transformed


class CustomWhitespaceRemover(BaseEstimator, TransformerMixin):

    """
    CustomWhitespaceRemover
    -----------------------
    This transformer removes all whitespace characters from the specified columns in a DataFrame.

    Parameters
    ----------
    columns : list of strings
        The list of columns from which to remove whitespace.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': ['hello world', '   leading', 'trailing   '],
    ...     'B': ['no spaces', ' some  spaces ', '   whitespace   ']
    ... })
    >>> whitespace_remover = CustomWhitespaceRemover(columns=['A', 'B'])
    >>> transformed_data = whitespace_remover.fit_transform(data)
    >>> print(transformed_data)
                 A              B
    0   helloworld       nospaces
    1      leading     somespaces
    2     trailing     whitespace
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column] = X_transformed[column].astype(str).apply(lambda x: re.sub(r"\s+", "", x) if pd.notnull(x) else x)
        return X_transformed


class CustomLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            self.encoders[column] = LabelEncoder()
            self.encoders[column].fit(X[column])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            X_transformed[column] = self.encoders[column].transform(X[column])
        return X_transformed


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for column in self.columns:
            self.encoders[column] = OneHotEncoder(sparse_output=False)
            self.encoders[column].fit(X[[column]])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns:
            encoded = pd.DataFrame(
                self.encoders[column].transform(X[[column]]),
                columns=self.encoders[column].get_feature_names_out([column]),
                index=X.index,
            )
            X_transformed = pd.concat(
                [X_transformed.drop(columns=column), encoded], axis=1
            )
        return X_transformed

class CustomCategoryDivider(BaseEstimator, TransformerMixin):
    
    """
    CustomCategoryDivider
    ---------------------
    This transformer splits a categorical column into multiple columns based on a delimiter and assigns each part 
    before the delimiter as a category and the part after as an entry to that category.

    Parameters
    ----------
    column : str
        The name of the column to split.

    Attributes
    ----------
    columns_ : list
        The unique categories extracted from the column after splitting.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'Category_Description': ['Fruit - Apple', 'Vegetable - Carrot', 'Fruit - Banana']
    ... })
    >>> divider = CustomCategoryDivider(column='Category_Description')
    >>> transformed_data = divider.fit_transform(data)
    >>> print(transformed_data)
       Fruit   Vegetable
    0  Apple   NaN
    1  NaN     Carrot
    2  Banana  NaN
    """

    def __init__(self, column: str) -> None:
        self.column = column
        self.columns_ = None

    def fit(self, X, y=None):
        # Use regex to extract the category part
        categories = X[self.column].str.extract(r'^(.+?)\s*-\s*(.*)$', expand=True)[0].unique()
        self.columns_ = categories
        return self

    def transform(self, X):
        # Use regex to extract both parts
        split_columns = X[self.column].str.extract(r'^(.+?)\s*-\s*(.*)$', expand=True)

        unique_categories = self.columns_
        new_df = pd.DataFrame(index=X.index, columns=unique_categories)

        for i, row in split_columns.iterrows():
            category = row[0]
            description = row[1]
            if category in unique_categories:
                new_df.at[i, category] = description

        X_transformed = pd.concat([X, new_df], axis=1)
        
        X_transformed = X_transformed.drop(columns=[self.column])
        
        return X_transformed


class CustomStandardScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed


class CustomOutlierDetector(BaseEstimator, TransformerMixin):
    def __init__(self, data: pd.DataFrame, IQR_multiplier: float = 1.5):
        self.data = data
        self.IQR_multiplier = IQR_multiplier
    
    def detect_outliers_iqr(self, column):
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - self.IQR_multiplier * IQR
        upper_bound = Q3 + self.IQR_multiplier * IQR
        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        return outliers


# Custom transformer class to detect and remove outliers
class CustomOutlierRemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.numeric_cols = None
        self._outliers = None

    # This function identifies the numerical columns
    def fit(self, X, y=None):
        self.numeric_cols = X.select_dtypes(include=np.number).columns
        return self

    def transform(self, X):
        if self.numeric_cols is None:
            raise ValueError("Call 'fit' before 'transform'.")

        # Make a copy of numerical columns
        X_transformed = X.copy()

        z_scores = stats.zscore(X_transformed[self.numeric_cols])

        # Concat with non-numerical columns
        self._outliers = (abs(z_scores) > self.threshold).any(axis=1)
        return X_transformed[~self._outliers]

    @property
    def outliers(self):
        return self._outliers


# Custom transformer for Normalization
class CustomMinMaxScaler(BaseEstimator, TransformerMixin):

    def __init__(self, columns: List[str]) -> None:
        self.columns = columns
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.columns] = self.scaler.transform(X[self.columns])
        return X_transformed


#====================================================================#
#                              Functions                             #
#====================================================================#

def unique_column_content_check(df: pd.DataFrame):

    """
    Returns the unique values and their count for each categorical column in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check for unique values in its categorical columns.

    Returns
    -------
    dict
        A dictionary where keys are the column names and values are tuples containing the unique values and their count.

    Description
    -----------
    This function iterates over each column in the provided DataFrame. For columns with a data type of `object` or `str`, 
    it identifies and stores the unique values present in that column along with the count of these unique values. The 
    results are returned in the form of a dictionary.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': ['apple', 'banana', 'apple'],
    ...     'B': [1, 2, 3],
    ...     'C': ['dog', 'cat', 'dog']
    ... })
    >>> unique_column_content_check(data)
    {'A': (array(['apple', 'banana'], dtype=object), 2), 
     'C': (array(['dog', 'cat'], dtype=object), 2)}
    """

    store_unique = {}
    for column in df.columns:
        if (df[column].dtype == 'object') or (df[column].dtype == 'str'):  # Check if the column is categorical
            unique_values = df[column].unique()
            store_unique[column] = (unique_values, len(unique_values))

    return store_unique


def calculate_metrics(y_valid, y_pred, regressor_name):
    '''
    Returns if possible
    -------------------
    R^2 Score: [r2],
    Root Mean Squared Error (RMSE): [rmse],
    Mean Absolute Percentage Error (MAPE): [mape],
    Symmetric Mean Absolute Percentage Error (sMAPE): [smape],
    Mean Squared Logarithmic Error (MSLE): [msle],
    Root Mean Squared Logarithmic Error (RMSLE): [rmsle]
    '''
    
    mse = mean_squared_error(y_valid, y_pred)
    rmse = np.sqrt(mse)
    try:
        msle = mean_squared_log_error(y_valid, y_pred)
    except ValueError:
        msle = "Can't for negative output"
        rmsle = "Can't for negative output"
    else:
        rmsle = np.sqrt(msle)
    mape = np.mean(np.abs((y_valid - y_pred) / y_valid)) * 100
    smape = np.mean(2 * np.abs(y_pred - y_valid) / (np.abs(y_valid) + np.abs(y_pred))) * 100
    r2 = r2_score(y_valid, y_pred)
    
    metrics = {
        'R^2 Score': [r2],
        'Root Mean Squared Error (RMSE)': [rmse],
        'Mean Absolute Percentage Error (MAPE)': [mape],
        'Symmetric Mean Absolute Percentage Error (sMAPE)': [smape],
        'Mean Squared Logarithmic Error (MSLE)': [msle],
        'Root Mean Squared Logarithmic Error (RMSLE)': [rmsle]
    }
    
    return pd.DataFrame(metrics, index=[regressor_name]).T


def aggregate_metrics_and_create_heatmaps(metrics_list):

    """
    Aggregates a list of metric DataFrames into a single DataFrame and creates heatmaps for specified metrics.

    Parameters
    ----------
    metrics_list : list of pandas.DataFrame
        A list of DataFrames, each containing evaluation metrics for different models or datasets.

    Description
    -----------
    This function concatenates the provided list of metric DataFrames along the columns to create an aggregated DataFrame.
    It then generates heatmaps for several metrics, including 'R^2 Score', 'Root Mean Squared Error (RMSE)', 
    'Mean Absolute Percentage Error (MAPE)', 'Symmetric Mean Absolute Percentage Error (sMAPE)', 
    'Mean Squared Logarithmic Error (MSLE)', and 'Root Mean Squared Logarithmic Error (RMSLE)'. 
    Columns with the value "Can't for negative output" are excluded from the heatmaps. The heatmaps are 
    sorted in ascending order for all metrics except 'R^2 Score', which is sorted in descending order.

    The heatmaps are displayed with specific formatting:
    - Metrics are plotted with a 'crest' color map.
    - The Root Mean Squared Error (RMSE) metric uses integer formatting for annotations.
    - Other metrics use floating-point formatting with three decimal places for annotations.

    Examples
    --------
    >>> import pandas as pd
    >>> metrics_df1 = pd.DataFrame({
    ...     'R^2 Score': [0.9, 0.8],
    ...     'Root Mean Squared Error (RMSE)': [10, 12],
    ...     'Mean Absolute Percentage Error (MAPE)': [5.0, 6.0]
    ... }, index=['Model1', 'Model2'])
    >>> metrics_df2 = pd.DataFrame({
    ...     'R^2 Score': [0.85, 0.75],
    ...     'Root Mean Squared Error (RMSE)': [11, 13],
    ...     'Mean Absolute Percentage Error (MAPE)': [5.5, 6.5]
    ... }, index=['Model3', 'Model4'])
    >>> aggregate_metrics_and_create_heatmaps([metrics_df1, metrics_df2])

    This will generate and display heatmaps for each specified metric.
    """

    # Aggregate metrics into a single DataFrame
    aggregated_df = pd.concat(metrics_list, axis=1)

    # Define metrics to plot
    metrics = ['R^2 Score', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Percentage Error (MAPE)', 
               'Symmetric Mean Absolute Percentage Error (sMAPE)', 'Mean Squared Logarithmic Error (MSLE)', 
               'Root Mean Squared Logarithmic Error (RMSLE)']
    
    # Create heatmaps
    for metric in metrics:
        # Remove columns with 'Can't for negative output' if exist
        valid_series = aggregated_df.loc[metric].replace('Can\'t for negative output', np.nan).dropna()

        # Convert the series back to a DataFrame for heatmap plotting
        if metric != 'R^2 Score':
            valid_df = valid_series.to_frame().sort_values(by=metric, ascending=True)
        else:
            valid_df = valid_series.to_frame().sort_values(by=metric, ascending=False)

        plt.figure(figsize=(10, 6))
        if metric != 'Root Mean Squared Error (RMSE)':
            sns.heatmap(valid_df.astype(float), annot=True, cmap='crest',
                        cbar=True, linewidth=.5, fmt='.3f', annot_kws={"size": 15})
        else:
            sns.heatmap(valid_df.astype(float), annot=True, cmap='crest',
                        cbar=True, linewidth=.5, fmt='.0f', annot_kws={"size": 15})
        plt.title(f'{metric}')
        plt.show()


def corr_matrix_dataframe(df):
  
  """
    Computes the absolute correlation matrix for a DataFrame and returns it as a DataFrame sorted by correlation strength.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing numerical data.

    Returns
    -------
    pd.DataFrame
        A DataFrame representing the absolute correlation matrix sorted by correlation strength.

    Description
    -----------
    This function calculates the absolute correlation matrix for the input DataFrame, excluding the diagonal.
    It then sorts the correlations in descending order and returns them as a DataFrame with a single column 'correlation'.
    The resulting DataFrame contains pairs of features and their absolute correlation values, sorted by correlation strength.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, 5, 6],
    ...     'C': [7, 8, 9]
    ... })
    >>> corr_matrix = corr_matrix_dataframe(data)
    >>> print(corr_matrix)
          correlation
    A  B      1.000000
    B  C      1.000000
    A  C      1.000000
    """

  correlations = df.corr()
  np.fill_diagonal(correlations.values, np.nan)
  mask = np.tril(np.ones_like(correlations, dtype=bool))
  correlations[mask] = np.nan

  return pd.DataFrame(correlations.abs().unstack().dropna().sort_values(ascending=False), columns=['correlation'])


def perform_statistical_tests(names_of_tests: List[Literal['chi2 independence','ks normality']], df: pd.DataFrame, target_column: str, significance_level: float = 0.05):

    """
    Performs specified statistical tests on a DataFrame and prints the results.

    Parameters
    ----------
    names_of_tests : list
        A list of strings specifying the names of statistical tests to perform. Supported tests are 
        'chi2 independence' and 'ks normality'.
    df : pandas.DataFrame
        The input DataFrame containing the data to be tested.
    target_column : str
        The name of the target column in the DataFrame to be used in the tests.
    significance_level : float, optional
        The significance level to use for the tests (default is 0.05).

    Description
    -----------
    This function performs the specified statistical tests on the input DataFrame. It supports the 
    following tests:
    
    - 'chi2 independence': Performs the Chi-squared test of independence between each feature and the 
      target column. If the p-value indicates rejecting independence, it also calculates and prints 
      Cramer's V.
    - 'ks normality': Performs the Kolmogorov-Smirnov test for normality on each numerical feature.

    For 'chi2 independence', it prints the chi-squared statistic, p-value, and whether the independence 
    hypothesis is rejected. If the independence is rejected, it also prints Cramer's V statistic.

    For 'ks normality', it prints the KS statistic, p-value, and whether the normality hypothesis is 
    rejected.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'feature1': [1, 2, 3, 4, 5],
    ...     'feature2': [5, 6, 7, 8, 9],
    ...     'target': [0, 1, 0, 1, 0]
    ... })
    >>> perform_statistical_tests(['chi2 independence', 'ks normality'], data, 'target')
    
    Output for 'chi2 independence':
    Chi2 independency test for feature1 and target:
        chi2: X, pvalue: Y
        Do we reject independence? True/False
        Cramer's V association for feature1 and target:
        Cramer's V: Z

    Output for 'ks normality':
    KS normality test for feature1:
        ks_stat: X, pvalue: Y
        Do we reject normality? True/False
    """

    df_for_EDA = df.copy()
    df_target_EDA = df_for_EDA[target_column]
    df_columns_EDA = df_for_EDA.drop(target_column, axis=1)

    for name in names_of_tests:
        if name == 'chi2 independence':

            for column in df_columns_EDA.columns:
                contingency_table = pd.crosstab(df_columns_EDA[column], df_target_EDA)
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                print(
f'''
Chi2 independency test for {column} and {target_column}:
    chi2: {chi2:.0f}, pvalue: {p:.3f}
    Do we reject independence? {p <= significance_level}
''')

                if p <= significance_level:
                        summed_table = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2 / (summed_table * (min(contingency_table.shape) - 1)))
                        print(
f'''    Cramer's V association for {column} and {target_column}:
    Cramer's V: {cramers_v:.3f}
''')


        if name == 'ks normality':
                for column in df_for_EDA.describe().columns:
                        ks_statistic, p = kstest(df_for_EDA[column].dropna(), 'norm')
                        print(
f'''
KS normality test for {column}:
    ks_stat: {ks_statistic:.0f}, pvalue: {p:.3f}
    Do we reject normality? {p <= significance_level}
''')
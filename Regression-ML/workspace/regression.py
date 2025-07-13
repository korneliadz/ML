# %% [markdown]
# ___
# # ***WARNING*** run ONLY the cells you need

# %% [markdown]
# ___
# # Loading the data and libraries
# ___

# %%
%load_ext autoreload
%autoreload 2

# Base packages
import os
import sys
import re
from typing import Optional, List, Literal

while any(marker in os.getcwd() for marker in ['workspace_p2t3']):
    os.chdir("..")

# Append 'classes_and_functions' directory to sys.path
sys.path.append('classes_and_functions_p2t3')

# Get the current working directory
current_directory = os.getcwd()
current_directory

# %%
# Third-party packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline

# Custom packages
from classes_and_functions_p2t3.custom_functions_classes import (
    DropColumnTransformer,
    CustomImputer,
    CustomStandardScaler,
    CustomLabelEncoder,
    CustomOneHotEncoder,
    CustomOutlierDetector,
    CustomOutlierRemover,
    CustomMinMaxScaler,
    NaNIndicator,
    CustomCategoryDivider,
    CustomWhitespaceRemover,
    calculate_metrics,
    unique_column_content_check,
    aggregate_metrics_and_create_heatmaps,
    corr_matrix_dataframe,
    perform_statistical_tests,
)

# %%
raw_data = pd.read_csv('attachments_p2t3/TrainAndValid.csv')
raw_data.head()

# %% [markdown]
# ___
# # ***EDA*** Before Cleaning
# ___ 

# %%
raw_data.dtypes

# %%
print("\nSummary statistics for numerical variables:")
print(raw_data.describe())

# %%
raw_data.describe().columns

# %% [markdown]
# Just checking, whether we displayed all of the columns with int/float type

# %%
print(np.sort(raw_data['YearMade'].unique()))

# %% [markdown]
# We can already see, that there are outliers in YearMade feature, unless they are a trebuchet

# %%
print("\nSummary statistics for categorical variables:")
for column in raw_data.select_dtypes(include=['object']).columns:
    print("\n", column, ":")
    print(raw_data[column].value_counts())

# %%
unique_column_content_check(raw_data)

# %% [markdown]
# There are some features that are mistakingly put as a cathegorical data, when they should be numerical. Those are:
#  - Blade_Width 
#  - Tire_Size 
#  - Undercarriage_Pad_Width 
#  - Stick_Lenght

# %%
print('Number of nulls in different features')
raw_data.isnull().sum()

# %%
nulls_percentage = raw_data.isnull().mean() * 100
print('Percentage of nulls in each feature')
nulls_percentage

# %%
low_nulls = nulls_percentage[nulls_percentage <= 10]
print(f'Features with less than 10% of nulls:\n {low_nulls.index}')
print()

null_to_exterminate = nulls_percentage[nulls_percentage >= 40]
print(f'Features with more than 40% of nulls:\n {null_to_exterminate.index}')
print()
print(r'There is one feature with nulls between 10% and 40% - fiSecondaryDesc')

# %%
outlier_detector = CustomOutlierDetector(raw_data)

num_cols = ['SalesID', 'SalePrice', 'MachineID', 'ModelID', 'datasource',
       'auctioneerID', 'YearMade', 'MachineHoursCurrentMeter']

for var in num_cols:
    outliers = outlier_detector.detect_outliers_iqr(var)

print("Outliers identified using the IQR method:")
print(outliers[num_cols])

# %%
for var in num_cols:
    plt.figure(figsize=(8, 6))
    plt.hist(raw_data[var], bins=100, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()

# %%
correlation_matrix = raw_data[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, linewidth=.5, fmt='.2f', cmap='crest')
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
sns.pairplot(raw_data[raw_data.describe().columns], diag_kind='kde')
plt.show()

# %%
corr_matrix_dataframe(raw_data[raw_data.describe().columns])

# %% [markdown]
# Unfortunately, we can't say there is any kind of correlation between numerical features, other than between *SaleID* and *datasource*

# %%
for feature in num_cols:
    plt.scatter(raw_data.index, raw_data[feature], c='skyblue', s=3)
    plt.xlabel("index")
    plt.ylabel(feature)
    plt.title(f"{feature} and index")
    plt.show()

# %% [markdown]
# We can have some logical conclusions here:
# - Features like *SalesID* and *datasource* display some kind of linear trend with indices, which makes sense, as they should increase with every next purchase
# - *ModelID* and *MachineID* seem to have one too, but it is denser - it comes from the fact, that some Buldozers are similar, so they have ID's next to each other
# - *auctionerID* are stable at horizontal lines, as people who buy buldozers tend to repeat
# - In the rest, there is no relation to the number of index

# %% [markdown]
# ___
# # ***Cleaning*** and ***Preprocessing***
# ___

# %% [markdown]
# ### Cleaning

# %%
columns_to_terminate = null_to_exterminate.index
raw_data = raw_data.drop(columns=columns_to_terminate, axis=1) # Remove cols with a lot of nulls

raw_data.columns

# %%
raw_data.dtypes

# %%
for column in raw_data.columns: # Replacing ['None or Unspecified','#NAME?'] to nan
    raw_data[column] = raw_data[column].map(lambda x: np.nan if x in ['None or Unspecified','#NAME?'] else x)

for column in raw_data.columns: # Checking if ['None or Unspecified','#NAME?'] were replaced to nan
    if raw_data[column].dtype == 'object':  # Check if the column is categorical
        unique_values = raw_data[column].unique()
        if 'None or Unspecified' in unique_values or '#NAME?' in unique_values:
            print('replacing failed')


# %%
nulls_percentage_after = raw_data.isnull().mean() * 100
print('Percentage of nulls in each feature')
nulls_percentage_after

# %%
raw_data['saledate'] = pd.to_datetime(raw_data['saledate'])
raw_data['saledate'].dtype

# %%
raw_data['YearOfSale'] = raw_data['saledate'].dt.year
raw_data['MonthOfSale'] = raw_data['saledate'].dt.month
raw_data['DayOfSale'] = raw_data['saledate'].dt.day
raw_data

# %%
data_cleaning = make_pipeline(
    DropColumnTransformer(columns=["saledate"]),
    FunctionTransformer(lambda X: X.drop_duplicates(), validate=False),
    CustomOutlierRemover(),
)

df_cleaned = data_cleaning.fit_transform(raw_data)
df_cleaned.head()

# %% [markdown]
# ### Preprocessing

# %%
preprocessing_pipeline = make_pipeline(
   CustomCategoryDivider(column = 'fiProductClassDesc'),
   CustomWhitespaceRemover(columns=['fiSecondaryDesc', 'fiBaseModel', 'fiModelDesc']),
   NaNIndicator(columns=['auctioneerID', 'fiSecondaryDesc', 'Enclosure']),
   CustomLabelEncoder(columns=['fiModelDesc', 'fiBaseModel', 'fiSecondaryDesc', 'state']),
   CustomOneHotEncoder(columns=['ProductGroup', 'ProductGroupDesc', 'Enclosure', 'Hydraulics', 'Wheel Loader', 'Skid Steer Loader',
       'Hydraulic Excavator, Track', 'Backhoe Loader',
       'Track Type Tractor, Dozer', 'Motorgrader']),
   CustomMinMaxScaler(columns=['YearMade']),
   CustomImputer(strategy='most_frequent', columns=['auctioneerID'])
)
df_preprocessed = preprocessing_pipeline.fit_transform(df_cleaned)
df_preprocessed.head()

# %%
unique_column_content_check(df_preprocessed)

# %%
print(*df_preprocessed.columns)

# %% [markdown]
# ___
# # ***EDA*** After Cleaning
# ___

# %% [markdown]
# During cleaning we changed the feature *saledate* into numerical values and put them as three features - 
# *YearOfSale*, *MonthOfSale*, *DayOfSale*.
# 
# If they give some results during the numerical analysis, we will mention them.

# %%
df_cleaned.describe()

# %%
unique_column_content_check(df_cleaned)

# %%
print("\nSummary statistics for categorical variables of cleaned data:")
for column in df_cleaned.select_dtypes(include=['object']).columns:
    series_1 = pd.Series(data=raw_data[column].value_counts().values,
                     index=raw_data[column].value_counts().index,
                     name=f'{column}_raw')

    series_2 = pd.Series(data=df_cleaned[column].value_counts().values,
                        index=df_cleaned[column].value_counts().index,
                        name=f'{column}_cleaned')

    merged_df = pd.DataFrame({
        f'{column}_raw': series_1,
        f'{column}_cleaned': series_2
    }).reset_index(drop=True)
    merged_df.index = series_1.index.union(series_2.index)
    print(merged_df)

# %% [markdown]
# We can clearly see, that cleaned data indeed has less data in each feature and each category.
# 
# We only showed categorical features that were not gotten rid of, so that means that there were outliers and duplicated rows, which have been eliminated during cleaning.

# %%
num_cols_cleaned = df_cleaned.describe().columns

correlation_matrix = df_cleaned[num_cols_cleaned].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, linewidth=.5, fmt='.2f', cmap='crest')
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
corr_matrix_dataframe(df_cleaned[num_cols_cleaned])

# %%
sns.pairplot(df_cleaned[num_cols_cleaned], diag_kind='kde')
plt.show()

# %% [markdown]
# This time the results of correlations haven't changed much, however we can now see some correlation between *YearOfSale* and *YearMade*, other than *datasource* and *SalesID*

# %%
for var in num_cols_cleaned:
    plt.figure(figsize=(8, 6))
    plt.hist(df_cleaned[var], bins=len(df_cleaned[var])//7000, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.show()

# %% [markdown]
# After getting rid of outliers we can see more clearly the distributions of numerical data. Most of them display similarity to the normal distribution.

# %%
for feature in num_cols_cleaned:
    plt.scatter(df_cleaned.index, df_cleaned[feature], c='skyblue', s=3)
    plt.xlabel("index")
    plt.ylabel(feature)
    plt.title(f"{feature} and index")
    plt.show()

# %% [markdown]
# Nothing significant changes when it comes to the relationship between cleaned numerical data and indices.

# %%
perform_statistical_tests(['chi2 independence'], df_cleaned, 'SalePrice')

# %% [markdown]
# When it comes to independency, only SalesID and SalePrice got tested as independent (failed to reject independence).</br></br> Other pairs rejected independence but only a couple of them have Cramer's V association coefficient worth to mention.</br>
# </br>MachineID and SalePrice Cramer's V: 0.937 - Almost perfect
# </br>ProductGroup and SalePrice Cramer's V: 0.333 - Medium
# </br>ProductGroupDesc and SalePrice Cramer's V: 0.333 - Medium
# </br>Enclosure and SalePrice Cramer's V: 0.258 - Almost medium
# </br></br>So these pairs play a big role for our ML Models

# %%
perform_statistical_tests(['ks normality'], df_cleaned, 'SalePrice')

# %% [markdown]
# Every pair above rejected normality, so the distributions are not normally distributed (ideally), nevertheless using our eyes we can clearly see on the graphs in the previous steps, that the distributions resemble left/right skewed normal distributions or variations of them.

# %% [markdown]
# ___
# # ***Regression***
# ___

# %% [markdown]
# ### Splitting train and validation

# %%
df_train = df_preprocessed[df_preprocessed['YearOfSale'] <= 2011]
df_valid = df_preprocessed[df_preprocessed['YearOfSale'] == 2012]

# %%
X_train, y_train = df_train.drop(['SalePrice'], axis=1), df_train['SalePrice']
X_valid, y_valid = df_valid.drop(['SalePrice'], axis=1), df_valid['SalePrice']

# %%
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape

# %% [markdown]
# ### XGBoost

# %%
xgb = XGBRegressor(learning_rate=0.108, max_depth=8, n_estimators=555, subsample=0.8, tree_method='approx')

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_valid)

# %%
metrics_xgb_valid = calculate_metrics(y_valid, y_pred, 'XGBoost')
metrics_xgb_valid

# %% [markdown]
# ### Linear Regression

# %%
scaler_linreg = MinMaxScaler()
X_train_scaled_linreg = scaler_linreg.fit_transform(X_train)
X_valid_scaled_linreg = scaler_linreg.transform(X_valid)


LinReg = LinearRegression(copy_X=True, fit_intercept=False, positive=False)


LinReg.fit(X_train_scaled_linreg, y_train)


y_pred = LinReg.predict(X_valid_scaled_linreg)

# %%
metrics_linreg_valid = calculate_metrics(y_valid, y_pred, 'Linear Regression')
metrics_linreg_valid

# %% [markdown]
# ### Ridge Regression

# %%
scaler_ridge = MinMaxScaler()
X_train_scaled_ridge = scaler_ridge.fit_transform(X_train)
X_valid_scaled_ridge = scaler_ridge.transform(X_valid)


ridge = Ridge(alpha=0.1, fit_intercept=False, solver='auto')

ridge.fit(X_train_scaled_ridge, y_train)

y_pred = ridge.predict(X_valid_scaled_ridge)

# %%
metrics_ridge_valid = calculate_metrics(y_valid, y_pred, 'Ridge')
metrics_ridge_valid

# %% [markdown]
# ### Lasso Regression

# %%
scaler_lasso = MinMaxScaler()
X_train_scaled_lasso = scaler_lasso.fit_transform(X_train)
X_valid_scaled_lasso = scaler_lasso.transform(X_valid)


lasso = Lasso(alpha=1, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.01, warm_start=True, positive=False)

lasso.fit(X_train_scaled_lasso, y_train)

y_pred = lasso.predict(X_valid_scaled_lasso)

# %%
metrics_lasso_valid = calculate_metrics(y_valid, y_pred, 'Lasso')
metrics_lasso_valid

# %% [markdown]
# ### ElasticNet Regression

# %%
scaler_elasticnet = StandardScaler()
X_train_scaled_elasticnet = scaler_elasticnet.fit_transform(X_train)
X_valid_scaled_elasticnet = scaler_elasticnet.transform(X_valid)


elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5, copy_X=True, max_iter=1000, tol=0.01)

elastic_net.fit(X_train_scaled_elasticnet, y_train)

y_pred = elastic_net.predict(X_valid_scaled_elasticnet)

# %%
metrics_elasticnet_valid = calculate_metrics(y_valid, y_pred, 'ElasticNet')
metrics_elasticnet_valid

# %% [markdown]
# ### Linear SVM

# %%
pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(-1, 1))),
    ('nystroem', Nystroem(kernel='rbf', gamma=0.01, n_components=1400)),
    ('svr', LinearSVR(max_iter=10000, C=1000, epsilon=0.01))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_valid)

# %%
metrics_svm_valid = calculate_metrics(y_valid, y_pred, 'Support VM')
metrics_svm_valid

# %% [markdown]
# ### Decision Tree

# %%
dtr = DecisionTreeRegressor(random_state=21, max_depth=None, min_samples_split=2, min_samples_leaf=8)

dtr.fit(X_train, y_train)

y_pred = dtr.predict(X_valid)

# %%
metrics_dtr_valid = calculate_metrics(y_valid, y_pred, 'Decision Tree')
metrics_dtr_valid

# %% [markdown]
# ### AdaBoost

# %%
adaboost = AdaBoostRegressor(n_estimators=300, learning_rate=0.1, loss='linear', random_state=42)

adaboost.fit(X_train, y_train)

y_pred = adaboost.predict(X_valid)

# %%
metrics_adaboost_valid = calculate_metrics(y_valid, y_pred, 'AdaBoost')
metrics_adaboost_valid

# %% [markdown]
# ### Random Forest

# %%
RFR_model = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=14, min_samples_split=2, min_samples_leaf=1, max_features=None, bootstrap=True, oob_score=True, n_jobs=-1)

RFR_model.fit(X_train, y_train)

y_prediction = RFR_model.predict(X_valid)

# %%
metrics_rf_valid = calculate_metrics(y_valid, y_prediction, 'Random Forest')
metrics_rf_valid

# %% [markdown]
# ### Gradient Boosting

# %%
GBR_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=3, random_state=13)

GBR_model.fit(X_train, y_train)

y_prediction = GBR_model.predict(X_valid)

# %%
metrics_gb_valid = calculate_metrics(y_valid, y_prediction, 'Gradient Boosting')
metrics_gb_valid

# %% [markdown]
# ### k-Nearest Neighbours

# %%
scaler_knn = MinMaxScaler()
X_train_scaled_knn = scaler_knn.fit_transform(X_train)
X_valid_scaled_knn = scaler_knn.transform(X_valid)

knn = KNeighborsRegressor(n_neighbors=575, n_jobs=5)

knn.fit(X_train_scaled_knn, y_train)

y_prediction = knn.predict(X_valid_scaled_knn)

# %%
metrics_knn_valid = calculate_metrics(y_valid, y_prediction, 'k-Nearest Neighbors')
metrics_knn_valid

# %% [markdown]
# ___
# # ***Validation Comparison***
# ___

# %%
metrics_list_valid = [metrics_xgb_valid, metrics_linreg_valid, metrics_ridge_valid, metrics_lasso_valid, metrics_elasticnet_valid, metrics_adaboost_valid,
                      metrics_rf_valid, metrics_gb_valid, metrics_knn_valid, metrics_dtr_valid, metrics_svm_valid]

# %%
aggregate_metrics_and_create_heatmaps(metrics_list_valid)

# %% [markdown]
# ___
# # ***Test Data Scores***
# ___

# %%
# test_data = pd.read_csv('attachments_p2t3/Test.csv')
# test_data.head()

# %%
# test_data = test_data.drop(columns=columns_to_terminate, axis=1) # Remove cols with a lot of nulls

# test_data.columns

# %%
# for column in test_data.columns: # Replacing ['None or Unspecified','#NAME?'] to nan
#     test_data[column] = test_data[column].map(lambda x: np.nan if x in ['None or Unspecified','#NAME?'] else x)

# for column in test_data.columns: # Checking if ['None or Unspecified','#NAME?'] were replaced to nan
#     if test_data[column].dtype == 'object':  # Check if the column is categorical
#         unique_values = test_data[column].unique()
#         if 'None or Unspecified' in unique_values or '#NAME?' in unique_values:
#             print('replacing failed')

# %%
# nulls_percentage_after = test_data.isnull().mean() * 100
# print('Percentage of nulls in each feature')
# nulls_percentage_after

# %%
# test_data['saledate'] = pd.to_datetime(test_data['saledate'])
# test_data['saledate'].dtype

# %%
# test_data['YearOfSale'] = test_data['saledate'].dt.year
# test_data['MonthOfSale'] = test_data['saledate'].dt.month
# test_data['DayOfSale'] = test_data['saledate'].dt.day
# test_data

# %%
# df_test_cleaned = data_cleaning.fit_transform(test_data)

# %%
# df_test_preprocessed = preprocessing_pipeline.fit_transform(df_test_cleaned)

# %%
# X_test, y_test = df_test_preprocessed.drop(['SalePrice'], axis=1), df_test_preprocessed['SalePrice']

# %%
# y_pred_test = xgb.predict(X_test)

# calculate_metrics(y_test, y_pred_test)



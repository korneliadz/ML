# Blue Book for Bulldozers - Regression Journey

This project is a part of the kaggle.com competition, which goal was to predict the auction sale price for a piece of heavy equipment to create a "blue book" for bulldozers.
[More information here.](https://www.kaggle.com/c/bluebook-for-bulldozers/data)

![quote](https://i0.wp.com/danslee.co.uk/wp-content/uploads/2016/11/data.jpg?resize=768%2C525&ssl=1)
### Part 1 - EDA before cleaning 
After getting to know our dataset, we did some brief statistical descriptions of numerical data, such as min, max values, histograms, pairplots, correlations matrix.
We concluded that there is no significant correlation between features except for maybe two.
Also at the begging we have already noticed that there is a lot of outliers in YearMade feature, as it suggested that bulldozers were made in a year 1000.

For categorical data we have just checked the number of categories in each feature, and how many data is in each one.

Then it came to seeing how many null values are in our dataset, and unfortunately, it turned out that many features have close to 80-90% of nulls - *we have to get rid of them*.

### Part 2 - cleaning and preprocessing

This is the part were it gets interesting.

***Cleaning***

Cleaning was straightforward - get rid of features with more than 40% of nulls, duplicates and outliers.

***Preprocessing***

This was *actually* the fun part.
We had to create three new preprocessing steps to the pipeline specially for this dataset, but we generalised it for future uses:
- *CustomCategoryDivider* was added in order to create new features based on a given feature
- *CustomWhitespaceRemover* strips whitespaces from string - it was added, since some features had categories like 'H' and 'H ', which should be the same
- *NaNIndicator* identifies missing values in specified columns and creates new binary columns indicating the presence of missing values

These steps were crucial, especially the *CustomCategoryDivider*, as it bumped some linear models from $R^2\approx 0.5$ to $R^2\approx 0.75$.

Additionally, we took categorical feature *SaleDate*, and created three new numerical features - *YearOfSale*, *MonthOfSale*, *DayOfSale*.
### Part 3 - EDA after cleaning
We have decided to do similar things after cleaning the data for comparison, but also create some statistical tests, as now we are left with only significant features.
### Part 4 - Models 
Models used for this project:
- *XGBoost*
- *Linear Regression*
- *Ridge Regression*
- *Lasso Regression*
- *Elastic Net*
- *AdaBoost*
- *Random Forest*
- *Support Vector Machine*
- *k-Nearest Neighbours*
- *Decision Tree*

We measured them using:
- *$R^2$ Score*	
- *Root Mean Squared Error*
- *Mean Absolute Percentage Error* 
- *Symmetric Mean Absolute Percentage Error*
- *Mean Squared Logarithmic Error*
- *Root Mean Squared Logarithmic Error*

In order to get the best parameters we used *GridSearchCV*, however, since it takes so long each time, we took the best hyperparametrs from *model_name.best_params_* and put them in the considered regressor. 

Then, for better code readability we removed the process of searching the hyperparameters as we are only interested in its outcome.

Since the original kaggle competition focuses on the *Root Mean Squared Logarithmic Error*, we have also focused on reducing it.

### Part 5 - Visualisation
We have made a custom function for calculating every score measure mentioned above, for models' scores to be displayed in comparison to each and every regressor, putting the best one in each category on top.

### Part 6 - Conclusion
After conducting experiments with every model we came to the conlusion that the best performing ones in terms of *Root Mean Squared Logarithmic Error (RMSLE)* and *$R^2$ Score* are

1) XGBoost - $RMSLE = 0.214$, $R^2 = 0.893$
2) Random Forest - $RMSLE = 0.217$, $R^2 = 0.882$
3) Gradient Boosting - $RMSLE = 0.283$, $R^2 = 0.842$
# House Price Prediction with Machine Learning

This project aims to create a model that accurately predicts house prices based on various features. It involves data exploration, preprocessing, feature engineering, model training, and evaluation.

![18](https://github.com/tuanng1102/house-price-prediction/assets/147653892/0ff42cf0-6629-47ba-8524-b9501886a666)

# Dataset:

The dataset used is named "Housing.csv". It contains information about houses, including features such as:

```Number of bedrooms```,
```Number of bathrooms```,
```Area```,
```Furnishing status```,
```Air conditioning```,
```Number of stories```,
```Parking availability```,
and more

# Steps Involved:

### 1. Import Libraries:

- ```pandas``` for data manipulation
- ```numpy``` for numerical operations
- ```matplotlib``` and ```seaborn``` for visualizations
- ```sklearn``` for machine learning models and evaluation metrics
- ```catboost```, ```lightgbm```, and ```xgboost``` for additional regression algorithms

``` bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

%matplotlib inline 
import warnings
warnings.filterwarnings("ignore")

```

### 2. Load Dataset:

Read the CSV file into a pandas DataFrame.

``` bash
df = pd.read_csv("Housing.csv")
```

### 3. Exploratory Data Analysis (EDA):

- Create bar plots, box plots, distribution plots, scatter plots, and pair plots to visualize relationships between features and distribution of data.
- Identify and handle outliers in price and area using IQR (Interquartile Range) method.

``` bash
# Bar plot
sns.barplot(x=df['airconditioning'], y=df['bedrooms'], hue=df["furnishingstatus"])
```

![19](https://github.com/tuanng1102/house-price-prediction/assets/147653892/79131992-030c-472c-9494-1c7b923d3790)

``` bash
# Box plot
fig, axs = plt.subplots(2, 3, figsize=(10, 5))
plt1 = sns.boxplot(df['price'], ax=axs[0, 0])
plt2 = sns.boxplot(df['area'], ax=axs[0, 1])
plt3 = sns.boxplot(df['bedrooms'], ax=axs[0, 2])
plt4 = sns.boxplot(df['bathrooms'], ax=axs[1, 0])
plt5 = sns.boxplot(df['stories'], ax=axs[1, 1])
plt6 = sns.boxplot(df['parking'], ax=axs[1, 2])
plt.tight_layout()
```

![20](https://github.com/tuanng1102/house-price-prediction/assets/147653892/0e1daf00-678d-47af-97a5-b6104e5fe7c1)

``` bash
# Dealing with outliers in price
Q1 = df.price.quantile(0.25)
Q3 = df.price.quantile(0.75)
IQR = Q3 - Q1
df = df[(df.price >= Q1 - 1.5*IQR) & (df.price <= Q3 + 1.5*IQR)]
plt.boxplot(df.price)
```

![21](https://github.com/tuanng1102/house-price-prediction/assets/147653892/269618a2-eccd-4960-8170-4720a0011ab9)

``` bash
# Dealing with outliers in area
Q1 = df.area.quantile(0.25)
Q3 = df.area.quantile(0.75)
IQR = Q3 - Q1
df = df[(df.area >= Q1 - 1.5*IQR) & (df.area <= Q3 + 1.5*IQR)]
plt.boxplot(df.area)
```

![22](https://github.com/tuanng1102/house-price-prediction/assets/147653892/71ccda9a-ec5e-48b2-96e9-47d84ff91ad3)

``` bash
# After dealing with outliers
sns.boxplot(x='furnishingstatus', y='price', hue='airconditioning', data=df)
```

![23](https://github.com/tuanng1102/house-price-prediction/assets/147653892/4d204436-4ff7-4a86-ad8f-18dff15d7db2)

``` bash
# Distplot
sns.distplot(df["bathrooms"], hist=False)
sns.distplot(df["bedrooms"], hist=False)
sns.distplot(df["stories"], hist=False)
sns.distplot(df["parking"], hist=False)
```

![24](https://github.com/tuanng1102/house-price-prediction/assets/147653892/36a9f8f6-276e-4dac-93f0-1d5c073c3ac9)

![25](https://github.com/tuanng1102/house-price-prediction/assets/147653892/3cabb20a-02c5-414e-9b06-8cf51eaae46b)

``` bash
# Heatmap
sns.heatmap(df.corr(), cmap='viridis',annot=True)
```

![26](https://github.com/tuanng1102/house-price-prediction/assets/147653892/f7dd3464-44ab-416d-bd1f-9ce887550e1b)

``` bash
# Scatter plot
sns.scatterplot(y=df['price'], x=df['area'], hue=df['furnishingstatus'])
```

![27](https://github.com/tuanng1102/house-price-prediction/assets/147653892/493de333-7448-4d04-8a7a-bd1fe06e0922)

``` bash
# Pair plot
sns.pairplot(df, hue="furnishingstatus")
```

![28](https://github.com/tuanng1102/house-price-prediction/assets/147653892/62e0e1f7-3033-4189-bf66-50d7631cacd4)

### 4. Feature Engineering:

Create dummy variables for categorical features to make them suitable for numerical modeling.

``` bash
# Feature engineering
status = pd.get_dummies(data[['furnishingstatus', 'mainroad', 'guestroom', 'basement',
                              'hotwaterheating', 'airconditioning', 'prefarea']], drop_first=True)
data = pd.concat([data, status], axis=1)
data.drop(['furnishingstatus', 'mainroad', 'guestroom', 'basement',
           'hotwaterheating', 'airconditioning', 'prefarea'], axis=1, inplace=True)
```

### 5. Data Splitting:

- Split the data into features ```X``` and target variable ```y```.
- Further split ```X``` and ```y``` into training and testing sets (70% for training, 30% for testing).

``` bash
X = data.drop(['price'], axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### 6. Feature Scaling:

Scale the features using ```MinMaxScaler``` to normalize their values for better model performance.

``` bash
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 7. Model Training:

Train various regression models, including:
- ```Random Forest Regressor```
- ```Gradient Boost Regressor```
- ```XGBoost Regressor```
- ```Support Vector Regressor```
- ```Lasso Regression```
- ```Ridge Regression```
- ```LGBM Regressor```
- ```CatBoost Regressor```

``` bash
model = {
    'Random Forest Regressor': RandomForestRegressor(),
    'Gradient Boost Regressor': GradientBoostingRegressor(),
    'XGBoost': xgb.XGBRegressor(),
    'XGRF Regressor': xgb.XGBRFRegressor(),
    'Support Vector regressor': SVR(),
    'Lasso Reg': Lasso(),
    'Ridge Reg': Ridge(),
    'LGBM Reg': LGBMRegressor(),
    'Cat Boost': CatBoostRegressor()
}

pred = {}
for name, model in model.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    pred[name] = y_pred
```

### 8. Model Evaluation:

- Evaluate each model's performance using:
  - Mean Squared Error (MSE)
  - R-squared (R2) score
  - Visualize the Actual vs. Predicted values and residuals for each model to assess their fit.

``` bash
acc = {}
for name, y_pred in pred.items():
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    acc[name] = r2
    print(f"Results for {name} : ")
    print(f"Mean Square Error : {mse}")
    print(f"R2 Score : {r2}")
    plt.figure(figsize=(15, 6))

    # Plot Actual vs. Predicted values
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(y_test)), y_test, label='Actual Trend')
    plt.plot(np.arange(len(y_test)), y_pred, label='Predicted Trend')
    plt.xlabel('Data')
    plt.ylabel('Trend')
    plt.legend()
    plt.title('Actual vs. Predicted')
    # Plot Residuals
    residuals = y_test - y_pred

    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    plt.tight_layout()
    plt.show()
```
![91](https://github.com/tuanng1102/house-price-prediction/assets/147653892/3e2d1529-7004-4a58-9092-bf7f8d319c2a)

![92](https://github.com/tuanng1102/house-price-prediction/assets/147653892/0a07df61-1ba9-4246-b71a-2b47d983ace1)

![93](https://github.com/tuanng1102/house-price-prediction/assets/147653892/4bd29039-d082-4aec-879e-aaddf966ca5e)

![94](https://github.com/tuanng1102/house-price-prediction/assets/147653892/1b102873-c755-47c8-8f83-76971deb4550)

![95](https://github.com/tuanng1102/house-price-prediction/assets/147653892/8bc8f352-f276-46e4-8090-ab9c320df9bf)

![96](https://github.com/tuanng1102/house-price-prediction/assets/147653892/bedab66d-9703-4450-8480-c600f4cd0151)

![97](https://github.com/tuanng1102/house-price-prediction/assets/147653892/0c956aed-7553-4241-998d-cd585ba7eddb)

![98](https://github.com/tuanng1102/house-price-prediction/assets/147653892/d6896430-45db-4e89-b3f9-2ea192397a19)

![99](https://github.com/tuanng1102/house-price-prediction/assets/147653892/5c9ad604-c82b-479a-a6f7-f5c61269181e)

### 9. Results:

Display a DataFrame with the R2 scores (accuracy) for all trained models, allowing for comparison and selection of the best-performing model.

``` bash
data = pd.DataFrame.from_dict(acc, orient='index', columns=['Accuracy'])
print(data)
```

![100](https://github.com/tuanng1102/house-price-prediction/assets/147653892/1c052db9-e6b5-42e5-9fd1-f6f10e9eef2c)

![cc](https://github.com/tuanng1102/house-price-prediction/assets/147653892/2ba7f48a-47ae-4a54-bb2f-1a5840256fd1)

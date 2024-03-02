import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Import dataset
df = pd.read_csv("Housing.csv")

data = df.copy()

# Bar plot
sns.barplot(x=df['airconditioning'], y=df['bedrooms'], hue=df["furnishingstatus"])

# Box plot
fig, axs = plt.subplots(2, 3, figsize=(10, 5))
plt1 = sns.boxplot(df['price'], ax=axs[0, 0])
plt2 = sns.boxplot(df['area'], ax=axs[0, 1])
plt3 = sns.boxplot(df['bedrooms'], ax=axs[0, 2])
plt4 = sns.boxplot(df['bathrooms'], ax=axs[1, 0])
plt5 = sns.boxplot(df['stories'], ax=axs[1, 1])
plt6 = sns.boxplot(df['parking'], ax=axs[1, 2])
plt.tight_layout()

# Dealing with outliers in price
Q1 = df.price.quantile(0.25)
Q3 = df.price.quantile(0.75)
IQR = Q3 - Q1
df = df[(df.price >= Q1 - 1.5*IQR) & (df.price <= Q3 + 1.5*IQR)]
plt.boxplot(df.price)

# Dealing with outliers in area
Q1 = df.area.quantile(0.25)
Q3 = df.area.quantile(0.75)
IQR = Q3 - Q1
df = df[(df.area >= Q1 - 1.5*IQR) & (df.area <= Q3 + 1.5*IQR)]
plt.boxplot(df.area)

# After dealing with outliers
sns.boxplot(x='furnishingstatus', y='price', hue='airconditioning', data=df)

# Distplot
sns.distplot(df["bathrooms"], hist=False)
sns.distplot(df["bedrooms"], hist=False)
sns.distplot(df["stories"], hist=False)
sns.distplot(df["parking"], hist=False)

# Scatter plot
sns.scatterplot(y=df['price'], x=df['area'], hue=df['furnishingstatus'])

# Pair plot
sns.pairplot(df, hue="furnishingstatus")

# Feature engineering
status = pd.get_dummies(data[['furnishingstatus', 'mainroad', 'guestroom', 'basement',
                              'hotwaterheating', 'airconditioning', 'prefarea']], drop_first=True)
data = pd.concat([data, status], axis=1)
data.drop(['furnishingstatus', 'mainroad', 'guestroom', 'basement',
           'hotwaterheating', 'airconditioning', 'prefarea'], axis=1, inplace=True)

X = data.drop(['price'], axis=1)
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
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

# Model evaluation
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

# Accuracy
data = pd.DataFrame.from_dict(acc, orient='index', columns=['Accuracy'])
print(data)
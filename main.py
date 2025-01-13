import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


# tutorial https://www.geeksforgeeks.org/stock-price-prediction-using-machine-learning-in-python/ 

df = pd.read_csv('Tesla.csv')
print(df.head())
#print(df.describe())
#df.info()
#df.shape()

# plot the close price ---  
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

# drop redundant column 
df = df.drop(['Adj Close'], axis=1)
# check for null values - none
df.isnull().sum()


# distribution plot for the continuous features 
features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(df[col])
plt.show()
# shows two peaks in open, high, low and close, volume is left skewed


# feature engineering - split the date into separate columns 
splitted = df['Date'].str.split('/', expand=True)

df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')

df.head()

# add is quarter end column
df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head()

# drop date
df.drop('Date', axis=1).groupby('is_quarter_end').mean()

# add more columns
df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']

#  target feature which is a signal whether to buy or not 
# 1 - Indicates that the price increased the next day (a potential "buy" signal).
# 0 - Indicates that the price did not increase (a potential "hold" or "not buy" signal).
# binary classification label 
# train the model to predict this value
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


# heat map -----
plt.figure(figsize=(10, 10)) 

# As our concern is with the highly 
# correlated features only so, we will visualize 
# our heatmap as per that criteria only. 
sb.heatmap(df.drop('Date', axis=1).corr() > 0.9, annot=True, cbar=False)
plt.show()


# data splitting and visualization --- 
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)


# models 
models = [LogisticRegression(), SVC(
  kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(
    Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(
    Y_valid, models[i].predict_proba(X_valid)[:,1]))
  print()


# plot results --- 
predictions = [model.predict_proba(X_valid)[:, 1] for model in models]

plt.figure(figsize=(15, 5))

# Plot for each model
for i, model in enumerate(models):
    plt.subplot(1, 3, i + 1)  # Create subplots for each model
    plt.scatter(range(len(Y_valid)), Y_valid, label='Actual', alpha=0.6, color='blue')
    plt.scatter(range(len(Y_valid)), predictions[i], label='Predicted', alpha=0.6, color='red')
    plt.title(f'Model: {type(model).__name__}')
    plt.xlabel('Index')
    plt.ylabel('Target')
    plt.legend()

plt.tight_layout()
plt.show()

# Plot  -------------

plt.figure(figsize=(15, 7))

# Create a time-based index for x-axis (sample numbers)
sample_numbers = range(len(Y_valid))

# Get predictions
predictions = models[1].predict(X_valid)

# Plot actual target values
plt.scatter(sample_numbers, Y_valid, 
           label='Actual Target', 
           color='blue', 
           alpha=0.6)

# Plot predicted target values
plt.scatter(sample_numbers, predictions, 
           label='Predicted Target', 
           color='red', 
           alpha=0.6)

# Find where predictions match actuals
matching_indices = Y_valid == predictions
plt.scatter(np.array(sample_numbers)[matching_indices], 
           Y_valid[matching_indices],
           label='Matching Predictions',
           color='green',
           s=100,  # Larger size for emphasis
           facecolors='none',  # Make circle hollow
           linewidth=2)

plt.title('Actual vs Predicted Targets Over Time', fontsize=15)
plt.xlabel('Sample Number', fontsize=12)
plt.ylabel('Target Value', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
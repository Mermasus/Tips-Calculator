import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('tips.csv')
df.head()
df.shape
df.info()
df.describe().T
df.isnull().sum()

plt.subplots(figsize=(15,8))
for i, col in enumerate(['total_bill', 'tip']):
	plt.subplot(2,3, i + 1)
	sb.distplot(df[col])
plt.tight_layout()
plt.show()

plt.subplots(figsize=(15,8))
for i, col in enumerate(['total_bill', 'tip']):
	plt.subplot(2,3, i + 1)
	sb.distplot(df[col])
plt.tight_layout()
plt.show()

df.shape, df[(df['total_bill']<45) & (df['tip']<7)].shape
df = df[(df['total_bill']<45) & (df['tip']<7)]

feat = df.loc[:,'sex':'size'].columns

plt.subplots(figsize=(15,8))
for i, col in enumerate(['total_bill', 'tip']):
	plt.subplot(2,3, i + 1)
	sb.distplot(df[col])
plt.tight_layout()
plt.show()

plt.scatter(df['total_bill'], df['tip'])
plt.title('Total Bill VS Total Tip')
plt.xlabel('Total Bill')
plt.ylabel('Total Tip')
plt.show()

df.groupby(['size']).mean(numeric_only=True)
df.groupby(['time']).mean(numeric_only=True)
df.groupby(['day']).mean(numeric_only=True)

le = LabelEncoder()

for col in df.columns:
	if df[col].dtype == object:
		df[col] = le.fit_transform(df[col])

df.head()

plt.figure(figsize=(7,7))
sb.heatmap(df.corr() > 0.7, annot = True, cbar = False)
plt.show()

features = df.drop('tip', axis=1)
target = df['tip']

x_train, x_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=22)
x_train.shape, x_val.shape

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

model = [LinearRegression(), XGBRegressor(), RandomForestRegressor(), AdaBoostRegressor()]
for i in range(4):
	models[i].fit(x_train, y_train)

	print(f'{models[i]} : ')
	pred_train = models[i].predict(x_train)
	print('Training Accuracy : ', mae(y_train, pred_train))

	pred_val = models[i].predict(x_val)
	print('Validation Accuracy : ', mae(y_val, pred_val))
	print()

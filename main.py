# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

df = pd.read_csv('C:/Users/Monster/Desktop/winered.csv')


# See the number of rows and columns
print("Rows, columns: " + str(df.shape))
# See the first five rows of the dataset
df.head()

# Missing Values
print(df.isna().sum())

fig = px.histogram(df, x='quality')
fig.show()

corr = df.corr()
plt.pyplot.subplots(figsize=(15, 10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True,
            cmap=sns.diverging_palette(220, 20, as_cmap=True))

# Create Classification version of target variable
df['goodquality'] = [1 if x >= 7.5 else 0 for x in df['quality']]
# Separate feature variables and target variable
X = df.drop(['quality', 'goodquality'], axis=1)
y = df['goodquality']

# See proportion of good vs bad wines
df['goodquality'].value_counts()

for a in range(len(df.corr().columns)):
    for b in range(a):
        if abs(df.corr().iloc[a, b]) > 0.7:
            name = df.corr().columns[a]
            print(name)

new_df = df.drop('total sulfur dioxide', axis=1)
new_df.isnull().sum()

new_df.update(new_df.fillna(new_df.mean()))

# catogerical vars
next_df = pd.get_dummies(new_df, drop_first=True)
# display new dataframe

next_df["best quality"] = [1 if x >= 7 else 0 for x in df.quality]
print(next_df)

# See proportion of good vs bad wines
df['goodquality'].value_counts()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# importing module
from sklearn.preprocessing import MinMaxScaler

# creating normalization object
norm = MinMaxScaler()
# fit data
norm_fit = norm.fit(X_train)
new_xtrain = norm_fit.transform(X_train)
new_xtest = norm_fit.transform(X_test)
# display values
print(new_xtrain)

# importing modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_squared_error


# creating RandomForestClassifier constructor
rnd = RandomForestClassifier()
# fit data
fit_rnd = rnd.fit(new_xtrain, y_train)
# predicting score
rnd_score = rnd.score(new_xtest, y_test)

print('score of model is : ', rnd_score)
# display error rate
print('Calculating the error...')
# calculating mean squared error
#bu alttaki satırı buraya taşımamız gerekiyormuş lan bi yıl uğraştım
x_predict = list(rnd.predict(X_test))
rnd_MSE = mean_squared_error(y_test,x_predict)
# calculating root mean squared error
rnd_RMSE = np.sqrt(rnd_MSE)
# display MSE
print('mean squared error is : ', rnd_MSE)
# display RMSE
print('root mean squared error is : ', rnd_RMSE)

print(classification_report(y_test, x_predict))


predicted_df = {'predicted_values': x_predict, 'original_values': y_test}
# creating new dataframe
pd.DataFrame(predicted_df).head(20)

import pickle
file = 'sonundabitti'
#save file
save = pickle.dump(rnd,open(file,'wb'))
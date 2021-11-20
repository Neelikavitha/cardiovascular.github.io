 
import numpy as np
import pandas as pd
 
data=pd.read_csv(r'C:\Users\KAVITHA\OneDrive\Desktop\miniproject\Heart disease.csv')
data.head()
data.describe()
data.info()
print (data.TenYearCHD.value_counts()[0])
print (data.TenYearCHD.value_counts()[1])
data.drop('BPMeds', axis='columns', inplace=True)
data.drop('prevalentStroke', axis='columns', inplace=True)
data.drop('diabetes', axis='columns', inplace=True)
data.info()
data.isna().sum()
X = data.drop(['TenYearCHD'], axis=1, inplace=False)
print('X Data is \n' , X.head())
print('X shape is ' , X.shape)
y = data['TenYearCHD']
print('y Data is \n' , y.head())
print('y shape is ' , y.shape)
X = X.apply(lambda x: x.fillna(x.mean()),axis=0)
X.isnull().sum(axis = 0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=44, shuffle =True)
#Splitted Data
print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# # Applying LogisticRegression Model
from sklearn.linear_model import LogisticRegression
LogisticRegressionModel = LogisticRegression(penalty='l2',solver='sag',C=1.0,random_state=33)
LogisticRegressionModel.fit(X_train, y_train)
y_pred = LogisticRegressionModel.predict(X_test)

data.head(20)



# # Calculating train and test scores

print('LogisticRegressionModel Train Score is : ' , LogisticRegressionModel.score(X_train, y_train))
print('LogisticRegressionModel Test Score is : ' , LogisticRegressionModel.score(X_test, y_test))


# # calculating classification report

from sklearn.metrics import classification_report


ClassificationReport = classification_report(y_test,y_pred)
print('Classification Report is : ', ClassificationReport )


# # Calculating Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

CM = confusion_matrix(y_test, y_pred)
print('Confusion Matrix is : \n', CM)

sns.heatmap(CM, center = True)
plt.show()


import pickle
pickle.dump(LogisticRegressionModel,open('model.pkl','wb'))





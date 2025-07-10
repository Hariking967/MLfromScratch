# #TODO:testing Linear Regression:
#
# from sklearn.datasets import make_regression
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from LinearRegression import LinearRegression
#
# x, y = make_regression(n_samples=500, n_features=1, noise=10)
# x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.3)
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
# plt.scatter(x_train, y_train, c='b')
# model = LinearRegression()
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# plt.scatter(x_test, y_pred, c='g')
# plt.show()
#
# #TODO:testing Logistic Regression:
# import numpy as np
# from sklearn. model_selection import train_test_split
# from sklearn import datasets
# from LogisticRegression import LogisticRegression
# bc = datasets. load_breast_cancer ( )
# X, y = bc.data, bc.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# y_train = y_train.reshape(-1,1)
# y_test = y_test.reshape(-1,1)
# clf = LogisticRegression()
# clf. fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# def accuracy(y_pred, y_test) :
#     return np.sum(y_pred == y_test)/len(y_test)
# acc = accuracy(y_pred, y_test)
# print(acc)

# #TODO:testing Descicion Trees:
# import pandas as pd
# from DecisionTrees import DecisionTrees
# from sklearn.model_selection import train_test_split
# df = pd.read_csv('realistic_binary_debt_data.csv')
# cols = df.columns
# y_df = df['debt']
# x_df = df.drop(columns='debt')
# x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(x_df, y_df, random_state=42, test_size=0.2)
# model = DecisionTrees()
# model.fit(x_train_df, y_train_df)
# predictions = model.predict(x_test_df)
# score = 0
# for i in range(len(predictions)):
#     if y_test_df.iloc[i] == predictions[i]:
#         score += 1
# print("accuracy: ", score/len(predictions)*100, '%')

#TODO: testing KNN:
import pandas as pd
from sklearn.model_selection import train_test_split
from KNN import KNN
df = pd.read_csv('titanic_knn_clean.csv')
y_df = df['survived']
x_df = df.drop(columns='survived')
x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(x_df, y_df, random_state=42, test_size=0.2)
model = KNN(k=3)
model.fit(x_train_df, y_train_df)
predictions = model.predict(x_test_df)
score = 0
for i in range(len(predictions)):
    if y_test_df.iloc[i] == predictions[i]:
        score += 1
print("accuracy: ", score/len(predictions)*100, '%')
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

# #TODO: testing KNN:
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from KNN import KNN
# df = pd.read_csv('titanic_knn_clean.csv')
# y_df = df['survived']
# x_df = df.drop(columns='survived')
# x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(x_df, y_df, random_state=42, test_size=0.2)
# model = KNN(k=3)
# model.fit(x_train_df, y_train_df)
# predictions = model.predict(x_test_df)
# score = 0
# for i in range(len(predictions)):
#     if y_test_df.iloc[i] == predictions[i]:
#         score += 1
# print("accuracy: ", score/len(predictions)*100, '%')

# #TODO: testing NaiveBayes:
# from sklearn.model_selection import train_test_split
# from NaiveBayes import NaiveBayes
# import pandas as pd
# emails = [
#     "Win a free iPhone now",
#     "Your invoice is ready",
#     "Cheap medicines available",
#     "Meeting scheduled for tomorrow",
#     "Limited time offer on shoes",
#     "Let’s catch up this weekend",
#     "Congratulations you have won",
#     "Your Amazon order has shipped",
#     "Click here for free money",
#     "Lunch at 1 PM?",
#     "Earn money working from home",
#     "Can we reschedule our call?",
#     "You are selected for a prize",
#     "Final reminder: payment due",
#     "Get rich quick scheme inside",
#     "Important update from HR",
#     "Lowest prices guaranteed",
#     "Team meeting at 10 AM",
#     "Unlock your free gift now",
#     "Budget report attached",
#     "Exclusive deal just for you",
#     "Are we still on for coffee?",
#     "You are a lucky winner",
#     "Here’s the project file",
#     "Double your income today",
#     "Notes from today's class",
#     "Amazing offer, don’t miss it",
#     "Tomorrow’s agenda updated",
#     "100% free access now",
#     "Reminder: dentist appointment",
#     "Free vacation awaits you",
#     "Update your password",
#     "Claim your lottery prize",
#     "Report submission deadline",
#     "Win big now, click here",
#     "Join us for dinner tonight",
#     "Cheap loans available",
#     "Here’s the latest report",
#     "Act fast, limited offer",
#     "Check out this investment",
#     "Can we move our meeting?",
#     "Free gift card for you",
#     "Presentation slides attached",
#     "You have been selected",
#     "Looking forward to our trip",
#     "This is not a scam",
#     "Quarterly performance update",
#     "Download your free file",
#     "Status update on ticket",
#     "Final notice for your account"
# ]
# spam = [
#     1, 0, 1, 0, 1,
#     0, 1, 0, 1, 0,
#     1, 0, 1, 0, 1,
#     0, 1, 0, 1, 0,
#     1, 0, 1, 0, 1,
#     0, 1, 0, 1, 0,
#     1, 0, 1, 0, 1,
#     0, 1, 0, 1, 0,
#     1, 0, 1, 0, 1,
#     0, 1, 0, 1, 0
# ]
# df = pd.DataFrame({
#     "email": emails,
#     "spam": spam
# })
# print(df.head())
# y_df = df['spam']
# x_df = df['email']
# x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(x_df, y_df, random_state=42, test_size=0.2)
# model = NaiveBayes(alpha=2)
# model.fit(x_train_df, y_train_df)
# predictions = model.predict(x_test_df)
# score = 0
# for i in range(len(predictions)):
#     if y_test_df.iloc[i] == predictions[i]:
#         score += 1
# print("accuracy: ", score/len(predictions)*100, '%')

#TODO: testing Random Forest
import pandas as pd
from RandomForest import RandomForest
from sklearn.model_selection import train_test_split
df = pd.read_csv('realistic_binary_debt_data.csv')
cols = df.columns
y_df = df['debt']
x_df = df.drop(columns='debt')
x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(x_df, y_df, random_state=42, test_size=0.2)
model = RandomForest(n_trees=5)
model.fit(x_train_df, y_train_df)
predictions = model.predict(x_test_df)
score = 0
for i in range(len(predictions)):
    if y_test_df.iloc[i] == predictions[i]:
        score += 1
print("accuracy: ", score/len(predictions)*100, '%')
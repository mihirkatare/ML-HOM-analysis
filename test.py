import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

dsfolder = r"C:\Users\HP\Desktop\stockdata\NYSE"
df_fund = pd.read_csv(dsfolder+r"\prices-split-adjusted.csv")
df_AAPL = df_fund[df_fund["symbol"]=="AAPL"]
AAPL_c = df_AAPL["close"]
AAPL_c = AAPL_c.reset_index(drop=True)

print(AAPL_c)
# x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, test_size=0.2)
#
# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# model = Sequential()
# model.add(Dense(200, input_dim=40, activation='relu'))
# # model.add(Dense(12, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'] )
#
# model.fit(x_train,y_train, batch_size=25, epochs=200)
#
# loss_and_metrics = model.evaluate(x_test, y_test)

# clf = DecisionTreeClassifier(criterion="gini", ccp_alpha = 0.02)
# sc = cross_validate(clf, x, y, cv=10)
# clf = clf.fit(x,y)
# print(sc)
# print(max(count1,count0)/samples)


# tree.export_graphviz(clf,out_file="tree.dot",filled = True, feature_names = obj.names)

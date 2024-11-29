import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random

df = pd.read_csv("dataset.csv")
df.dropna()
dummy = pd.get_dummies(df, columns=['Profession'], dtype="int64") # classification based on profession
data = dummy.drop(columns=['id'])

sns.scatterplot(data)
plt.show()

x = data[['Academic Pressure', 'Work Pressure', 'CGPA']]
y = data["Depression"]

random.seed(1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
regression = LinearRegression()
regression.fit(xtrain, ytrain)
print(f"regression score (1) : {regression.score(xtest, ytest)}")


x = data[["CGPA"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
regression = LinearRegression()
regression.fit(xtrain, ytrain)
print(f"regression score (2) : {regression.score(xtest, ytest)}")

x = data[["CGPA", "Study Satisfaction"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
regression = LinearRegression()
regression.fit(xtrain, ytrain)
print(f"regression score (3) : {regression.score(xtest, ytest)}")


x = data[["Work/Study Hours"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
regression = LinearRegression()
regression.fit(xtrain, ytrain)
print(f"regression score (4) : {regression.score(xtest, ytest)}")


x = data[["Academic Pressure"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
regression = LinearRegression()
regression.fit(xtrain, ytrain)
print(f"regression score (5) : {regression.score(xtest, ytest)}")



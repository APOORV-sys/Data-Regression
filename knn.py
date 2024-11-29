import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import random

df = pd.read_csv("dataset.csv")
df.dropna()
dummy = pd.get_dummies(df, columns=['Profession'], dtype="int64") # classification based on profession
data = dummy.drop(columns=['id'])

sns.scatterplot(data)
plt.show()

x = data[['Academic Pressure', 'Work Pressure', 'CGPA']]
y = data["Depression"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(xtrain, ytrain)
print(f"knn regression accuracy (1) : {knn.score(xtest, ytest)} or {knn.score(xtest, ytest)*100:.3f}%")

x = data[["CGPA"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(xtrain, ytrain)
print(f"knn regression accuracy (2) : {knn.score(xtest, ytest)} or {knn.score(xtest, ytest)*100:.3f}%")

x = data[["CGPA", "Study Satisfaction"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(xtrain, ytrain)
print(f"knn regression accuracy (3) : {knn.score(xtest, ytest)} or {knn.score(xtest, ytest)*100:.3f}%")

x = data[["Work/Study Hours"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(xtrain, ytrain)
print(f"knn regression accuracy (4) : {knn.score(xtest, ytest)} or {knn.score(xtest, ytest)*100:.3f}%")

x = data[["Academic Pressure"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(xtrain, ytrain)
print(f"knn regression accuracy (5) : {knn.score(xtest, ytest)} or {knn.score(xtest, ytest)*100:.3f}%")




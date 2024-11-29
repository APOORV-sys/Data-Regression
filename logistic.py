import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import random

df = pd.read_csv("dataset.csv")
df.dropna()
dummy = pd.get_dummies(df, columns=['Profession'], dtype="int64") # classification based on profession
data = dummy.drop(columns=['id'])

sns.scatterplot(data)
plt.show()

#based on academi pressure,work pressure,cgpa
x = data[['Academic Pressure', 'Work Pressure', 'CGPA']]
y = data["Depression"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y)

model = LogisticRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print(f"Logistic Regression score (1) : {model.score(xtrain, ytrain)}")
print(f"accuracy (1) : {accuracy} or {accuracy*100:.3f}%")


#based on cgpa
x = data[["CGPA"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print(f"Logistic Regression score (2) : {model.score(xtrain, ytrain)}")
print(f"accuracy (2) : {accuracy} or {accuracy*100:.3f}%")


#based on cgpa,stdy satisfaction
x = data[["CGPA", "Study Satisfaction"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print(f"Logistic Regression score (3) : {model.score(xtrain, ytrain)}")
print(f"accuracy (3) : {accuracy} or {accuracy*100:.3f}%")

# based on workstudy hours
x = data[["Work/Study Hours"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print(f"Logistic Regression score (3) : {model.score(xtrain, ytrain)}")
print(f"accuracy (4) : {accuracy} or {accuracy*100:.3f}%")

#based on academic pressure
x = data[["Academic Pressure"]]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print(f"Logistic Regression score (3) : {model.score(xtrain, ytrain)}")
print(f"accuracy (5) : {accuracy} or {accuracy*100:.3f}%")

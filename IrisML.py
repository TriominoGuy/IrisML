#importing all the necessary modules
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import pandas as pd
import scipy
#import matplotlib.pyplot as plt

#imports the csv file and converts it to a pandas dataframe, and changes the column names to the one in the array for ease of use later
column_names=["sepal-length", "sepal-width", "petal-length", "petal-width", "Species"]
data = pd.read_csv("/Users/administrator/Downloads/Iris.csv", names=column_names)
print(data.head)
print(data.describe())
print(list(data))

#data.plot(kind="box", subplots=True, layout=(2,2), sharex=False, sharey=False)
#data.hist()
#plt.show()

#splits the data into training data and testing data
train, test = train_test_split(data, test_size = 0.9)
print(train.shape)
print(test.shape)
print(test)

namesarray = ["sepal-length", "sepal-width", "petal-length", "petal-width"]
trainx = train[namesarray]
testx = test[namesarray]

#fitting each model with the training data and then making predictions on the test data 
LRmodel = LogisticRegression(solver = "liblinear", multi_class = "auto")
LRmodel.fit(trainx, train.Species)
LRprediction = LRmodel.predict(testx)

KNCmodel = KNeighborsClassifier()
KNCmodel.fit(trainx, train.Species)
KNCprediction = KNCmodel.predict(testx)

DTCmodel = DecisionTreeClassifier()
DTCmodel.fit(trainx, train.Species)
DTCprediction = DTCmodel.predict(testx)

#prints the result of each species prediction for each model and compares it to the actual species
print("Logistic Regression: ")
for i in range(test.shape[0]):
    print("Prediction for test data", i, "is", LRprediction[i],", actual species is", list(test["Species"])[i])

print("K Neighbours Classifier: ")
for j in range(test.shape[0]):
    print("Prediction for test data", j, "is", KNCprediction[j],", actual species is", list(test["Species"])[j])
    
print("Decision Tree Classifier: ")
for k in range(test.shape[0]):
    print("Prediction for test data", k, "is", DTCprediction[k],", actual species is", list(test["Species"])[k]) 

#prints the accuracy of each model for comparison
print("Accuracy for Logistic Regression is", metrics.accuracy_score(LRprediction, test.Species))
print("Accuracy for K Neighbours Classifier is", metrics.accuracy_score(KNCprediction, test.Species))
print("Accuracy for Decision Tree Classifier is", metrics.accuracy_score(DTCprediction, test.Species))


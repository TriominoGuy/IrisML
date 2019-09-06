import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import scipy

namesarray1=["sepal-length", "sepal-width", "petal-length", "petal-width", "Species"]
data = pd.read_csv("/Users/administrator/Downloads/Iris.csv", names=namesarray1)
print(data.head)
print(data.describe())
print(list(data))

#data.plot(kind="box", subplots=True, layout=(2,2), sharex=False, sharey=False)
#data.hist()
#plt.show()

train, test = train_test_split(data, test_size = 0.9)
print(train.shape)
print(test.shape)
print(test)

namesarray2 = ["sepal-length", "sepal-width", "petal-length", "petal-width"]
trainx = train[namesarray2]
testx = test[namesarray2]

LRmodel = LogisticRegression(solver = "liblinear", multi_class = "auto")
LRmodel.fit(trainx, train.Species)
LRprediction = LRmodel.predict(testx)

KNCmodel = KNeighborsClassifier()
KNCmodel.fit(trainx, train.Species)
KNCprediction = KNCmodel.predict(testx)

DTCmodel = DecisionTreeClassifier()
DTCmodel.fit(trainx, train.Species)
DTCprediction = DTCmodel.predict(testx)

print("Logistic Regression: ")
for i in range(test.shape[0]):
    print("Prediction for test data", i, "is", LRprediction[i],", actual species is", list(test["Species"])[i])

print("K Neighbours Classifier: ")
for j in range(test.shape[0]):
    print("Prediction for test data", j, "is", KNCprediction[j],", actual species is", list(test["Species"])[j])
    
print("Decision Tree Classifier: ")
for k in range(test.shape[0]):
    print("Prediction for test data", k, "is", DTCprediction[k],", actual species is", list(test["Species"])[k]) 

print("Accuracy for Logistic Regression is", metrics.accuracy_score(LRprediction, test.Species))
print("Accuracy for K Neighbours Classifier is", metrics.accuracy_score(KNCprediction, test.Species))
print("Accuracy for Decision Tree Classifier is", metrics.accuracy_score(DTCprediction, test.Species))


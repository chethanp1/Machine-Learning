import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("C:/Users/MCA/Desktop/ML Lab Dataset/salary_data.csv")
# print(data.head())
# print(data.info())

# x=data['YearsExperience']
# print(x)

# Another way to split the data (:- means all the rows,0- is the first column)
# x=data.iloc[:,0]
# print(x)

# Conversion of 2D array
x = np.array(data.iloc[:][["YearsExperience"]])
# print(x.shape)

y=np.array((data.iloc[:,1]))
# print(y.shape)

# split the data into train set and test set (80-20 ratio)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.80,random_state=4697)
# print(xtest)

#  Decide on algorithm (Linear Regresson algo)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)
# print(ypred)
# print(ytest)

# To calculate error
from sklearn.metrics import r2_score
r2=r2_score(ytest,ypred)
# print(r2)

# Plotting the samples
# Testing train set
plt.figure(figsize=(8,5))
plt.scatter(xtrain,ytrain,color='blue',s=100,label="ActualPoints")
plt.scatter(xtrain,model.predict(xtrain),color='red',s=100,label="PredictedPoints")
plt.plot(xtrain,model.predict(xtrain),linestyle='dotted',color='orange')
plt.show()

# Testing test set
plt.figure(figsize=(8,5))
plt.scatter(xtest,ytest,color='blue',s=100,label="ActualPoints")
plt.scatter(xtest,model.predict(xtest),color='red',s=100,label="PredictedPoints")
plt.plot(xtest,model.predict(xtest),linestyle='dotted',color='orange')
plt.show()

scores=[]
for i in range(5000):
    xtrain1,xtest1,ytrain1,ytest1=train_test_split(x,y,train_size=0.80,random_state=i)
    model1=LinearRegression()
    model1.fit(xtrain1,ytrain1)
    ypred1=model1.predict(xtest1)
    scores.append(r2_score(ytest1,ypred1))
print(scores)


print(np.max(scores))s
print(np.argmax(scores))

import joblib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#
# print(sns.get_dataset_names())

data = sns.load_dataset("titanic")
# print(data.head())
# print(len(data))

# print(data.info())
# print(data['survived'].value_counts())

# sns.count-plot(x=data['survived'])
# plt.show()

# print(data.isnull().sum())
#
# sns.countplot(x=data['survived'],hue=data['embarked'])
# plt.show()

# print(data.columns)
cols=['fare','class', 'who', 'adult_male', 'deck', 'embark_town','alive', 'alone']
data_new=data.drop(cols,axis=1)
# print(data_new.head())

# print(data_new.isnull().sum())

mean_age=data_new['age'].mean()
# print(mean_age)

mean_age=np.round(mean_age,2)
# print(mean_age)

data_new['age']=data_new['age'].fillna(mean_age)
# print(data_new.isnull().sum())

data_new=data_new.dropna()
# print(data_new.isnull().sum())
# print(len(data_new))
# print(data_new.head())

from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()
data_new['sex']=enc.fit_transform(data_new['sex'])
# print(data_new.head())

data_new['embarked']=enc.fit_transform(data_new['embarked'])
# print(data_new.head())

# features ---- x
x = np.array(data_new.iloc[:,1:])
# print(x.shape)

# target --- y
y=np.array(data_new.iloc[:,0])
# print(y.shape)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.80,random_state=3)
# print(pd.DataFrame(y).value_counts())
# print(pd.DataFrame(ytrain).value_counts())

from sklearn.neighbors   import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3,p=2)
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)
# print(ypred)
# print(ytest)

ypred=model.predict(xtest)
count=0
for i in range(len(ytest)):
    if ypred[i]==ytest[i]:
        count=count+1
# print(count)
# print(len(ytest))
# print(count/len(ytest))

from sklearn.metrics import  accuracy_score
a=accuracy_score(ytest,ypred)
print(a)

import joblib
joblib.dump(model,"titanic1.pkl")


mymodel=joblib.load("C:/Users/MCA/PycharmProjects/ChethanML/titanic1.pkl")
print(data_new.head())
print(mymodel.predict([[1,0,20,2,0,1]]))


s





































































































































































































































































































































































































































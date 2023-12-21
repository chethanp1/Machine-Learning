import pandas as pd
import numpy as np

data = (pd.read_csv("C:/Users/MCA/Desktop/ML Lab Dataset/movies.csv"))

#print(type(data))
#print(data)
#print(data.tail(3))

# print(data.sample(random_state=3))
# print(data.info())
# print(data.isnull().sum())
#
# # Working with missing values
# data_1=data.dropna(axis=0,how="all")
# print(len(data))
# print(len(data_1))

# data_1=data.dropna(axis=0,how="any")
# print(len(data_1))

#Deleteing rows
# data_1=data.dropna(axis=0,how="all",subset=["GENRE"])
# print(len(data_1))

# print(data.isnull().sum())

data=data.drop(["Gross"],axis=1)
# print(data.isnull().sum())

# print(data['VOTES'])
data['VOTES']=data['VOTES'].fillna("0")
# print(data['VOTES'])
# print(data.isnull().sum())

# # print(data['RunTime'])
meanRT=data['RunTime'].mean()
# # print(meanRT)
meanRT=round(meanRT,1)
# # print(meanRT)
data['RunTime']=data['RunTime'].fillna(meanRT)
# # print(data['RunTime'])
# print(data.isnull().sum())


# print(data['RATING'])
meanRATING=data['RATING'].mean()
# print(meanRATING)
meanRATING=round(meanRATING,1)
# print(meanRATING)
data['RATING']=data['RATING'].fillna(meanRATING)
# print(data['RATING'])
print(data.isnull().sum())

# print(data['GENRE'])
data['GENRE']=data['GENRE'].fillna('Comedy')
# print(data["GENRE"])
# print(data.isnull().sum())


# print(data['YEAR'])
data['YEAR']=data['YEAR'].fillna('1999')
# print(data["YEAR"])
print(data.isnull().sum())
S

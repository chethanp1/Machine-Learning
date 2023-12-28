import pandas as pd
import numpy as np

data=pd.read_csv("C:/Users/MCA/Desktop/ML Lab Dataset/enjoysport.csv")
# print(data)

concepts=np.array(data)[:,:-1]
# print(concepts)

target=np.array(data)[:,-1]
# print(target)

def train(con, tar):
    for i, val in enumerate(tar):
        if val == 'yes':
            specific_h = con[i].copy()
    for i, val in enumerate(con):
        if tar[i] == 'yes':
            for x in range(len(specific_h)):
                if val[x] != specific_h[x]:
                    specific_h[x] = '?'
                else:
                    pass
    return specific_h
print(train(concepts,target))
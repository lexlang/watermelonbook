import pandas as pd
import numpy as np
'''
抄袭别人代码，梯度上升
'''


df=pd.read_csv("watermelon3.0alpha.csv")
df['one']=1

trains=np.mat(df[['one','density','Sugar_content']])
labels=np.mat(df[['label']])

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def gred(trains,labels,iter=100000):
    alpha=0.0001
    m,n=trains.shape
    weights=np.ones((n,1))
    for i in range(iter):
        print("train time:"+str(i))
        p=sigmoid(trains.dot(weights))
        error=labels-p
        weights+=alpha*np.dot(trains.T,error)
    return weights

weights=gred(trains,labels)
print(weights)

import pandas as pd
import numpy as np

df=pd.read_csv("watermelon3.0alpha.csv")

X0=np.mat(df.loc[df['label']==0][['density','Sugar_content']])
X1=np.mat(df.loc[df['label']==1][['density','Sugar_content']])

def LDA(X0,X1):
    u0=np.mean(X0,axis=0)
    u1 = np.mean(X1, axis=0)
    SW=(X0-u0).T.dot(X0-u0)+(X1-u1).T.dot(X1-u1)
    return np.linalg.inv(SW).dot((u0-u1).T)

print(LDA(X0,X1))

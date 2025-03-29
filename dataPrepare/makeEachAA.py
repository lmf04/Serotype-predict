import numpy as np
import pandas as pd
hmtp210=pd.read_csv("D:/Python/ML/dataPrepare/hmtp210.csv")
eachAA = np.zeros((54, 361))
eachAA =pd.DataFrame(eachAA)
x=0
for seq in hmtp210.iloc[0:,3]:
    seq=list(seq)
    y=0
    for aa in seq:
        eachAA.iat[x, y] = aa
        y=y+1
    x=x+1

Eachhmtp210= pd.concat([hmtp210.iloc[:,:3], eachAA],axis=1)

Eachhmtp210.to_csv("D:/Python/ML/dataPrepare/eachAA.csv",index=False)
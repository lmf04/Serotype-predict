import numpy as np
import pandas as pd

data=pd.read_csv("D:/Python/predictCut - 副本.csv")
data.insert(data.shape[1],"mostPossible",0)

for index in range(0,len(data)):
    if data.iloc[index,2]>data.iloc[index,3] and data.iloc[index,2]>data.iloc[index,4]:
        Largest=data.iloc[index,2]
        if data.iloc[index,4]>data.iloc[index,3]:
            second=data.iloc[index,4]
        else:
             second=data.iloc[index,3]
    elif data.iloc[index,4]>data.iloc[index,2] and data.iloc[index,4]>data.iloc[index,3]:
        Largest=data.iloc[index,4]
        if data.iloc[index,2]>data.iloc[index,3]:
            second=data.iloc[index,2]
        else:
            second=data.iloc[index,3]        
    elif data.iloc[index,3]>data.iloc[index,2] and data.iloc[index,4]>data.iloc[index,4]:
        Largest=data.iloc[index,3]
        if data.iloc[index,4]>data.iloc[index,2]:
            second=data.iloc[index,4]
        else:
            second=data.iloc[index,2]

    if Largest>=second*3:
        confidence=2
    elif Largest>=second*1.5:
        confidence=1
    else:
        confidence=0

    data.iat[index,5]=confidence



data.to_csv("D:/Python/predictCutF.csv",index=False)
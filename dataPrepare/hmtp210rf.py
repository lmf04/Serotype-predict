import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

hmtp210 = pd.read_csv("D:\Python\hmtp210.csv")
print(hmtp210)
header = hmtp210.columns.tolist()  
print(header)

aaindex = pd.read_csv("D:/Python/RF/aaindex1.csv")
#print(aaindex)
Hydrophobicity= aaindex[aaindex['Description'] == "ARGP820101"]


for r in Hydrophobicity.columns[1:,]:
    print(r=='R')
def has(v):
    rr=False
    for r in Hydrophobicity.columns[1:,]:
        if(r==v):
            rr=True
            break
    return rr
print(has('R'))


for index in range(1, 1637):

    hmtp210.insert(hmtp210.shape[1], index, 0)

hmtp210.insert(hmtp210.shape[1], 1638, 0)
hmtp210.insert(hmtp210.shape[1], 1639, 0)
 

for index,row in hmtp210.iterrows():
    list=[]
    print(row['als'])
    count=4
    for char in row['als']:
        #print(char)
        count=count+1
        if has(char):
            print(char)
            print(str(float( Hydrophobicity.iloc[0][char])))
            hmtp210.iat[index,count]=float(Hydrophobicity.iloc[0][char])
            list.append(str(float( Hydrophobicity.iloc[0][char])))
        else:
            hmtp210.iat[index,count]=0
            list.append('0')
    #print(','.join(list))
    #hmtp210.iat[index,4]=','.join(list)
    print(list.__len__())
    #print(row['result'])
#print(hmtp210.loc[5])
hmtp210.to_csv("D:/Python/RF/test2.csv")

 


 



polarity = aaindex[aaindex['Description'] == "GRAR740102"]
print(polarity)

def sequence_to_features(sequence, matrix):
    # 假设序列中的每个氨基酸在矩阵中都有对应的行
    # 这里简单地取每个氨基酸对应行的平均值作为特征（这只是一个示例）
    # 在实际应用中，你可能需要设计更复杂的特征提取方法
    features = []
    for aa in sequence:
        row = matrix.loc[aa]  # 获取氨基酸对应的行
        features.append(row.values.mean())  # 计算平均值并添加到特征列表中
    # 将特征列表转换为NumPy数组（或Pandas Series），以便后续处理
    return pd.Series(features)
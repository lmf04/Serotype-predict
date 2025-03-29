from joblib import dump, load
import pandas as pd
import numpy as np
 
data = np.load('D:/Python/ML/UN/UNhmtp210.npy',allow_pickle=True)

x=data[:,2:]
modelSVM = load('D:/Python/ML/SVC/SVMmodelV2.joblib')

#制作预测结果与实际结果的对比表格
data=pd.DataFrame(data)
sequence=data.iloc[:,1]
data = data.iloc[:, :1]  # 使用iloc按位置选择列
data = data.rename(columns={0: 'RefSeqID'})
data.insert(data.shape[1], "predSero", 0)


for i in range(0,346):
    eachAAseq=x[i,:].reshape(1, -1)
    print(eachAAseq)
    seroPred = modelSVM.predict(eachAAseq)
    data.iat[i,1]=seroPred



# 置信度可以是决策函数值的绝对值（距离决策边界的远近）
Confidence= modelSVM.decision_function(x)
proba=modelSVM.predict_proba(x)
    #print(proba)
proba=pd.DataFrame(proba)
Confidence=pd.DataFrame(Confidence)
data = pd.concat([data,proba,Confidence,sequence],axis=1) 
data.to_csv("D:/Python/ML/result/predUNHMTp210PC.csv",index=False)

#画图

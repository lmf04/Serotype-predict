from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

numhmtp210=pd.read_csv("D:/Python/trainHmtp.csv",header=None)
numhmtp210=numhmtp210.values

# x = numhmtp210[:,4:]
# y = numhmtp210[:,0:3]

x = numhmtp210[:,4:]
y = numhmtp210[:,0:3]

count=0
for item in y[:,2]:

    y[count,2]=str(item)
    count=count+1

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

#print('Class labels:', np.unique(y))   
clf = DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42)
 # SVC 建模：使用 SVC类，线性核函数
# modelSVM = LinearSVC(C=100)  # SVC 建模：使用 LinearSVC类，运行结果同上
clf.fit(x_train,y_train[:,2])  # 用样本集 X,y 训练 SVM 模型

print('分类准确度：{:.4f}'.format(clf.score(x, y[:,2])))  # 对
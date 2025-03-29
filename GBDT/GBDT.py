import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold #k折交叉验证
from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.metrics import make_scorer
from sklearn.ensemble import BaggingClassifier,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor

numhmtp210=pd.read_csv("D:/Python/ML/sourceData/numhmtp210.csv")
numhmtp210 = numhmtp210.values
x = numhmtp210[:,4:]
y = numhmtp210[:,2]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

print('Class labels:', np.unique(y))   
y_train
# knn=KNeighborsClassifier()
# knn.fit(x_train,y_train)

gbdt =DecisionTreeRegressor(n_estimators=100, 
                                 learning_rate=0.1)
gbdt.fit(x_train,y_train)

# bag_knn = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=100, max_samples=0.8,
#                             max_features=0.7)
# bag_knn.fit(x_train,y_train)
print('KNN集成算法，得分是：', gbdt.score(x_test,y_test))
print('KNN集成算法，得分是：', gbdt.score(x_test,y_test))
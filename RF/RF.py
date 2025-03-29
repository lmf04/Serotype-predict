import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

numhmtp210=pd.read_csv("D:/Python/ML/sourceData/numhmtp210.csv",header=None)

numhmtp210 = numhmtp210.values

print(type(numhmtp210))



x = numhmtp210[:,4:]
y = numhmtp210[:,2]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

forest = RandomForestClassifier(n_estimators=200, random_state=0,max_depth=5)
forest.fit(x_train, y_train)


y_pred = forest.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)



print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)




# x_columns = numhmtp210 .columns[5:]
# x_train=pd.DataFrame(x_train)
# indices = np.argsort(importances)[::-1]
# for f in range(x_train.shape[1]):
# # 对于最后需要逆序排序，我认为是做了类似决策树回溯的取值，从叶子收敛
# # 到根，根部重要程度高于叶子。
#     print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))


# # 筛选变量（选择重要性比较高的变量）
# threshold = 0.15
# x_selected = x_train[:,importances > threshold]


# plt.figure(figsize=(10,6))
# plt.title("各个氨基酸的重要程度",fontsize = 18)
# plt.ylabel("import level",fontsize = 15,rotation=90)
# plt.rcParams['font.sans-serif'] = ["SimHei"]
# plt.rcParams['axes.unicode_minus'] = False
# for i in range(x_columns.shape[0]):
#     plt.bar(i,importances[indices[i]],color='orange',align='center')
#     plt.xticks(np.arange(x_columns.shape[0]),x_columns,rotation=90,fontsize=15)
# plt.show()



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap

numhmtp210=pd.read_csv("D:/Python/trainHmtp1.csv",header=None)
numhmtp210 = numhmtp210.values


x = numhmtp210[:,4:]
y = numhmtp210[:,0:3]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)

#print('Class labels:', np.unique(y))   

modelSVM = SVC(kernel='linear', C=200,probability=True)  # SVC 建模：使用 SVC类，线性核函数
# modelSVM = LinearSVC(C=100)  # SVC 建模：使用 LinearSVC类，运行结果同上
modelSVM.fit(x_train,y_train[:,2])  # 用样本集 X,y 训练 SVM 模型

print("\nSVM model: Y = w0 + w1*x1 + w2*x2") # 分类超平面模型
print('截距: w0={}'.format(modelSVM.intercept_))  # w0: 截距, YouCans
print('系数: w1={}'.format(modelSVM.coef_))  # w1,w2: 系数, XUPT
print('分类准确度：{:.4f}'.format(modelSVM.score(x, y[:,2])))  # 对训练集的分类准确度

from joblib import dump, load
 
# 假设你已经训练好了一个模型，命名为 model
dump(modelSVM, 'D:/Python/ML/SVC/SVMmodel.joblib')
 
# 加载模型
#loaded_model = load('model.joblib')

y_pred = modelSVM.predict(x_test)

# 5. 计算混淆矩阵
cm = confusion_matrix(y_test[:,2], y_pred)
 

# 6. 绘制混淆矩阵
plt.rcParams['font.size'] = 24
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y[:,2]), yticklabels=np.unique(y[:,2]))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for SVM on HMTp210 Dataset',y=1.03)
plt.show()

y_pred = modelSVM.predict(x)
cm = confusion_matrix(y[:,2], y_pred)
 

# 6. 绘制混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y[:,2]), yticklabels=np.unique(y[:,2]))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for SVM on Iris Dataset')
plt.show()


#制作预测结果与实际结果的对比表格
dfhmtp210=pd.DataFrame(numhmtp210)
sequence=dfhmtp210.iloc[:,3]
dfhmtp210 = dfhmtp210.iloc[:, :3]  # 使用iloc按位置选择列
dfhmtp210 = dfhmtp210.rename(columns={0: 'EntrezID', 1: 'strain',2:"serotype"})
dfhmtp210.insert(dfhmtp210.shape[1], "predSero", 0)
 
for i in range(0,54):
    eachAAseq=x[i,:].reshape(1, -1)
    seroPred = modelSVM.predict(eachAAseq)
    dfhmtp210.iat[i,3]=seroPred


dfhmtp210 = pd.concat([dfhmtp210,sequence],axis=1) 
dfhmtp210.to_csv("D:/Python/result/predResHMTp210.csv",index=False)


# 获取DataFrame的列名

xdf=pd.DataFrame(x)
feature_names = xdf.columns
feature_names
# 将列名转换为字符串列表（通常已经是字符串，但这一步可以确保类型）

feature_names = [int(col)+1 for col in feature_names]
feature_names= [str(col) for col in feature_names]
result = permutation_importance(modelSVM, x, y[:,2], scoring='accuracy', n_repeats=30, random_state=42, n_jobs=-1)


importance=np.zeros((267,2))
importance=pd.DataFrame(importance)

# 绘制特征重要性
for i in range(x.shape[1]):
    print(f"Feature {feature_names[i]} importance: {result["importances_mean"][i]} +/- {result["importances_std"][i]}")
    importance.iat[i,0]=feature_names[i]
    importance.iat[i,1]=f"{result["importances_mean"][i]}+/-{result["importances_std"][i]}"
importance.to_csv("D:/Python/importance.csv",index=False)

# 绘制条形图
plt.rcParams['font.size'] = 12
plt.figure(figsize=(10, 80))
plt.title("Permutation Importance")
plt.barh(range(x.shape[1]), result["importances_mean"], color="b", xerr=result["importances_std"])

plt.yticks(range(x.shape[1]), feature_names)
plt.xlabel("Importance (decrease in accuracy)")
plt.savefig('D:/Python/SVCPermutationImportanceH2.png')
plt.show()

#用SHAP解释模型
# shap.initjs()
# shap_values = shap.TreeExplainer(modelSVM).shap_values(x_train).shap.summary_plot(shap_values, x_train)
# shap.summary_plot(shap_values[0], x_train)

# explainer = shap.TreeExplainer(modelSVM)
# shap_value_single = explainer.shap_values(x = x_train.iloc[0,:])
# shap.force_plot(base_value = explainer.expected_value[1],shap_values = shap_value_single[1],features = x_train.iloc[0,:])

# 获取测试集的预测概率
df=pd.read_excel("D:/Python/ML/result/res.xlsx")
counts = df.groupby(['Predicted Serotype', 'Strain']).size().unstack(fill_value=0)
 
# 绘制堆叠条形图
counts.plot(kind='bar', stacked=True)
 
# 设置图表标题和标签
plt.subplots(figsize=(10, 6))
plt.title('Bacterial Strain Distribution by Predcited Serotype')
plt.xlabel('Strain Name')
plt.ylabel('Count')
plt.legend(title='Predicted Serotype')
 
# 显示图表
plt.show()






 
# 为了绘图，我们需要将数据转换为适合分组条形图的形式
# 我们将创建一个新的DataFrame，其中包含每种细菌种类（实际和预测）和对应的菌株类型计数
# 注意：这里我们假设每个菌株类型在每个细菌种类下只有一个样本，因此计数为1
# 在实际应用中，您可能需要根据您的数据计算计数
 
# 为实际种类创建DataFrame
actual_counts = df.groupby(['Serotype', 'Strain']).size().reset_index(name='Count')
actual_counts['Type'] = 'Serotype'  # 添加一个列来区分实际和预测
 
# 为预测种类创建DataFrame
predicted_counts = df.groupby(['Predicted Serotype', 'Strain']).size().reset_index(name='Count')
predicted_counts['Type'] = 'Predicted Serotype'  # 添加一个列来区分实际和预测
 
# 合并两个DataFrame
combined = pd.concat([actual_counts, predicted_counts])

fig, ax = plt.subplots(figsize=(10, 6))
width = 0.4  # 条形的宽度
position = np.arange(len(combined[combined['Type'] == 'Serotype']))
len(combined['Serotype'].unique()) 
combined['Serotype']
position
# 绘制实际种类的条形图
actual_subset = combined[combined['Type'] == 'Serotype']
ax.bar(position -width/2, actual_subset['Count'], width, label='Actual', align='edge')
 
# 绘制预测种类的条形图
predicted_subset = combined[combined['Type'] == 'Predicted Serotype']
ax.bar(position +width/2, predicted_subset['Count'], width, label='Predicted', align='edge')
 
# 设置x轴标签
ax.set_xticks(position)
ax.set_xticklabels(combined[combined['Type'] == 'Serotype'])  # 这里应该使用实际种类的唯一值作为标签
# 注意：由于我们合并了实际和预测的数据，这里的标签可能不完全对应预测种类，但在这个例子中，我们假设种类数量相同且顺序一致
q=combined[combined['Type'] == 'Serotype']
# 添加图例
ax.legend()
 
# 设置图表标题和标签
ax.set_title('Bacterial Classification by Actual and Predicted Classes')
ax.set_xlabel('Bacterial Class')
ax.set_ylabel('Count')
 
# 显示图表
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()
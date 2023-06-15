#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import DecisionTreeClassifier #分类树
from sklearn.ensemble import RandomForestClassifier #随机森林在ensemble模块下，这里是分类器
from sklearn.model_selection import train_test_split #划分测试集与分类集
from sklearn.model_selection import cross_val_score #交叉验证
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np 
import pandas as pd


# # 1.读取数据

# In[2]:


choose_gcy_or_qcy=0
yes_choose_new_feature=0
yes_pca=1
pca_kind_feature=30
dict1=[17,70,86]
dict2=[2,4,23,30,36,38,41,48,58,64,79]
# #定义数据集地址
# dir_name="./zx_data/train.csv"
# dir_name1="./zx_data/pre_contest_test1.csv"
#
# # 读入数据
# df_train=pd.read_csv(dir_name)
# df_test=pd.read_csv(dir_name1)




#定义数据集地址
dir_name="./zx_data/train.csv"

# 读入数据
df = pd.read_csv(dir_name)
train_threshold = int(df.shape[0] * 0.9)
df_train = df[0:train_threshold]
df_test  = df[train_threshold:]
print(df_test.head)
df_test = df_test.reset_index(drop=True)  # 需要重置行索引
print(df_test.head)
# exit()

# # 2.数据分析

# # 3.数据预处理

# ## 3.1 补全缺失数据RandomForestClassifier

# In[3]:

"""
训练集，平均数填充
"""
dict_unique={}  # 为1表明特征值重复过多，或者空值非常多的特征
dict_label={}
for i in df_train.columns:
    # 给特征值重复较多，或者空值非常多的特征，补全缺失数据
    if i=='sample_id' or i=='label':
        continue
    a=df_train[i]
    ddd={}
    for j in range(6):
        ddd[j]=[]
    for j in range(len(df_train[i])):
        if pd.isnull(df_train[i][j])==False:
            if df_train['label'][j]==0 or df_train['label'][j]==1:
                ddd[df_train['label'][j]].append(df_train[i][j])
            else:
                ddd[df_train['label'][j]]=df_train[i][j]
                
    if len(a.unique())<=200 and len(a.unique())!=1:  # 属性重复值
        print("==========",i,len(a.unique()),"========")
        dict_unique[i]=1
        for j in range(len(df_train[i])):
            if pd.isnull(df_train[i][j])==True:
                if df_train['label'][j]==0 or df_train['label'][j]==1:#用众数补充
#                     df_train[i][j]=max(set(ddd[df_train['label'][j]]), key=ddd[df_train['label'][j]].count)
                    df_train[i][j]=sum(ddd[df_train['label'][j]])/len(ddd[df_train['label'][j]])
                else:
                    df_train[i][j]=ddd[2]
                    dict_label[i]=ddd[2]
print("success")


# In[4]:

"""
测试集，众数填充
"""
dict_label2={}
for i in dict_unique.keys():
    aa=[]
    for j in range(len(df_test[i])):
        if pd.isnull(df_test[i][j])==False:
            aa.append(df_test[i][j])
    dict_label2[i]=max(set(aa), key=aa.count)  # 得到对应的众数
for i in df_test.columns:
    if i=='sample_id' or i=='label':
        continue
    a=df_test[i]
    if len(a.unique())<=200 and len(a.unique())!=1:
        print("==========",i,len(a.unique()),"========")
        dict_unique[i]=1
        numm=0
        for j in range(len(df_test[i])):
            if pd.isnull(df_test[i][j])==True:
                numm=numm+1
#             for k in dict_unique:
#                 if pd.isnull(df_test[k][j])==False:
#                     print(df_test[k][j],dict_label2[k])
#             print()
#         print(numm)
        for j in range(len(df_test[i])):
            if pd.isnull(df_test[i][j])==True:
                    df_test[i][j]=dict_label2[i]


# In[5]:


from sklearn.impute import SimpleImputer
import numpy as np
# 使用了SimpleImputer类来处理缺失值，均值填充
im = SimpleImputer(missing_values=np.NaN, strategy='mean')
result = im.fit_transform(df_train)
result_test=im.fit_transform(df_test)


# In[6]:


print(len(result))


# In[7]:


X_sum=[]
x_train,y_train=[],[]
x_test,y_test=[],[]
for i in result:
    cc=[]
    for j in range(1,105):
        if j-1 not in dict1:
            cc.append(float(i[j]))
    x_train.append(cc)
    X_sum.append(cc)
    y_train.append(int(i[106]))
for i in result_test:
    cc=[]
    for j in range(1,105):
        if j-1 not in dict1:
            cc.append(float(i[j]))
    x_test.append(cc)
    X_sum.append(cc)
    y_test.append(int(i[106]))
print(len(x_train),len(x_test),len(y_train))


# ## 3.2 特征正则化处理

# In[8]:


def zssd(arr, epsilon = 1e-12):
    return (arr - np.mean(arr, axis=0)) / (np.std(arr, axis=0) + epsilon)

scalers_func = {'zssd':zssd}

def normalization(arr,norm_type):
    return scalers_func[norm_type](arr)

x_train=normalization(x_train, 'zssd')
x_test=normalization(x_test, 'zssd')
print(len(x_train),len(x_test),len(y_train))


# ## 3.3 根据互信息筛选特征

# In[9]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

if yes_choose_new_feature==1:
    result=mutual_info_classif(x_train,y_train)
    dele_list=[]
    for i in range(len(result)):
        if result[i]<=0.01:
            dele_list.append(i)

    x_train=np.delete(x_train,dele_list,axis=1)
    x_test=np.delete(x_test,dele_list,axis=1)

    result.sort()
    print(result)
    if len(x_train[0])!=len(x_test[0]):
        print("错误！！！！！")
    print("经过互信息筛选后的特征维度=",len(x_train[0]))
else:
    print("不经过互信息筛选,并且特征维度=",len(x_train[0]))


# In[10]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# if choose_gcy_or_qcy==0:
#     #过采样
#     model_smote=SMOTE()
#     x_train,y_train=model_smote.fit_resample(x_train,y_train)
#     print("已对数据进行过采样")
# else:
#     #欠采样
#     model_RandomUnderSampler=RandomUnderSampler() 
#     x_train,y_train=model_RandomUnderSampler.fit_resample(x_train,y_train)
#     print("已对数据进行欠采样")
print(len(x_train),len(x_test),len(y_train))
X_sum=[]
for i in x_train:
    X_sum.append(i)
for i in x_test:
    X_sum.append(i)


# ## 3.4 PCA 特征降维

# In[11]:


from sklearn.decomposition import PCA #导入主成分分析库

if yes_pca==1:
    pca = PCA(n_components=pca_kind_feature)
    fit= pca.fit(X_sum)
    print("已对数据进行PCA特征降维，且维度=",pca_kind_feature)
else:
    print("不进行pca降维")
print(len(x_train),len(x_test),len(y_train))


# ## 3.5 再次划分训练集、测试集+对训练集采样处理

# In[12]:


x_train=X_sum[:len(y_train)]
x_test=X_sum[len(y_train):]

# X_sum_train=[[],[],[],[],[],[]]
# Y_sum_train=[[],[],[],[],[],[]]

# for i in range(len(y_train)):
#     label1=y_train[i]
#     X_sum_train[label1].append(x_train[i])
# #     if y_train[i]==0 or y_train[i]==1:
# #         Y_sum_train[label1].append(0)
# #     else:
# #         Y_sum_train[label1].append(1)
#     Y_sum_train[label1].append(y_train[i])

# x_train,x_valid=[],[]
# y_train,y_valid=[],[]
# for i in range(6):
#     x_train1, x_valid1, y_train1, y_valid1 = train_test_split(X_sum_train[i],Y_sum_train[i],test_size=0.2,random_state=3)
#     x_train+=x_train1
#     y_train+=y_train1
#     x_valid+=x_valid1
#     y_valid+=y_valid1


# In[13]:


ans=[0,0,0,0,0,0]
for i in y_train:
    ans[i]=ans[i]+1
print(ans)


# # 4.随机森林模型训练

# In[19]:


from imblearn . ensemble import EasyEnsembleClassifier
# 进行类别不平衡数据的分类任务
model3 = EasyEnsembleClassifier(n_estimators=100,random_state=3,base_estimator=RandomForestClassifier(random_state=0,n_estimators=60))
model3.fit(x_train,y_train)


# In[20]:


# score_system = model3.score(x_valid,y_valid)
# print("系统估算验证集上的分数为=",score_system)


# # 5.预测测试集

# In[21]:


pred_y=model3.predict(x_test)
ans=[0,0,0,0,0,0]
sum1=0
for i in pred_y:
    ans[i]=ans[i]+1
for i in ans:
    print(max(i-167,167-i),end=",")
    sum1=sum1+max(i-167,167-i)
print()
print(sum1)
print(ans)


# In[15]:


num=0
s="{"
for i in pred_y:
    s=s+'"'+str(num)+'":'+str(i)+","
    num=num+1
s=s[:len(s)-1]+"}"
print(s)

from sklearn.metrics import accuracy_score

print("随机森林预测正确率")
print(accuracy_score(y_test, pred_y))


# In[ ]:





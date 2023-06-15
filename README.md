# zhongxinpy
智能运维赛题





### 问题记录

1. 模型在训练过程中出现`Train Loss`与`Val Loss`都是nan的情况
   * 原因：训练的数据有的特征存在值都是一样的，这样会造成求导为0，无法学习
   * 解决方法：去掉对应特征值
   
2. epoch超过5以后，训练集损失值下降明显，验证集损失值下降非常缓慢，导致验证集的准确率基本维持在60%.

3. 在提出问题2时，师兄给了一些建议，包括把预测结果和原来结果进行对比，我觉得这个思路很好，但是我对比以后没有更好的方法，用到的feature有91个，得到的结果对模型进一步没有帮助，但是发现我对数据是`归一化`，而不是`标准化`。

4. 数据类别数量不平衡问题，

   * 卷积采用上采样

   * 随机森林用的

     * ```
       # 进行类别不平衡数据的分类任务
       model3 = EasyEnsembleClassifier(n_estimators=100,random_state=3,base_estimator=RandomForestClassifier(random_state=0,n_estimators=60))
       ```



### 两种方法

1. 卷积预测`main.py`，精确度为0.6697
2. 随机森林预测，`pytorch_easyensemble.py`，精确度为0.8698
3. 
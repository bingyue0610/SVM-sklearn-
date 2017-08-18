#coding: utf-8
'''
l  C：C-SVC的惩罚参数C?默认值是1.0

C越大，相当于惩罚松弛变量，希望松弛变量接近0，即对误分类的惩罚增大，趋向于对训练集全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。

l  kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

  　　0 – 线性：u'v

 　　 1 – 多项式：(gamma*u'*v + coef0)^degree

  　　2 – RBF函数：exp(-gamma|u-v|^2)

  　　3 –sigmoid：tanh(gamma*u'*v + coef0)

l  degree ：多项式poly函数的维度，默认是3，选择其他核函数时会被忽略。

l  gamma ： ‘rbf’,‘poly’ 和‘sigmoid’的核函数参数。默认是’auto’，则会选择1/n_features

l  coef0 ：核函数的常数项。对于‘poly’和 ‘sigmoid’有用。

l  probability ：是否采用概率估计？.默认为False

l  shrinking ：是否采用shrinking heuristic方法，默认为true

l  tol ：停止训练的误差值大小，默认为1e-3

l  cache_size ：核函数cache缓存大小，默认为200

l  class_weight ：类别的权重，字典形式传递。设置第几类的参数C为weight*C(C-SVC中的C)

l  verbose ：允许冗余输出？

l  max_iter ：最大迭代次数。-1为无限制。

l  decision_function_shape ：‘ovo’, ‘ovr’ or None, default=None3

l  random_state ：数据洗牌时的种子值，int值

主要调节的参数有：C、kernel、degree、gamma、coef0
'''

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

#example 1
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
clf = SVC()
clf.fit(X, y)
print(clf.predict([[-0.8, -1]]))

'''

decision_function(X):	Distance of the samples X to the separating hyperplane.
fit(X, y[, sample_weight]):	Fit the SVM model according to the given training data.
get_params([deep]):	Get parameters for this estimator.
predict(X):	Perform classification on samples in X.
score(X, y[, sample_weight]):	Returns the mean accuracy on the given test data and labels.
set_params(**params):	Set the parameters of this estimator.

clf.predict_proba & clf.predict_log_proba可以用的条件是：
在训练模型时，选择参数 probability=Ture, default:probability=False

svc.n_support_：各类各有多少个支持向量
svc.support_：各类的支持向量在训练样本中的索引
svc.support_vectors_：各类所有的支持向量
'''

print clf.support_vectors_  # support vectors
print clf.support_  # indeices of support vectors
print clf.n_support_  # number of support vectors for each class

#example 2
X=np.array([[1,1],[1,2],[1,3],[1,4],[2,1],[2,2],[3,1],[4,1],[5,1],

       [5,2],[6,1],[6,2],[6,3],[6,4],[3,3],[3,4],[3,5],[4,3],[4,4],[4,5]])
Y=np.array([1]*14+[-1]*6)
T=np.array([[0.5,0.5],[1.5,1.5],[3.5,3.5],[4,5.5]])
svc=SVC(kernel='poly',degree=2,gamma=1,coef0=0)
svc.fit(X,Y)
pre=svc.predict(T)
print pre
print svc.n_support_
print svc.support_
print svc.support_vectors_
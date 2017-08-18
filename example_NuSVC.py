#coding:utf-8
'''
nu：训练误差的一个上界和支持向量的分数的下界。应在间隔（0，1 ]。

'''
#example 1
import numpy as np
from sklearn.svm import NuSVC
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
clf = NuSVC()
clf.fit(X, y)
print(clf.predict([[-0.8, -1]]))

#example 2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# from sklearn.linear_model import SGDClassifier

# we create 40 separable points
rng = np.random.RandomState(0)
n_samples_1 = 1000
n_samples_2 = 100
X = np.r_[1.5 * rng.randn(n_samples_1, 2), 0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
y = [0] * (n_samples_1) + [1] * (n_samples_2)
print X
print y

# fit the model and get the separating hyperplane
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

w = clf.coef_[0]
a = -w[0] / w[1]  # a可以理解为斜率
xx = np.linspace(-5, 5)
yy = a * xx - clf.intercept_[0] / w[1]  # 二维坐标下的直线方程

# get the separating hyperplane using weighted classes
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)

ww = wclf.coef_[0]
wa = -ww[0] / ww[1]
wyy = wa * xx - wclf.intercept_[0] / ww[1]  # 带权重的直线

# plot separating hyperplanes and samples
h0 = plt.plot(xx, yy, 'k-', label='no weights')
h1 = plt.plot(xx, wyy, 'k--', label='with weights')
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.legend()

plt.axis('tight')
plt.show()

'''
support_ : array-like, shape = [n_SV]
Indices of support vectors.
support_vectors_ : array-like, shape = [n_SV, n_features]
Support vectors.
n_support_ : array-like, dtype=int32, shape = [n_class]
Number of support vectors for each class.
dual_coef_ : array, shape = [n_class-1, n_SV]
Coefficients of the support vector in the decision function. For multiclass, coefficient for all 1-vs-1 classifiers.
The layout of the coefficients in the multiclass case is somewhat non-trivial. See the section about multi-class
classification in the SVM section of the User Guide for details.
coef_ : array, shape = [n_class-1, n_features]
Weights assigned to the features (coefficients in the primal problem). This is only available in the case of a linear kernel.
coef_ is readonly property derived from dual_coef_ and support_vectors_.
intercept_ : array, shape = [n_class * (n_class-1) / 2]
Constants in decision function.
'''
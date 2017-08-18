#coding:utf-8
'''
Parameters:
penalty : string, ‘l1’ or ‘l2’ (default=’l2’)
Specifies the norm used in the penalization. The ‘l2’ penalty is the standard used in SVC. The ‘l1’ leads to c
oef_ vectors that are sparse.

loss : string, ‘hinge’ or ‘squared_hinge’ (default=’squared_hinge’)
Specifies the loss function. ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’
is the square of the hinge loss.

dual : bool, (default=True)
Select the algorithm to either solve the dual or primal optimization problem. Prefer dual=False when n_samples >
 n_features.

tol : float, optional (default=1e-4)
Tolerance for stopping criteria.

C : float, optional (default=1.0)
Penalty parameter C of the error term.

multi_class : string, ‘ovr’ or ‘crammer_singer’ (default=’ovr’)
Determines the multi-class strategy if y contains more than two classes. "ovr" trains n_classes one-vs-rest classifiers
 while "crammer_singer" optimizes a joint objective over all classes. While crammer_singer is interesting from a
 theoretical perspective as it is consistent, it is seldom used in practice as it rarely leads to better accuracy and
 is more expensive to compute. If "crammer_singer" is chosen, the options loss, penalty and dual will be ignored.

fit_intercept : boolean, optional (default=True)
Whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations (i.e.
data is expected to be already centered).

intercept_scaling : float, optional (default=1)
When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling], i.e. a “synthetic”
feature with constant value equals to intercept_scaling is appended to the instance vector. The intercept becomes
intercept_scaling * synthetic feature weight Note! the synthetic feature weight is subject to l1/l2 regularization as
all other features. To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept)
intercept_scaling has to be increased.

class_weight : {dict, ‘balanced’}, optional
Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one.
The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies
 in the input data as n_samples / (n_classes * np.bincount(y))

verbose : int, (default=0)
Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in liblinear that,
if enabled, may not work properly in a multithreaded context.

random_state : int, RandomState instance or None, optional (default=None)
The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by
 the random number generator; If RandomState instance, random_state is the random number generator; If None, the random
  number generator is the RandomState instance used by np.random.

max_iter : int, (default=1000)
The maximum number of iterations to be run.

Attributes:
coef_ : array, shape = [n_features] if n_classes == 2 else [n_classes, n_features]
Weights assigned to the features (coefficients in the primal problem). This is only available in the case of a linear kernel.
coef_ is a readonly property derived from raw_coef_ that follows the internal memory layout of liblinear.

intercept_ : array, shape = [1] if n_classes == 2 else [n_classes]
Constants in decision function.
'''
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
X, y = make_classification(n_features=4, random_state=0)
clf = LinearSVC(random_state=0)
clf.fit(X, y)
print(clf.coef_)
print(clf.intercept_)
print(clf.predict([[0, 0, 0, 0]]))

'''
decision_function(X):	Predict confidence scores for samples.
densify():	Convert coefficient matrix to dense array format.
fit(X, y[, sample_weight]):	Fit the model according to the given training data.
get_params([deep]):	Get parameters for this estimator.
predict(X):	Predict class labels for samples in X.
score(X, y[, sample_weight]):	Returns the mean accuracy on the given test data and labels.
set_params(**params):	Set the parameters of this estimator.
sparsify():	Convert coefficient matrix to sparse format.
'''
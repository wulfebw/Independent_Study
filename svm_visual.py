"""
filename: svm_visual.py
date: 5/21/2014
author: Blake

This file implements the SVM model. Uses scikit-learn which provides great documentation:
http://scikit-learn.org/stable/modules/svm.html
"""

import numpy as np
import pylab as pl
import csv
import matplotlib.pyplot as plt

from sklearn import svm, preprocessing, metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import LeaveOneOut
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.dummy import DummyClassifier

from file_utils import load_csv_svm

# Constants
C = 1000 
GAMMA = 1 
WEIGHT = 'auto' 
KERNEL = 'rbf'
USE_PCA = False
PCA_COMPONENTS = 8

def visual_ssp_svm_driver(inputfile):
	x, y, labels = load_csv_svm(inputfile)
	x_scaled = preprocessing.scale(x)

	if USE_PCA:
		pca = PCA(n_components=PCA_COMPONENTS)
		x = pca.fit_transform(x_scaled)
		print(pca.explained_variance_ratio_) 
	else:
		x = x_scaled

	clf = svm.SVC(gamma=GAMMA, C=C, class_weight=WEIGHT, kernel=KERNEL, cache_size=600)
	cv = cross_validation.StratifiedKFold(y, 10)
	#cv = cross_validation.LeavePOut(len(y), len(y)/10)
	metric = 'accuracy' # accuracy, precision, recall, f1
	scores = cross_validation.cross_val_score(clf, x, y, cv=cv, scoring=metric) # 'precision' 'recall' 'f1'
	print('{2}\tmean: {0}\tstd: {1}'.format(scores.mean(), scores.std(), metric))

	metric = 'f1' # accuracy, precision, recall, f1
	scores = cross_validation.cross_val_score(clf, x, y, cv=cv, scoring=metric) # 'precision' 'recall' 'f1'
	print('{2}\tmean: {0}\tstd: {1}'.format(scores.mean(), scores.std(), metric))

def svm_param_selection(inputfile):
	"""
	Performs grid search on the SVM params and gives a pretty grid output as well as the 
	optimized params. See: http://scikit-learn.org/stable/modules/grid_search.html

	:type inputfile: string
	:param inputfile: input samples file

	"""

	x, y, labels = load_csv_svm(inputfile)
	x_scaled = preprocessing.scale(x)

	if USE_PCA:
		pca = PCA(n_components=PCA_COMPONENTS)
		x = pca.fit_transform(x_scaled)
		print(pca.explained_variance_ratio_) 
	else:
		x = x_scaled

	C_range = 10. ** np.arange(-1, 7, 1)
	gamma_range = 10. ** np.arange(-6, 3, 1)
	param_grid = dict(gamma=gamma_range, C=C_range)

	grid = GridSearchCV(svm.SVC(kernel=KERNEL, class_weight=WEIGHT, cache_size=1000), param_grid=param_grid, cv=StratifiedKFold(y=y, n_folds=2), scoring='f1')
	grid.fit(x, y)
	print("The best classifier is: ", grid.best_estimator_)

	score_dict = grid.grid_scores_
	scores = [x[1] for x in score_dict]
	scores = np.array(scores).reshape(len(C_range), len(gamma_range))

	pl.figure(figsize=(8, 6))
	pl.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
	pl.imshow(scores, interpolation='nearest', cmap=pl.cm.spectral)
	pl.xlabel('gamma')
	pl.ylabel('C')
	pl.colorbar()
	pl.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
	pl.yticks(np.arange(len(C_range)), C_range)
	pl.show()

def my_plot_learning_curve(inputfile):
	"""

	"""

	x, y, labels = load_csv_svm(inputfile)
	x_scaled = preprocessing.scale(x)

	if USE_PCA:
		pca = PCA(n_components=PCA_COMPONENTS)
		x = pca.fit_transform(x_scaled)
		print(pca.explained_variance_ratio_) 
	else:
		x = x_scaled

	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234, test_size=0.3)
	clf = svm.SVC(gamma=GAMMA, C=C, class_weight=WEIGHT, kernel=KERNEL, cache_size=400)
	#clf = DummyClassifier(strategy='stratified',random_state=0)


	train_errors = []
	test_errors = []
	train_sizes = range(30,len(x_train)+1, 30)
	for index in train_sizes:
		if index > len(x_train):
			index = len(x_train)
		cur_training_x = x_train[:index]
		cur_training_y = y_train[:index]
		clf.fit(cur_training_x, cur_training_y)
		train_score = clf.score(cur_training_x, cur_training_y)
		train_errors.append(1-train_score)
		test_score = clf.score(x_test, y_test)
		test_errors.append(1-test_score)

	plt.figure()
	plt.xlabel("Training examples")
	plt.ylabel("Score")
	plt.grid()
	plt.plot(train_sizes, train_errors, 'o-', color="r", label="Training error")
	plt.plot(train_sizes, test_errors, 'o-', color="g", label="Cross-validation error")
	plt.legend(loc="best")
	plt.show()


def svm_ssp_metrics(inputfile):
	"""
	This is essentially a helper function which returns all the metrics of an SVM's performance.
	Returns accuracy, precision, recall, F1 Score, confusion matrix

	:type inputfile: string
	:param inputfile: samples file

	:type w: float
	:param w: class weighting
	"""

	x, y, labels = load_csv_svm(inputfile)
	x_scaled = preprocessing.scale(x)

	if USE_PCA:
		pca = PCA(n_components=PCA_COMPONENTS)
		x = pca.fit_transform(x_scaled)
		print(pca.explained_variance_ratio_) 
	else:
		x = x_scaled

	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234,test_size=0.3)
	clf = svm.SVC(gamma=GAMMA, C=C, class_weight=WEIGHT, kernel=KERNEL, cache_size=400)	# gamma=.01, C=.01, 
	y_pred = clf.fit(x_train, y_train).predict(x_test)

	dummy_clf = DummyClassifier(strategy='stratified',random_state=0) # most_frequent, uniform, stratified
	dummy_y_pred = dummy_clf.fit(x_train, y_train).predict(x_test)

	print("\nClassification report for classifier %s:\n\n%s" % (clf, metrics.classification_report(y_test, y_pred)))
	print('Accuracy: {0}\n'.format(accuracy_score(y_test, y_pred)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))
	
	if KERNEL == 'linear':
		print('\nfeature_weights: {0}'.format(clf.coef_))

	print("\nClassification report for classifier %s:\n\n%s" % (dummy_clf, metrics.classification_report(y_test, dummy_y_pred)))
	print('Accuracy: {0}\n'.format(accuracy_score(y_test, dummy_y_pred)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, dummy_y_pred))

def svm_dummy_comparison(inputfile):
	x, y, labels = load_csv_svm(inputfile)
	x_scaled = preprocessing.scale(x)


	if USE_PCA:
		pca = PCA(n_components=PCA_COMPONENTS)
		x = pca.fit_transform(x_scaled)
		print(pca.explained_variance_ratio_) 
	else:
		x = x_scaled

	visual_svm_clf = svm.SVC(gamma=GAMMA, C=C, class_weight=WEIGHT, kernel=KERNEL, cache_size=400)	# gamma=.01, C=.01, 
	dummy_svm_clf = DummyClassifier(strategy='most_frequent',random_state=0) # most_frequent, uniform, stratified

	cv = cross_validation.StratifiedKFold(y, 30)
	#cv = cross_validation.LeaveOneOut(len(y))
	metric = 'f1' # accuracy, precision, recall, f1

	visual_scores = cross_validation.cross_val_score(visual_svm_clf, x, y, cv=cv, scoring=metric)
	dummy_scores = cross_validation.cross_val_score(dummy_svm_clf, x, y, cv=cv, scoring=metric)

	print(metric)
	# print('real_scores: {0}'.format(visual_scores))
	print('avg_real: {0}'.format(np.mean(visual_scores)))
	# print('dummy_scores: {0}'.format(dummy_scores))
	print('avg_dumb: {0}'.format(np.mean(dummy_scores)))


def plot_decision_boundary(inputfile):
	x, y, labels = load_csv_svm(inputfile)

	x_max = 100
	x_min = -100
	y_max = 100
	y_min = -100

	plt.scatter(x[:, 0], x[:, 1], c=y, cmap=pl.cm.Paired)
	plt.show()


def feature_selection(inputfile):
	x, y, labels = load_csv_svm(inputfile)
	x_scaled = preprocessing.scale(x)

	if USE_PCA:
		pca = PCA(n_components=PCA_COMPONENTS)
		x = pca.fit_transform(x_scaled)
		print(pca.explained_variance_ratio_) 
	else:
		x = x_scaled

	selector = SelectPercentile(f_classif, percentile=10)
	selector.fit(x, y)

	x_scaled = preprocessing.scale(x)
	x = x_scaled
	clf = svm.SVC(kernel='linear')
	clf.fit(x, y)

	svm_weights = (clf.coef_ ** 2).sum(axis=0)
	print(svm_weights)

if __name__ == '__main__':
	
	samples_file = 'samples_f.csv'
	
	#visual_ssp_svm_driver(samples_file)
	#svm_ssp_metrics(samples_file)
	#svm_param_selection(samples_file)
	#svm_dummy_comparison(samples_file)
	#my_plot_learning_curve(samples_file)
	#plot_decision_boundary(samples_file)
	#feature_selection(samples_file)


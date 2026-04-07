# -*- coding: utf-8 -*-

import scipy.misc, scipy.io, scipy.optimize
from sklearn import svm
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot(data):
	positives = data[data[:, 2] == 1]
	negatives = data[data[:, 2] == 0]

	plt.plot(positives[:, 0], positives[:, 1], 'b+')
	plt.plot(negatives[:, 0], negatives[:, 1], 'yo')

# 绘制SVM决策边界
def visualize_boundary(X, trained_svm):
	kernel = trained_svm.get_params()['kernel']
	if kernel == 'linear':
		w = trained_svm.coef_[0]#coef_.shape=(classNum,featureNum)=(1,2)  w.shape=(2,)
		i = trained_svm.intercept_#i.shape=(classNum,)=(1,)
		xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
		a = -w[0] / w[1]
		b = i[0] / w[1]
		yp = a * xp - b
		plt.plot(xp, yp, 'b-')
	elif kernel == 'rbf':
		x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
		x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
		
		X1, X2 = np.meshgrid(x1plot, x2plot)
		vals = np.zeros(np.shape(X1))
		
		for i in range(0, np.shape(X1)[1]):
			this_X = np.c_[X1[:, i], X2[:, i]]
			vals[:, i] = trained_svm.predict(this_X)
		
		plt.contour(X1, X2, vals, colors='blue')

def gaussian_kernel(x1, x2, sigma):
	return np.exp(-np.sum((x1 - x2) ** 2) / (2 * sigma ** 2))

def dataset3_params_ver3(X, y, X_val, y_val):
	np.c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
	sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
	gammas = map(lambda x: 1.0 / x, sigma_values)
	
	raveled_y = y.ravel()

	rbf_svm = svm.SVC()
	parameters = {'kernel': ('rbf',), 'C': np.c_values, 'gamma': list(gammas)}
	grid = model_selection.GridSearchCV(rbf_svm, parameters)
	best = grid.fit(X, raveled_y).best_params_

	return best

def dataset2_params_ver2(X, y, X_val, y_val):
	np.c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
	sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

	raveled_y = y.ravel()  # Else the SVM will give you annoying warning
	m_val = np.shape(X_val)[0]  # number of entries in validation data
	
	rbf_svm = svm.SVC(kernel='rbf')

	best = {'score': -999, 'C': 0.0, 'sigma': 0.0}

	for C in np.c_values:
		for sigma in sigma_values:
			# train the SVM first
			rbf_svm.set_params(C=C)
			rbf_svm.set_params(gamma=1.0 / sigma)
			rbf_svm.fit(X, raveled_y)

			score = rbf_svm.score(X_val, y_val)
			
			# get the lowest error
			if score > best['score']:
				best['score'] = score
				best['C'] = C
				best['sigma'] = sigma

	best['gamma'] = 1.0 / best['sigma']
	return best

def params_search(X, y, X_val, y_val):
	np.c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
	sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

	raveled_y = y.ravel()
	m_val = np.shape(X_val)[0]
	
	rbf_svm = svm.SVC(kernel='rbf')

	best = {'error': 999, 'C': 0.0, 'sigma': 0.0}

	for C in np.c_values:
		for sigma in sigma_values:
			# train the SVM first
			rbf_svm.set_params(C=C)
			rbf_svm.set_params(gamma=1.0 / sigma)
			rbf_svm.fit(X, raveled_y)

			# test it out on validation data
			predictions = []
			for i in range(0, m_val):
				prediction_result = rbf_svm.predict(X_val[i].reshape(-1, 2))
				predictions.append(prediction_result[0])

			# sadly if you don't reshape it, numpy doesn't know if it's row or column vector
			predictions = np.array(predictions).reshape(m_val, 1)
			error = (predictions != y_val.reshape(m_val, 1)).mean()
			
			# get the lowest error
			if error < best['error']:
				best['error'] = error
				best['C'] = C
				best['sigma'] = sigma

	best['gamma'] = 1.0 / best['sigma']
	return best

# 线性可分SVM
def part1():
	# --------------- 步骤1 ------------------
	# 加载数据集1
	mat = scipy.io.loadmat("dataset_1.mat")
	X, y = mat['X'], mat['y']#X.shape=(51, 2) y.shape=(51, 1)

	# 绘制数据集1
	plt.title('数据集1分布')
	plot(np.c_[X, y])
	plt.savefig('output/part1_linearSVM/originData.png')
	plt.close()

	# --------------- 步骤2 ------------------
	# 训练线性SVM（C = 1）
	linear_svm = svm.SVC(C=1, kernel='linear')
	linear_svm.fit(X, y.ravel())
	# 绘制C=1的SVM决策边界
	plt.title('C=1的SVM决策边界')
	plot(np.c_[X, y])
	visualize_boundary(X, linear_svm)
	plt.savefig('output/part1_linearSVM/linear_C=1.png')
	plt.close()

	# --------------- 步骤3 ------------------
	# 训练线性SVM（C = 100）
	# your code here
	linear_svm = svm.SVC(C=100, kernel='linear')
	linear_svm.fit(X, y.ravel())
	# 绘制C=100的SVM决策边界
	# your code here
	plt.title('C=100的SVM决策边界')
	plot(np.c_[X, y])
	visualize_boundary(X, linear_svm)
	plt.savefig('output/part1_linearSVM/linear_C=100.png')
	plt.close()

# 非线性可分SVM
def part2():
	# --------------- 步骤1 ------------------
	# 计算高斯核函数
	x1 = np.array([1, 2, 1])
	x2 = np.array([0, 4, -1])
	sigma = 2
	print("样本x1和x2之间的相似度: %f" % gaussian_kernel(x1, x2, sigma))

	# --------------- 步骤2 ------------------
	# 加载数据集2
	mat = scipy.io.loadmat("dataset_2.mat")
	X, y = mat['X'], mat['y']

	# 绘制数据集2
	plt.title('数据集2分布')
	plot(np.c_[X, y])
	plt.show(block=True)

	# 训练高斯核函数SVM
	sigma = 0.01
	rbf_svm = svm.SVC(C=1, kernel='rbf', gamma=1.0 / sigma)  # gamma is actually inverse of sigma
	rbf_svm.fit(X, y.ravel())

	# 绘制非线性SVM的决策边界
	# your code here

# 参数搜索
def part3():
	# --------------- 步骤1 ------------------
	# 加载数据集3和验证集
	mat = scipy.io.loadmat("dataset_3.mat")
	X, y = mat['X'], mat['y']
	X_val, y_val = mat['Xval'], mat['yval']

	# 绘制数据集3
	plt.title('数据集3分布')
	plot(np.c_[X, y])
	plt.show(block=True)

	# 绘制验证集
	plt.title('验证集分布')
	plot(np.c_[X_val, y_val])
	plt.show(block=True)

	# 训练高斯核函数SVM并搜索使用最优模型参数
	rbf_svm = svm.SVC(kernel='rbf')
	# your code here
	
	# 绘制决策边界
	plt.title('参数搜索后的决策边界')
	plot(np.c_[X, y])
	visualize_boundary(X, rbf_svm)
	plt.show(block=True)
	
	# best = dataset2_params_ver2(X, y, X_val, y_val)
	# rbf_svm.set_params(C=best['C'])
	# rbf_svm.set_params(gamma=best['gamma'])

	# plot(np.c_[X, y])
	# visualize_boundary(X, rbf_svm)
	# plt.show(block=True)

	# best = dataset3_params_ver3(X, y, X_val, y_val)
	# rbf_svm.set_params(C=best['C'])
	# rbf_svm.set_params(gamma=best['gamma'])

	# plot(np.c_[X, y])
	# visualize_boundary(X, rbf_svm)
	# plt.show(block=True)


def main():
	np.set_printoptions(precision=6, linewidth=200)
	part1()
	part2()
	part3()
	

if __name__ == '__main__':
	main()

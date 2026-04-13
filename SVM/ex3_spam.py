# -*- coding: utf-8 -*-

import csv
import re
import pickle

import numpy as np
import nltk, nltk.stem.porter
import scipy.misc, scipy.io, scipy.optimize
from sklearn import svm

isTrainMode = False
def vocaburary_mapping()->dict[str,int]:
	'''

	Returns:
		dict[str,int]:单词对应编号
	'''
	vocab_list = {}
	with open('vocab.txt', 'r') as file:
		reader = csv.reader(file, delimiter='\t')
		for row in reader:
			vocab_list[row[1]] = int(row[0])
			
	return vocab_list

def feature_extraction(word_indices)->np.ndarray:
	'''

		Returns:
			np.ndarray:邮件中存在的单词置位1
	'''
	features = np.zeros((1899, 1))
	for index in word_indices:
		features[index] = 1
	return features

def email_preprocess(email):
	with open(email, 'r') as f:
		email_contents = f.read()
	vocab_list = vocaburary_mapping()
	word_indices = []
	email_contents = email_contents.lower()
	email_contents = re.sub('<[^<>]+>', ' ', email_contents)
	email_contents = re.sub('[0-9]+', 'number', email_contents)
	email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
	email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)
	email_contents = re.sub('[$]+', 'dollar', email_contents)
	tokens = re.split('[ ' + re.escape("@$/#.-:&*+=[]?!(){},'\">_<;%\n") + ']+', email_contents)
	tokens = [t for t in tokens if t]

	stemmer = nltk.stem.porter.PorterStemmer()
	for token in tokens:
		if len(token) == 0:
			continue

		token = re.sub('[^a-zA-Z0-9]', '', token)
		token = stemmer.stem(token.strip())

		if token in vocab_list:
			word_indices.append(vocab_list[token])
			
	return word_indices, ' '.join(tokens)

# 预处理
def part_1():
	print("=" *  27+"part1"+"=" * 27)
	word_indices, processed_contents = email_preprocess('emailSample1.txt')
	print(word_indices)
	print(processed_contents)

# 特征提取
def part_2():
	print("=" *  27+"part2"+"=" * 27)
	word_indices, processed_contents = email_preprocess('emailSample1.txt')
	features = feature_extraction(word_indices)#features.shape = (1899,1)

# SVM模型训练
def part_3():
	print("=" * 27 + "part3" + "=" * 27)
	#加载训练集
	mat = scipy.io.loadmat("spamTrain.mat")
	X, y = mat['X'], mat['y']
	# linear_svm = pickle.load(open("linear_svm.svm", "rb")) # 模型加载
	#训练SVM
	if isTrainMode:
		linear_svm = svm.SVC(C=0.1, kernel='linear')
		linear_svm.fit(X, y.ravel())
		pickle.dump(linear_svm, open("linear_svm.svm", "wb")) # 模型保存
	else:
		with open("linear_svm.svm", "rb") as f:
			linear_svm = pickle.load(f)

	#--------------正确率---------------------
	# 预测并计算训练集正确率
	predictions = linear_svm.predict(X)#X.shape=(4000, 1899) predictions.shape=(4000,)
	predictions = predictions.reshape(-1, 1)#predictions.shape=(4000,1)
	print("C=0.1的线性核SVM训练后对于训练集的正确率为{}%".format((predictions == y).mean() * 100.0))
	print("C=0.1的线性核SVM训练后对于训练集的得分为{}".format(linear_svm.score(X, y)))

	# 加载测试集
	mat = scipy.io.loadmat("spamTest.mat")
	X_test, y_test = mat['Xtest'], mat['ytest']
	# 预测并计算测试集正确率
	predictions = linear_svm.predict(X_test)
	predictions = predictions.reshape(np.shape(predictions)[0], 1)
	print('C=0.1的线性核SVM训练后对于测试集的正确率为{}%'.format((predictions == y_test).mean() * 100.0))
	print("C=0.1的线性核SVM训练后对于测试集的得分为{}".format(linear_svm.score(X_test, y_test)))

	# -----------------------------------
	vocab_list = vocaburary_mapping()
	reversed_vocab_list = dict((v, k) for (k, v) in vocab_list.items())
	sorted_indices = np.argsort(linear_svm.coef_, axis=None)

	print("=== 普通邮件特征词 ===")
	for i in sorted_indices[0:15]:
		print('{0}:{1}'.format(reversed_vocab_list[i],linear_svm.coef_[0][i]))
	print("=== 垃圾邮件特征词 ===")
	for i in sorted_indices[-15:]:
		print('{0}:{1}'.format(reversed_vocab_list[i], linear_svm.coef_[0][i]))

	# 验证：统计特征词在 spam/ham 中的出现频率
def SpamVerify():
	mat = scipy.io.loadmat("spamTrain.mat")
	X, y = mat['X'], mat['y']
	spam_mask = (y.ravel() == 1)
	ham_mask = (y.ravel() == 0)
	n_spam = spam_mask.sum()
	n_ham = ham_mask.sum()
	with open("linear_svm.svm", "rb") as f:
		linear_svm = pickle.load(f)
	# -----------------------------------
	vocab_list = vocaburary_mapping()
	reversed_vocab_list = dict((v, k) for (k, v) in vocab_list.items())
	sorted_indices = np.argsort(linear_svm.coef_, axis=None)

	print(f"\n{'词':<12} {'spam出现率':>10} {'ham出现率':>10} {'spam/ham比':>10}")
	print("-" * 45)

	# 取权重最大的5个词和最小的5个词
	check_indices = list(sorted_indices[-5:]) + list(sorted_indices[:5])
	for i in check_indices:
		word = reversed_vocab_list[i]
		weight = linear_svm.coef_[0, i]
		spam_rate = X[spam_mask, i].sum() / n_spam
		ham_rate = X[ham_mask, i].sum() / n_ham
		ratio = spam_rate / ham_rate if ham_rate > 0 else float('inf')
		print(f"{word:<12} {spam_rate:>10.3f} {ham_rate:>10.3f} {ratio:>10.2f}x  (w={weight:.3f})")


def part_4():
	print("=" * 27 + "part4" + "=" * 27)
	with open("linear_svm.svm", "rb") as f:
		linear_svm = pickle.load(f)
	for filename in ['hamTest.txt','spamTest.txt','spamSample1.txt', 'spamSample2.txt']:
		word_indices, _ = email_preprocess(filename)
		features = feature_extraction(word_indices).T  # (1, 1899)
		prediction = linear_svm.predict(features)[0]
		label = '垃圾邮件' if prediction == 1 else '普通邮件'
		print(f'{filename}: {label}')


if __name__ == '__main__':
	part_1()
	part_2()
	part_3()
	SpamVerify()
	part_4()
# print(vocaburary_mapping())

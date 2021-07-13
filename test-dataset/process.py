# -*- coding: utf-8 -*-
# @Author: bao
# @Date:   2021-03-17 14:39:11
# @Last Modified by:   bao
# @Last Modified time: 2021-03-17 17:25:41

import os
import pandas as pd 
import csv 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import numpy as np
plt.style.use('ggplot')
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def counter(draw_chart=True, limit=None):
	"""Count and visualize class distribution in this dataset
	Args: 
		- draw_chart: whether to draw class distribution chart
	Return:
		- files: variable contains all file infomation: contained class(es), file name
		- classes: class name and number of object in each class
	"""

	classes = dict()
	files   = dict()

	if limit is not None:
		c = 0

	csv_reader = csv.reader(open("train_v2.csv"))

	for index, line in tqdm(enumerate(csv_reader), desc="Counting"):

		# Skip the first line
		if index == 0 or len(line) == 0:
			continue

		file_name = line[0]
		_classes  = line[1].strip().split(" ")

		# init container of each file
		files[file_name] = dict({"contain": _classes}) 

		for _cls in _classes:

			# Check whether the counter has this class
			if _cls not in classes.keys(): 
				classes[_cls] = 1
			else:
				classes[_cls] += 1

		if limit is not None:
			c += 1
			if c > limit:
				break 

	if draw_chart:
		# Visualize
		xlabels = list(classes.keys())
		yvalues = list(classes.values())

		x_pos = [i for i, _ in enumerate(xlabels)]

		plt.bar(x_pos, yvalues, color='green')
		plt.xlabel("Classes")
		plt.ylabel("Number")
		plt.title("Class distribution")

		plt.xticks(x_pos, xlabels, rotation=90)

		for idx, y in enumerate(yvalues):
			xy = (idx, y)                                       # <--
			plt.annotate('%s' %y, xy=xy, textcoords='data') # <--

		plt.savefig("classes.png")

	# print(classes.keys())
	return files, classes


def spliter(files, classes, k=5):
	"""Split training set to k-folds with same class ratio
	Args:
		- files: file data variable(get from counter function)
	Return:
		- text files contain file names in each fold
	"""

	class_list = list(classes.keys())

	y = []
	
	# Init array contain instance of each class in an image 
	for file_name in tqdm(files.keys(), total=len(files), desc="Refinding"):
		y.append([(1 if _cls in files[file_name]["contain"] else 0) for _cls in classes.keys()])

	# Let's split
	X = np.array([i for i in range(0, len(files.keys()))])	
	y = np.array(y)

	msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)

	for train_index, test_index in msss.split(X, y):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

	# Save training set
	train_string = "image_name,tags\n"
	valid_string = "image_name,tags\n"

	for index, file_name in enumerate(files.keys()):
		if index in train_index:
			train_string += "test-dataset/train-jpg/%s.jpg,%s\n" %(file_name, " ".join(files[file_name]["contain"]))

		if index in test_index:
			valid_string += "test-dataset/train-jpg/%s.jpg,%s\n" %(file_name, " ".join(files[file_name]["contain"]))

	with open("train.csv", "w+") as f:
		f.write(train_string)

	with open("valid.csv", "w+") as f:
		f.write(valid_string)

if __name__ == '__main__':
	# files, classes = counter(limit=4000)
	files, classes = counter() 
	spliter(files, classes)
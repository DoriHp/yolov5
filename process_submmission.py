# -*- coding: utf-8 -*-
# @Author: bao
# @Date:   2021-03-01 16:13:01
# @Last Modified by:   bao
# @Last Modified time: 2021-03-23 15:29:37


import csv
import os
from tqdm import tqdm 
import pandas as pd
import numpy as np
from glob import glob
import shutil
from ensemble_boxes import *
import cv2
from numpy import random
from utils import *

# Remove all detection that have confidence below a thres
def conf_filter(input_csv, output_csv, conf_thres=0.1, class_conf=None):
	
	csv_file = csv.reader(open(input_csv))
	
	with open(output_csv, "w") as f:
		write_string = "image_id,PredictionString\n"

		for index, line in tqdm(enumerate(csv_file), desc="Completed"):
			if index == 0 or len(line) == 0:
				continue

			filename = line[0]
			append_string = filename + ","
			# Get all detection in this image
			data = line[1].split(" ")
			detections = [data[i:i+6] for i in range(0, len(data) - 1, 6)]
			for detection in detections:
				class_id = detection[0]

				if class_id != "14":
					prob = detection[1]
					if class_conf is None:
						if float(prob) >= conf_thres:
							append_string += "%s " % " ".join(detection)
					else:
						if float(prob) >= class_conf[int(class_id)]:
							append_string += "%s " % " ".join(detection)
				else:
					append_string += "%s " % " ".join(detection)

			if len(append_string) == len(filename) + 1:
				append_string += "14 1 0 0 1 1"

			write_string += append_string + "\n"


		f.write(write_string)


def filter_2cls(row, low_thr=0.08, high_thr=0.95):
	prob = row['target']
	if prob<low_thr:
		## Less chance of having any disease
		row['PredictionString'] = '14 1 0 0 1 1'
	elif low_thr<=prob<high_thr:
		## More change of having any diesease
		row['PredictionString']+=f' 14 {prob} 0 0 1 1'
	elif high_thr<=prob:
		## Good chance of having any disease so believe in object detection model
		row['PredictionString'] = row['PredictionString']
	else:
		raise ValueError('Prediction must be from [0-1]')
	return row

# 2 class filter + 14 class detection
def combinator(rf_classify, rf_detection):
	pred_14cls = pd.read_csv(rf_detection)
	pred_2cls = pd.read_csv(rf_classify)

	print("*" * 30, "\nSample taken from result files:")
	print(pred_14cls.head())
	print(pred_2cls.head())

	print("*" * 30, "\nMerge 2 files together:")
	pred = pd.merge(pred_14cls, pred_2cls, on = 'image_id', how = 'left')
	print(pred.head())

	no_finding_before = pred['PredictionString'].value_counts().iloc[[0]]
	print("Number of 'No finding' images before using 2 class filter: ", no_finding_before)

	sub = pred.apply(filter_2cls, axis=1)
	print("File header after using 2 class filter: ")
	print(sub.head())

	no_finding_after = sub['PredictionString'].value_counts().iloc[[0]]
	print("Number of 'No finding' images after using 2 class filter: ", no_finding_after)

	sub[['image_id', 'PredictionString']].to_csv('submission.csv',index = False)

	print("Filtering completed!")


def fused_box(csv_files=[], policy="normal"):
	"""
		Args:
		- csv_files: list of csv files that contain detections from each model for test set
		Output:
		- csv file contain result from wbf algorthim
	""" 

	assert len(csv_files) > 0, "CSV list cannnot be empty!"
	shapes = dict()

	image_folder = "/home/yolov5/dataset/test"
	all_detections = dict()

	for model_id, file in enumerate(csv_files):
		csv_file = csv.reader(open(file))

		all_detections[model_id] = dict()

		for index, line in tqdm(enumerate(csv_file), desc="Retrieve detections from %s file" %file):
			# Skip for the first line
			if index == 0 or len(line) == 0:
				continue

			file_name = line[0]

			if file_name not in shapes.keys():
				image = cv2.imread(os.path.join(image_folder, file_name + ".jpg"))
				img_height, img_width, _ = image.shape
				shapes[file_name] = (img_width, img_height)
			else:
				img_width, img_height = shapes[file_name]

			# Init dict that contains all detections data of this file
			all_detections[model_id][file_name] = dict({
					"bboxes": [],
					"scores": [],
					"labels": []
				})

			append_string = file_name + ","
			# Get all detection in this image
			data = line[1].split()
			detections = [data[i:i+6] for i in range(0, len(data) - 1, 6)]
			for detection in detections:
				class_id = detection[0]
				# Except class_id 14 ("No finding")
				if class_id != "14":
					all_detections[model_id][file_name]["labels"].append(int(detection[0]))
					all_detections[model_id][file_name]["scores"].append(float(detection[1]))

					# Normalize coor to [0, 1]

					x1, y1, x2, y2 = [float(j) for j in detection[2:]]
					x1 = x1 / img_width
					x2 = x2 / img_width 
					y1 = y1 / img_height
					y2 = y2 / img_height				
					all_detections[model_id][file_name]["bboxes"].append([x1, y1, x2, y2])						

	file_list = all_detections[0].keys()

	# Params for 
	if policy == "normal":
		model_weights = [1 for i in range(len(csv_files))] # Treat all models are equally important
	else:
		model_weights = [1]
	# iou_thr = 0.55 # Fused boxes have this overlaped value
	iou_thr = 0.4 # optimal iou threshold
	skip_box_thr = 0.01 # Skip box with this conf
	sigma = 0.1

	with open("wbf3_submission.csv", "w") as f:
		write_string = "image_id,PredictionString\n"
		for file_name in tqdm(file_list, total=len(file_list), desc="Fusing box"):
			write_string += file_name + ","

			if policy == "normal":
				# Normal fuse boxes
				###########################
				
				bboxes = []
				scores = []
				labels = []
				for model_id in range(len(csv_files)):
					bboxes.append(all_detections[model_id][file_name]["bboxes"])
					scores.append(all_detections[model_id][file_name]["scores"])
					labels.append(all_detections[model_id][file_name]["labels"])

				# if file_name == "a1bffc253b662f0ed464ab6e7ab6950a":
					# print(bboxes, scores, labels)

				# In case of no detected object
				if all(len(sc) == 0 for sc in scores):
					write_string += "14 1 0 0 1 1\n"
				else:
					bboxes, scores, labels = nms(bboxes, scores, labels, weights=model_weights, iou_thr=iou_thr)					
					# bboxes, scores, labels = weighted_boxes_fusion(bboxes, scores, labels, weights=model_weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
				
				###########################		
			else:
				# Fused box with policy: class [0, 3] only have 1 detection, other class keep
				###########################
				norm_boxes = []
				norm_scores = []
				norm_labels = []

				spec_boxes = []
				spec_scores = []
				spec_labels = []

				for model_id in range(len(csv_files)):
					for idx, class_id in enumerate(all_detections[model_id][file_name]["labels"]):
						# With cardiomegaly and aortic enlargement classes, we expect each image have only 1 box
						if int(class_id) in [0, 3]:
							spec_boxes.append(all_detections[model_id][file_name]["bboxes"][idx])
							spec_scores.append(all_detections[model_id][file_name]["scores"][idx])
							spec_labels.append(int(class_id))
						# For other classes
						else:	
							norm_boxes.append(all_detections[model_id][file_name]["bboxes"][idx])
							norm_scores.append(all_detections[model_id][file_name]["scores"][idx])
							norm_labels.append(int(class_id))

				# In case of no detected object
				if len(norm_scores) == 0 and len(spec_scores) == 0:
					write_string += "14 1 0 0 1 1\n"
				else:

					fn_boxes = np.array([], dtype=np.int64).reshape(0, 4)
					fs_boxes = np.array([], dtype=np.int64).reshape(0, 4)
					fn_labels = None
					fs_labels = None

					# If we demand only one of the label per image, we set iou threshold to 0
					if len(norm_boxes) != 0:
						fn_boxes, fn_scores, fn_labels = weighted_boxes_fusion([norm_boxes], [norm_scores], [norm_labels], weights=model_weights, iou_thr=0, skip_box_thr=skip_box_thr)
					if len(spec_boxes) != 0:
						# print("Length of spec_boxes: ", len(spec_boxes))
						fs_boxes, fs_scores, fs_labels = weighted_boxes_fusion([spec_boxes], [spec_scores], [spec_labels], weights=model_weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
						# print("Length of fs_boxes: ", len(fs_boxes))

					bboxes = np.vstack((fn_boxes, fs_boxes))
					if fs_labels is not None and fn_labels is None:
						labels = fs_labels
						scores = fs_scores
					if fn_labels is not None and fs_labels is None:
						labels = fn_labels
						scores = fn_scores
					if fn_labels is not None and fs_labels is not None:
						labels = np.concatenate((fn_labels, fs_labels))
						scores = np.concatenate((fn_scores, fs_scores))

				###########################
				# print(file_name)

		# In case of bboxes coor not valid
		if len(bboxes) == 0:
			write_string += "14 1 0 0 1 1\n"
			continue	

		(img_width, img_height) = shapes[file_name]

		for idx, bbox in enumerate(bboxes):
			if len(bbox) != 0:
				x1, y1, x2, y2 = bbox
				x1 *= img_width
				x2 *= img_width 
				y1 *= img_height
				y2 *= img_height	

				# print(names[idx], ": ", scores[idx])
				# Last bbox 
				write_string += "%d %.5f %d %d %d %d " %(labels[idx], scores[idx], x1, y1, x2, y2)
				if idx != len(bboxes) - 1:
					write_string += " "
				else:
					write_string += "\n"

		f.write(write_string)


# Show image and lablels get from submission files
def show_img(input_csv, mode="control"):

	csv_file = csv.reader(open(input_csv))
	img_dir = "F:/Dev/X-ray/x-ray original/test"
	
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

	if mode == "control":
		selected_file = input("Select detection from image to show: ")

		# Whether user found thir requested file name
		exist = False

		for index, line in tqdm(enumerate(csv_file), desc="Completed"):
			if index == 0 or len(line) == 0:
				continue

			filename = line[0]

			if filename == selected_file:
				exist = True
				im0 = cv2.imread(os.path.join(img_dir, filename + ".jpg"))
				# Get all detection in this image
				data = line[1].split()
				detections = [data[i:i+6] for i in range(0, len(data) - 1, 6)]
				for detection in detections:
					class_id = int(detection[0])
					if class_id != 14:
						prob = 	float(detection[1])
						plot_vals = [int(float(i)) for i in detection[2:6]]
						plot_one_box(plot_vals, im0, prob, colors[class_id], names[class_id])

				display_img = resize_to_fit(im0)

				cv2.imshow("Test", display_img)
				cv2.waitKey(0)  # 1 millisecond
				break 

		cv2.destroyAllWindows()
		if not exist:
			print("Cannot find requested file name")

	if mode == "conseque":
		for index, line in tqdm(enumerate(csv_file), desc="Completed"):
			if index == 0 or len(line) == 0:
				continue

			filename = line[0]

			im0 = cv2.imread(os.path.join(img_dir, filename + ".jpg"))
			# Get all detection in this image
			data = line[1].split()
			detections = [data[i:i+6] for i in range(0, len(data) - 1, 6)]
			for detection in detections:
				class_id = int(detection[0])
				if class_id != 14:
					prob = 	float(detection[1])
					plot_vals = [int(float(i)) for i in detection[2:6]]
					plot_one_box(plot_vals, im0, prob, colors[class_id], names[class_id])

			display_img = resize_to_fit(im0)

			cv2.imshow("Test", display_img)
			cv2.waitKey(0)  # 1 millisecond

		cv2.destroyAllWindows()

	if mode == "random":
		while True:

			rand = random.randint(0, 3000)
			for index, line in tqdm(enumerate(csv_file), desc="Completed"):
				if index == 0 or len(line) == 0 or index != rand:
					continue

				filename = line[0]

				im0 = cv2.imread(os.path.join(img_dir, filename + ".jpg"))
				# Get all detection in this image
				data = line[1].split()
				detections = [data[i:i+6] for i in range(0, len(data) - 1, 6)]
				for detection in detections:
					class_id = int(detection[0])
					if class_id != 14:
						prob = 	float(detection[1])
						plot_vals = [int(float(i)) for i in detection[2:6]]
						plot_one_box(plot_vals, im0, colors[class_id], names[class_id])

				display_img = resize_to_fit(im0)

				cv2.imshow("Test", display_img)
				cv2.waitKey(0)  # 1 millisecond

			cont = input("Continue? (y or n): ")
			if cont == "n":
				break

	cv2.destroyAllWindows()


# Keep only the bbox with highest conf in specified classes
def keep_highest(input_csv, classes=[]):
	csv_file = csv.reader(open(input_csv))
	
	with open("highest.csv", "w") as f:
		write_string = "image_id,PredictionString\n"

		for index, line in tqdm(enumerate(csv_file), desc="Completed"):
			if index == 0 or len(line) == 0:
				continue

			filename = line[0]
			append_string = filename + ","
			# Get all detection in this image
			data = line[1].split()
			detections = [data[i:i+6] for i in range(0, len(data) - 1, 6)]
				
			# Save highest record for each specified class in each file
			highest = dict()
			for _cls in classes:
				highest[_cls] = {
								"prob": 0.0,
								"string": None
							}

			for detection in detections:

				try:
					class_id = int(detection[0])
				except:
					class_id = 14

				if class_id != 14:

					if class_id in classes: 
						prob = float(detection[1])
						if prob > highest[class_id]["prob"]:
							# print(prob, highest[class_id]["prob"])
							highest[class_id]["prob"] = prob 
							highest[class_id]["string"] = "%s " % " ".join(detection)
					else:
						append_string += "%s " % " ".join(detection)

				else:
					append_string += "%s " % " ".join(detection)

			for _cls in classes:
				if highest[_cls]["string"] is not None:
					append_string += highest[_cls]["string"]						

			if len(append_string) == len(filename) + 1:
				append_string += "14 1 0 0 1 1"

			write_string += append_string + "\n"


		f.write(write_string)	


# Discard class that outside bounding box
def discard_out_bound(input_csv, bound_csv="F:/Dev/X-ray/lung-segmentation/bound.csv", keep_classes=[10]):
	
	bound_reader = csv.reader(open(bound_csv))
	# Retrive bound values of all images
	bounds = dict()

	# Retrieving and check boundary
	for index, line in tqdm(enumerate(bound_reader), desc="Retrieving bound values"):
		if index == 0 or len(line) == 0:
			continue

		filename = line[0]
		[xmin, ymin, xmax, ymax] = [int(i) for i in line[1:5]]
		bounds[filename] = (xmin, ymin, xmax, ymax)

		img = cv2.imread(os.path.join("F:/Dev/X-ray/x-ray original/test", filename + ".jpg"))
		cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

		img = resize_to_fit(img)
		cv2.imshow("Boundary", img)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

	csv_file = csv.reader(open(input_csv))

	with open("discard.csv", "w") as f:
		write_string = "image_id,PredictionString\n"

		for index, line in tqdm(enumerate(csv_file), desc="Discarding outside bbox"):
			if index == 0 or len(line) == 0:
				continue

			filename = line[0]
			append_string = filename + ","
			# Get all detection in this image
			data = line[1].split()
			detections = [data[i:i+6] for i in range(0, len(data) - 1, 6)]
				
			for detection in detections:

				try:
					class_id = int(detection[0])
				except:
					class_id = 14

				if class_id != 14:

					if class_id not in keep_classes: 
						[xmin, ymin, xmax, ymax] = [int(i) for i in detection[2:6]]
						# Check box center is within lung bound 
						cen_x = (xmin + xmax) / 2 
						cen_y = (ymin + ymax) / 2
						[b_xmin, b_ymin, b_xmax, b_ymax] = bounds[filename]
						if cen_x > b_xmin and cen_x < b_xmax and cen_y > b_ymin and cen_y < b_ymax:
							append_string += "%s " % " ".join(detection)
					else:
						append_string += "%s " % " ".join(detection)

				else:
					append_string += "%s " % " ".join(detection)

			if len(append_string) == len(filename) + 1:
				append_string += "14 1 0 0 1 1"

			write_string += append_string + "\n"

		f.write(write_string)		

if __name__ == '__main__':
	# fused_box(["fold1.csv", "fold2.csv", "fold3.csv", "fold4.csv", "fold5.csv"])
	combinator("classify_csv_b4.csv", "wbf3_submission.csv")

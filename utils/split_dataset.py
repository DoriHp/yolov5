# -*- coding: utf-8 -*-
# @Author: devBao
# @Date:   2021-02-23 08:27:37
# @Last Modified by:   devBao
# @Last Modified time: 2021-02-23 10:31:08

# This code will read train.txt and valid.txt (used in darknet repo) to organize dataset folder used for training process

import os 
import shutil
from tqdm import tqdm

def split():
	
	files = dict({"train": "/home/training/X-Ray/train_1_2_3_4.txt",
					"valid": "/home/training/X-Ray/valid_0.txt"})

	for k, v in files.items():
		with open(v, "r") as f:
			lines = f.readlines()
			for line in tqdm(lines, total=len(lines), desc="Process %s set" %k):
				if k == "train":
					des_dir = "../dataset/train/"
				if k == "valid":
					des_dir = "../dataset/valid/"

				shutil.copy(line.strip(), des_dir)
				label_path = line.strip().replace("JPEGImages", "labels").replace(".jpg", ".txt")
				shutil.copy(label_path, des_dir)


if __name__ == '__main__':
	split()
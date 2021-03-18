# -*- coding: utf-8 -*-
# @Author: bao
# @Date:   2021-03-08 08:40:46
# @Last Modified by:   bao
# @Last Modified time: 2021-03-17 16:39:49

import argparse
import logging
import os
from pathlib import Path
from typing import List
import csv
import pandas as pd 
from copy import deepcopy

import math
import torch
import torch.nn as nn
from torch import distributed
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision

from torch.cuda import amp
from tqdm import tqdm

from models.common import Classify
from utils.general import set_logging, check_file, increment_path
from utils.torch_utils import model_info, select_device, is_parallel

import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics

# Settings
logger = logging.getLogger(__name__)
set_logging()

# Show images
def imshow(img):

	plt.imshow(np.transpose((img / 2 + 0.5).numpy(), (1, 2, 0)))  # unnormalize
	plt.savefig('images.jpg')

def distributed_is_initialized():
	if distributed.is_available():
		if distributed.is_initialized():
			return True
	return False

class CheXpertDataSet(data.Dataset):
	def __init__(self, image_list_file, transform=None, policy="ones"):
		"""
		image_list_file: path to the file containing images with corresponding labels.
		transform: optional transform to be applied on a sample.
		Upolicy: name the policy with regard to the uncertain labels
		"""
		image_paths = []
		labels = []

		with open(image_list_file, "r") as f:
			csvReader = csv.reader(f)
			next(csvReader, None)
			k=0
			for line in csvReader:
				k+=1
				image_path = line[0]
				label = line[5:]
				
				for i in range(14):
					if label[i]:
						a = float(label[i])
						if a == 1:
							label[i] = 1
						elif a == -1:
							if policy == "ones":
								label[i] = 1
							elif policy == "zeroes":
								label[i] = 0
							else:
								label[i] = 0
						else:
							label[i] = 0
					else:
						label[i] = 0
				
				image_paths.append(image_path)
				labels.append(label)

		self.image_paths = image_paths
		self.labels = labels
		self.transform = transform

	def __getitem__(self, index):
		"""Take the index of item and returns the image and its labels"""
		
		image_path = self.image_paths[index]
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		label = self.labels[index]
		if self.transform is not None:
			image = self.transform(image=image)
			return image['image'], torch.FloatTensor(label)
		return image, torch.FloatTensor(label)

	def __len__(self):
		return len(self.image_paths)


class PlanetDataset(data.Dataset):
	def __init__(self, image_list_file, transform=None):
		"""
		image_list_file: path to the file containing images with corresponding labels.
		transform: optional transform to be applied on a sample.
		"""
		image_paths = []
		labels = []
		names = ['haze', 'primary', 'agriculture', 'clear', 'water', 'habitation', 'road', 'cultivation', 'slash_burn', 'cloudy', 'partly_cloudy', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']

		with open(image_list_file, "r") as f:
			csvReader = csv.reader(f)
			next(csvReader, None)
			k=0
			for line in csvReader:
				k+=1
				image_path = line[0]
				contained_labels = line[1].strip().split(" ")
				label = [0 for i in names]

				for idx, i in enumerate(names):
					if i in contained_labels:
						label[idx] = 1
				
				image_paths.append(image_path)
				labels.append(label)

		self.image_paths = image_paths
		self.labels = labels
		self.transform = transform

	def __getitem__(self, index):
		"""Take the index of item and returns the image and its labels"""
		
		image_path = self.image_paths[index]
		image = cv2.imread(image_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		label = self.labels[index]
		if self.transform is not None:
			image = self.transform(image=image)
			return image['image'], torch.FloatTensor(label)
		return image, torch.FloatTensor(label)

	def __len__(self):
		return len(self.image_paths)		


class DatasetLoader(data.DataLoader):

	def __init__(self,
				 train: bool,
				 batch_size: int,
				 num_workers: int,
				 mean: List[float] = [0.485, 0.456, 0.406],
				 std: List[float] = [0.229, 0.224, 0.225],
				 **kwargs):

		if train:
			transform = A.Compose([
				A.Resize(height=320, width=320),
				A.HorizontalFlip(p=0.5),
				A.CLAHE(),
				A.InvertImg(),
				A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
				A.Normalize(mean=mean, std=std),
				ToTensorV2(),
			])
		else:
			transform = A.Compose([
				A.Resize(height=320, width=320),
				A.Normalize(mean=mean, std=std),
				ToTensorV2(),
			])

		# image_list_file = 'data/CheXpert-v1.0-small/test-train.csv' if train else image_list_file = 'data/CheXpert-v1.0-small/test-valid.csv' 
		image_list_file = "test-dataset/train.csv" if train else "test-dataset/valid.csv"

		# dataset = CheXpertDataSet(image_list_file=image_list_file, transform=transform)
		dataset = PlanetDataset(image_list_file=image_list_file, transform=transform)

		sampler = None
		if train and distributed_is_initialized():
			sampler = data.distributed.DistributedSampler(dataset)

		super(DatasetLoader, self).__init__(dataset,
											batch_size=batch_size,
											shuffle=(sampler is None),
											sampler=sampler,
											num_workers=num_workers,
											**kwargs)


def train():
	bs, epochs, nw = opt.batch_size, opt.epochs, opt.workers

	# Dataloaders
	trainloader = DatasetLoader(train=True, batch_size=bs, num_workers=nw)
	testloader = DatasetLoader(train=False, batch_size=bs, num_workers=nw)
	# names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
	# 		   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
	# 		   'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']

	names = ['haze', 'primary', 'agriculture', 'clear', 'water', 'habitation', 'road', 'cultivation', 'slash_burn', 'cloudy', 'partly_cloudy', 'conventional_mine', 'bare_ground', 'artisinal_mine', 'blooming', 'selective_logging', 'blow_down']

	nc = len(names)
	# print(f'Training {opt.model} on CheXpertDataSet dataset with {nc} classes...')
	print(f'Training {opt.model} on Planet dataset with {nc} classes...')

	# Show images
	# images, labels = iter(trainloader).next()
	# imshow(torchvision.utils.make_grid(images[:16]))
	# print(' '.join('%5s' % names[labels[j]] for j in range(16)))

	# Model
	# YOLOv5 Classifier
	model = torch.hub.load('ultralytics/yolov5', opt.model, pretrained=True, autoshape=False)
	model.model = model.model[:8]
	m = model.model[-1]  # last layer
	ch = m.conv.in_channels if hasattr(m, 'conv') else sum([x.in_channels for x in m.m])  # ch into module
	c = Classify(ch, nc)  # Classify()
	c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
	model.model[-1] = c  # replace

	# x = torch.randn(1, 3, 320, 320).to(device)

	model_info(model)
	model = model.to(device)
	# print(model, model(x))

	# Optimizer
	lr0 = 0.0001 * bs  # intial lr
	lrf = 0.01  # final lr (fraction of lr0)
	if opt.adam:
		optimizer = optim.Adam(model.parameters(), lr=lr0 / 10)
	else:
		optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.9, nesterov=True)

	# Scheduler
	lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
	scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

	# Train
	criterion = nn.BCEWithLogitsLoss(reduction="mean")  # loss function
	# scaler = amp.GradScaler(enabled=cuda)

	# Directories
	wdir = save_dir / 'weights'
	wdir.mkdir(parents=True, exist_ok=True)  # make dir
	last = wdir / 'last.pt'
	best = wdir / 'best.pt'

	writer = SummaryWriter(save_dir)
	print(f"\n{'epoch':10s}{'gpu_mem':10s}{'train_loss':12s}{'val_loss':12s}{'accuracy':12s}")
	max_auc = 0

	for epoch in range(epochs):  # loop over the dataset multiple times
		mloss = 0.  # mean loss
		model.train()
		pbar = tqdm(enumerate(trainloader), total=len(trainloader))  # progress bar
		for i, (images, labels) in pbar:
			# print(type(images), type(labels))
			images, labels = images.to(device), labels.to(device)
			images = F.interpolate(images, scale_factor=4, mode='bilinear', align_corners=False)

			# Forward
			with amp.autocast(enabled=cuda):
				loss = criterion(model(images), labels)

			# Backward
			loss.backward()  # scaler.scale(loss).backward()

			# Optimize
			optimizer.step()  # scaler.step(optimizer); scaler.update()
			optimizer.zero_grad()

			# Print
			mloss += loss.item()
			mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
			pbar.desc = f"{'%s/%s' % (epoch + 1, epochs):10s}{mem:10s}{mloss / (i + 1):<12.3g}"

			# Test
			if i == len(pbar) - 1:
				val_loss, auc_all, auc_mean = test(model, testloader, names, criterion, pbar=pbar)  # test
				writer.add_scalar("eval/loss", val_loss, epoch + 1)
				writer.add_scalar("eval/auc", auc_mean, epoch + 1)
				# print(auc_all)
				for i in range(len(names)):
					writer.add_scalar("eval/auc %s" %names[i], auc_all[i], epoch + 1)

				ckpt = {'epoch': epoch,
						'model': deepcopy(model.module if is_parallel(model) else model).half(),
						'optimizer': optimizer.state_dict()}

				if auc_mean > max_auc:
					torch.save(ckpt['model'], best)

				torch.save(ckpt, last)
				
		writer.add_scalar("train/loss", mloss, epoch + 1)
		# Test
		scheduler.step()
	
	# Show predictions
	# images, labels = iter(testloader).next()
	# predicted = torch.max(model(images), 1)[1]
	# imshow(torchvision.utils.make_grid(images))
	# print('GroundTruth: ', ' '.join('%5s' % names[labels[j]] for j in range(4)))
	# print('Predicted: ', ' '.join('%5s' % names[predicted[j]] for j in range(4)))


def compute_auc(gt, pred, names):
	
	output = []
	
	gt_np = gt.cpu().numpy()
	pred_np = pred.cpu().numpy()
	
	draw_auc(gt, pred, names)
	for i in range(len(names)):
		try:
			output.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
		except ValueError:
			output.append(1)
			pass
	return output

def draw_auc(gt, pred, names):
	"""
		Args:
		- names: list of class name(s)
		- gt: ground truth value
		- pred: predictions from model
		- save_dir: where to save the figure
	"""
	fig = plt.figure(figsize=(30, 10))
	for i in range(len(names)):
		fpr, tpr, _ = metrics.roc_curve(gt.cpu()[:,i], pred.cpu()[:,i])
		roc_auc = metrics.auc(fpr, tpr)
		f = plt.subplot(2, 7, i+1)

		plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
		f.set_aspect('equal')

		plt.title(names[i], fontsize = 14.0)
		plt.legend(loc = 'lower right')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('FPR', fontsize = 14.0)
		plt.xlabel('TPR', fontsize = 14.0)

	fig.tight_layout()
	plt.savefig((save_dir / "roc.png"), dpi=1000, bbox_inches='tight')

def test(model, dataloader, names, criterion=None, verbose=True, pbar=None):

	gt = torch.FloatTensor().to(device)
	pred = torch.FloatTensor().to(device)

	model.eval()
	loss = 0
	with torch.no_grad():
		for images, labels in dataloader:
			images, labels = images.to(device), labels.to(device)
			images = F.interpolate(images, scale_factor=4, mode='bilinear', align_corners=False)
			y = model(images)
				
			# used for compute AUC
			pred = torch.cat((pred, y), 0)

			if criterion:
				loss += criterion(y, labels)
				# print(loss)

			gt = torch.cat((gt, labels), 0).to(device)

	aurocIndividual = compute_auc(gt, pred, names)
	aurocMean = np.array(aurocIndividual).mean()

	# Update tqdm description
	if pbar:
		pbar.desc += f"{loss / len(dataloader):<12.3g}{aurocMean:<12.3g}"

	if verbose:  # all classes
		class_and_auc = list(zip(names, aurocIndividual))
		df = pd.DataFrame(class_and_auc, columns = ['Class', 'AUC']) 
		# Present AUC for each class
		print(df)

	return (loss / len(dataloader)), aurocIndividual, aurocMean

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='yolov5s', help='initial weights path')
	parser.add_argument('--data', type=str, default='xray', help='cifar10, cifar100 or mnist')
	parser.add_argument('--hyp', type=str, default='data/hyp.classifier.yaml', help='hyperparameters path')
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs')
	parser.add_argument('--img-size', nargs='+', type=int, default=[256, 256], help='[train, test] image sizes')
	parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
	parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
	parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
	parser.add_argument('--project', default='runs/train', help='save to project/name')
	parser.add_argument('--name', default='classifier', help='save to project/name')
	parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
	opt = parser.parse_args()

	device = select_device(opt.device, batch_size=opt.batch_size)
	cuda = device.type != 'cpu'
	opt.hyp = check_file(opt.hyp)  # check files
	opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 if 1
	opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
	save_dir = Path(opt.save_dir)

	train()

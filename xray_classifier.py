# -*- coding: utf-8 -*-
# @Author: bao
# @Date:   2021-03-08 08:40:46
# @Last Modified by:   bao
# @Last Modified time: 2021-03-09 14:27:26

import argparse
import logging
import os
from pathlib import Path
from typing import List
import csv

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as T
from torch.utils import data
from torch import distributed

from torch.cuda import amp
from tqdm import tqdm

from models.common import Classify, MLClassify
from utils.general import set_logging, check_file, increment_path
from utils.torch_utils import model_info, select_device

import matplotlib.pyplot as plt
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from sklearn.metrics import roc_auc_score

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
		else:
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

		if train:
			# Paths to the files with training, and validation sets.
			# Each file contains pairs (path to image, output vector)
			image_list_file = 'data/CheXpert-v1.0-small/test-train.csv'
		else:
			image_list_file = 'data/CheXpert-v1.0-small/test-valid.csv'

		dataset = CheXpertDataSet(image_list_file=image_list_file, transform=transform)

		sampler = None
		if train and distributed_is_initialized():
			sampler = data.distributed.DistributedSampler(dataset)

		super(DatasetLoader, self).__init__(dataset,
											batch_size=batch_size,
											shuffle=(sampler is None),
											sampler=sampler,
											num_workers=num_workers,
											**kwargs)

class DenseNet121(nn.Module):
	"""Model modified.
	The architecture of our model is the same as standard DenseNet121
	except the classifier layer which has an additional sigmoid function.
	"""
	def __init__(self, out_size):
		super(DenseNet121, self).__init__()
		self.densenet121 = torchvision.models.densenet121(pretrained=False)
		num_ftrs = self.densenet121.classifier.in_features
		self.densenet121.classifier = nn.Sequential(
			nn.Linear(num_ftrs, out_size),
			nn.Sigmoid()
		)

	def forward(self, x):
		x = self.densenet121(x)
		return x

def train():
	data, bs, epochs, nw = opt.data, opt.batch_size, opt.epochs, opt.workers

	# Dataloaders
	trainloader = DatasetLoader(train=True, batch_size=bs, num_workers=nw)
	testloader = DatasetLoader(train=False, batch_size=bs, num_workers=nw)
	names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
			   'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
			   'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
	nc = len(names)
	print(f'Training {opt.model} on CheXpertDataSet dataset with {nc} classes...')

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
	c = MLClassify(ch, nc)  # Classify()
	c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
	model.model[-1] = c  # replace

	x = torch.randn(1, 3, 320, 320).to(device)

	# print(model)

	model_info(model)
	model = model.to(device)
	print(model, model(x))

	# summary(model, (3, 320, 320))

	# _model = DenseNet121(14).to(device)
	# model_info(_model)
	# _model.eval()

	# print(_model(x))	

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
	criterion = nn.BCEWithLogitsLoss(size_average = True)  # loss function
	# scaler = amp.GradScaler(enabled=cuda)
	print(f"\n{'epoch':10s}{'gpu_mem':10s}{'train_loss':12s}{'val_loss':12s}{'accuracy':12s}")
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
				test(model, testloader, names, criterion, pbar=pbar)  # test

		# Test
		scheduler.step()
	
	# Show predictions
	# images, labels = iter(testloader).next()
	# predicted = torch.max(model(images), 1)[1]
	# imshow(torchvision.utils.make_grid(images))
	# print('GroundTruth: ', ' '.join('%5s' % names[labels[j]] for j in range(4)))
	# print('Predicted: ', ' '.join('%5s' % names[predicted[j]] for j in range(4)))

def computeAUROC (dataGT, dataPRED, classCount):
	
	outAUROC = []
	
	datanpGT = dataGT.cpu().numpy()
	datanpPRED = dataPRED.cpu().numpy()
	
	for i in range(classCount):
		try:
			outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
		except ValueError:
			pass
	return outAUROC

def test(model, dataloader, names, criterion=None, verbose=True, pbar=None, use_gpu=True):

	if use_gpu:
		outGT = torch.FloatTensor().cuda()
		outPRED = torch.FloatTensor().cuda()
	else:
		outGT = torch.FloatTensor()
		outPRED = torch.FloatTensor()

	model.eval()
	pred, targets, loss = [], [], 0
	with torch.no_grad():
		for images, labels in dataloader:
			images, labels = images.to(device), labels.to(device)
			images = F.interpolate(images, scale_factor=4, mode='bilinear', align_corners=False)
			y = model(images)
				
			# used for compute AUC
			outPRED = torch.cat((outPRED, y), 0)
			outGT = torch.cat((outGT, labels), 0).to(device)

			targets.append(labels)
			if criterion:
				loss += criterion(y, labels)
				# print(loss)

	aurocIndividual = computeAUROC(outGT, outPRED, len(names))
	aurocMean = np.array(aurocIndividual).mean()

	if pbar:
		pbar.desc += f"{loss / len(dataloader):<12.3g}{aurocMean:<12.3g}"

	if verbose:  # all classes
		print ('AUROC mean ', aurocMean)
		
		for i in range (0, len(aurocIndividual)):
			print (names[i], ' ', aurocIndividual[i])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='yolov5s', help='initial weights path')
	parser.add_argument('--data', type=str, default='xray', help='cifar10, cifar100 or mnist')
	parser.add_argument('--hyp', type=str, default='data/hyp.classifier.yaml', help='hyperparameters path')
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs')
	parser.add_argument('--img-size', nargs='+', type=int, default=[320, 320], help='[train, test] image sizes')
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

	train()

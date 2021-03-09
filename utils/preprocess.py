# -*- coding: utf-8 -*-
# @Author: bao
# @Date:   2021-02-26 08:36:03
# @Last Modified by:   bao
# @Last Modified time: 2021-02-26 10:07:45

import cv2

class ContrastIncrease(object):

	def __init__(self):
		pass 

	def __call__(self, img):

		#-----Converting image to LAB Color model----------------------------------- 
		lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

		#-----Splitting the LAB image to different channels-------------------------
		l, a, b = cv2.split(lab)

		#-----Applying CLAHE to L-channel-------------------------------------------
		clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
		cl = clahe.apply(l)

		#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
		limg = cv2.merge((cl,a,b))

		#-----Converting image from LAB Color model to RGB model--------------------
		final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

		return final

if __name__ == '__main__':

	img = cv2.imread("F:/Dev/yolov5/runs/history/training1/train/train_batch1.jpg")	
	pre = ContrastIncrease()
	_img = pre(img)

	cv2.imshow("Original", img)
	cv2.imshow("Test", _img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
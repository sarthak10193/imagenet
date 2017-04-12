import os
import re
import random 
import numpy as np
import cv2

path = "/home/sarthak/PycharmProjects/imagenet/imagenet/dataFiles/"

allimageslist = []

folders = os.listdir(path)
for folder in folders:
	if(re.search(r'images-', folder)):
		images = os.listdir(path + folder)
		for image in images:

			mypath = path+folder +"_" + image+ " " + folder[-1:]+"\n"
			print(mypath)
			allimageslist.append(mypath)


# split the training data into Train and Cross Validation
random.shuffle(allimageslist)


train = allimageslist[:int(len(allimageslist)*.80)]
valid = allimageslist[int(len(allimageslist)*.80):]

print("Creating the Training and validation sets @ /home/sarthak/PycharmProjects/imagenet/imagenet/trainData/")

with open("/home/sarthak/PycharmProjects/imagenet/imagenet/trainData/train.txt", 'a') as f:
	for imagepath in train:
		try:
			print(imagepath[:-3].replace("_", "/"))
			img = cv2.imread(imagepath[:-3].replace("_", "/"))
			img = cv2.resize(img, (227, 227))

			f.write(imagepath)
		except Exception as e:
			print("error")


with open("/home/sarthak/PycharmProjects/imagenet/imagenet/trainData/valid.txt", 'a') as f:
	for imagepath in valid:
		try:
			img = cv2.imread(imagepath[:-3].replace("_", "/"))
			img = cv2.resize(img, (227, 227))
			f.write(imagepath)
		except Exception as e:
			print("error")




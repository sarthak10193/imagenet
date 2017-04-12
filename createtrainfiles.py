import os
import re
import random 
import cv2
IMAGE_SIZE = 227

"""
1. get the <image path,classID> tuple for each image
2. shuffle the tuples
3. divide the list of above tuples into training set and validation set
4. filter out valid images
5. create files train.txt and valid.txt having this <image path, classID> tuple

"""

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
			img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

			f.write(imagepath)
		except Exception as e:
			print("error")


with open("/home/sarthak/PycharmProjects/imagenet/imagenet/trainData/valid.txt", 'a') as f:
	for imagepath in valid:
		try:
			img = cv2.imread(imagepath[:-3].replace("_", "/"))
			img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
			f.write(imagepath)
		except Exception as e:
			print("error")




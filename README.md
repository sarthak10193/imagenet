# imagenet
imagenet classification task using vgg16 and alexnet in Tensoflow 

#Files and Folders description

vggModel.py    		# defines the structure of the vgg16 model in tensorflow
alexnet.py     		# defines the structure of the alexnet model in tensorflow
caffe_classes.py 	# contains the 1000 classes structure related to the imagenet classification task
train.py   		# run train.py to perform imagenet classication task using either the vgg16 or alexnet. This uses transfer learning
datagenerator		# helper for generating batches and rescaling images using opencv
download_data		# downloading the imagenet dataset from the imagenet website
createtrainfiles.py	# creating a file for referncing the paths of .jpeg data images on the system

#preTrainedWeights 
1.bvlc_alexnet.npy	[download link: http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy]
2.vgg16_weights.npz    [download link: https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz]

#dataFiles
images-class0 
images-class1 
images-class2

class0.txt
class1.txt
class2.txt

#trainData
1.Train.txt
2.valid.txt

#Run
to Run just run train.py and also make the appropriate changes for the model being used and which layers to train using trasnfer learning










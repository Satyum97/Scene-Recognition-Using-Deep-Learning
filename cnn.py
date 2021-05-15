'''
Authors: 
1. Tejas Gupta (TXG180021)
2. Satyam Bhikadiya (SXB180124)
'''

# Machine Learning Libraries
import numpy as np
from skimage.transform import rescale, resize
import matplotlib.image as mpimg
from skimage import io

# Libraries to Read Image Dataset
import glob 
import os

# Convolution Neural Net Library
from architecture.CNN_Network import Convolution_Neural_Network



'''
# Reads the images from multiple sub directories of training or testing dataset and combines into
  large numpy array, normalizes and converts the image into grayscale
# params    : @catalog: Location of training or testing image dataset
# returns   : training or testing image dataset

'''
def load_dataset(catalog):
  global dataset
  data = []
  for img in glob.glob( catalog + '/*' + '/*.jpg'):
      data.append(resize(mpimg.imread(img)/255, (100, 100, 3))) # Converts the Image to grayscale 
      dataset = np.asarray(data)
  dataset -= int(np.mean(dataset))  # Normalizes the image dataset
  return dataset





'''
# Generating the class labels for the training or testing image dataset
# params  : @unitClasses    : Number of classes to be labelled
            @categories     : List of category
            @enumerate_Imgs : Count of Images
            @catalog        : Location of training or testing image dataset

# returns : Class labels for the training or testing image dataset

'''
def generate_labels(unitClasses,category,enumerate_Imgs,catalog):
  collections = glob.glob(catalog + '/*')
  for _,collection in enumerate(collections):
    collection_length = len(glob.glob( collection + '/*.jpg'))
    category+=[enumerate_Imgs]*collection_length
    enumerate_Imgs+=1
  return np.eye(unitClasses)[np.asarray(category)]



# Hyper Parameters
numOfTestImages= 1000
sample_group = 1
epochs = 3


# Loading Images and Generating Class Labels for Training Image Dataset
print('-----------Loading Images and Generating Class Labels for Training Image Dataset----------')
train_dataset = load_dataset(r"training_datasets")
generated_trainLabels = generate_labels(6,[],0,r"training_datasets")

# Loading Images and Generating Classes for Testing Image Dataset
print('-----------Loading Images and Generating Class Labels for Training Image Dataset----------')
test_dataset = load_dataset(r"testing_datasets")
generated_testLabels = generate_labels(6,[],0,r"testing_datasets")


model = Convolution_Neural_Network()



#  Training the CNN Model
print('---------------------Training Convolution_Neural_Network----------------------------------')
model.train(train_dataset, generated_trainLabels, sample_group, epochs, 'CNN_weights.pkl')


print('---------------------Testing Convolution_Neural_Network-----------------------------------')
#  Testing the CNN Model
model.test(test_dataset, generated_testLabels,numOfTestImagess)



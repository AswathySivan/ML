import model
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import load_model
import os
import cv2

input_shape=[224,224,3]

AlexNet_model = model.model()
AlexNet_model = load_model('AlexNet_model.h5')

labels_list = os.path.join("Dataset","labels.txt")
f = open(labels_list,"r") 
f = f.readlines()
labels_names = []
for line in f:
	labels_names.append(line.rstrip())


Test_folder = os.listdir(os.path.join("Dataset","Test_data"))

for files in Test_folder:
#	print(files)
	img = cv2.imread(os.path.join("Dataset","Test_data",files))
	img = np.array(img)
	img = np.resize(img, (input_shape[0], input_shape[1], input_shape[2]))
	img = img.reshape( 1,input_shape[0], input_shape[1], input_shape[2])
	img = img.astype("float32") / 255.0	
	Y_pred = AlexNet_model.predict(img)
	Y_pred_classes = np.argmax(Y_pred, axis = 1) 
	print(labels_names[Y_pred_classes[0]])


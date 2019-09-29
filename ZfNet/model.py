from keras.models import Sequential
from keras.layers import Conv2D,AveragePooling2D,Flatten,Dense,MaxPooling2D,BatchNormalization,Input
from keras.optimizers import *
import keras.backend as K


opt = {'sgd': SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
      'rmsprop': RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0),
      'adagrad': Adagrad(lr=0.01, epsilon=1e-08, decay=0.0),
      'adadelta': Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),
      'adam': Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)}

def model():

	
	model =Sequential()
	model.add(Conv2D(96, (7,7), strides=(2, 2),padding='valid',activation='relu',input_shape=(224,224,3)))
	model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2)))
	model.add(BatchNormalization(axis=3))

	model.add(Conv2D(256, (5,5), strides=(2, 2),padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2)))
	model.add(BatchNormalization(axis=3))

	model.add(Conv2D(384, (3,3), strides=(1, 1),padding='same',activation='relu'))
	model.add(Conv2D(384, (3,3), strides=(1, 1),padding='same',activation='relu'))
	model.add(Conv2D(256, (3,3), strides=(1, 1),padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2)))

	model.add(Flatten())
	model.add(Dense(4096,activation='relu'))
	model.add(Dense(4096,activation='relu'))
	# model.add(Dense(1000,activation='softmax'))
	model.add(Dense(10,activation='softmax'))	
	model.compile(loss="categorical_crossentropy", optimizer=opt['sgd'],metrics=["accuracy"])

	return model


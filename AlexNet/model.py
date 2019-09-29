from keras.models import Sequential
from keras.layers import Conv2D,AveragePooling2D,Flatten,Dense,MaxPooling2D,BatchNormalization
from keras.optimizers import SGD,Adam

opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


def model():

	model =Sequential()
	model.add(Conv2D(96, (11,11), strides=(4, 4),padding='valid',activation='relu',input_shape=(224,224,3)))
	model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2)))
	model.add(BatchNormalization(axis=3))

	model.add(Conv2D(256, (5,5), strides=(1, 1),padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2)))
	model.add(BatchNormalization(axis=3))

	model.add(Conv2D(384, (3,3), strides=(1, 1),padding='same',activation='relu'))
	model.add(Conv2D(384, (3,3), strides=(1, 1),padding='same',activation='relu'))
	model.add(Conv2D(256, (3,3), strides=(1, 1),padding='same',activation='relu'))
	model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2)))	
	model.add(Flatten())
	model.add(Dense(4096,activation='relu'))
	model.add(Dense(4096,activation='relu'))
	model.add(Dense(10,activation='softmax'))

	model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

	return model

from keras.models import Sequential
from keras.layers import Conv2D,AveragePooling2D,Flatten,Dense,MaxPooling2D
from keras.optimizers import SGD,Adam

# opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

def lenet_model():

	model =Sequential()
	model.add(Conv2D(6, (5,5), strides=(1, 1),padding='valid',activation='relu',input_shape=(32,32,1)))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
	model.add(Conv2D(16, (5,5), strides=(1, 1),padding='valid',activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2, 2)))
	model.add(Conv2D(120, (5,5), strides=(1, 1),padding='valid',activation='relu'))
	model.add(Flatten())
	model.add(Dense(84,activation='relu'))
	model.add(Dense(10,activation='softmax'))
	print(model.summary())
	model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

	return model

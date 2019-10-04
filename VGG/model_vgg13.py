from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,BatchNormalization

# opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

no_class=10

def model():
	model=Sequential()
	model.add(Conv2D(64,(3,3),strides=(1,1),padding="valid",activation="relu",input_shape=(224,224,3)))
	model.add(Conv2D(64,(3,3),strides=(1,1),padding="valid",activation="relu"))	
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(128,(3,3),strides=(1,1),padding="valid",activation="relu"))
	model.add(Conv2D(128,(3,3),strides=(1,1),padding="valid",activation="relu"))	
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(256,(3,3),strides=(1,1),padding="valid",activation="relu"))
	model.add(Conv2D(256,(3,3),strides=(1,1),padding="valid",activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(512,(3,3),strides=(1,1),padding="valid",activation="relu"))
	model.add(Conv2D(512,(3,3),strides=(1,1),padding="valid",activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(512,(3,3),strides=(1,1),padding="valid",activation="relu"))
	model.add(Conv2D(512,(3,3),strides=(1,1),padding="valid",activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	model.add(Flatten())
	model.add(Dense(4096,activation="relu"))
	model.add(Dense(4096,activation="relu"))
	# model.add(Dense(1000,activation="softmax"))
	model.add(Dense(no_class,activation="softmax"))
	model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
	return model		
	#return model
	#print(model.summary())

#model()
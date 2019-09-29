import model
import numpy as np
import keras
import matplotlib.pyplot as plt


images=np.load("Train_data.npy")
labels=np.load("Train_label.npy")

images = images.astype("float32") / 255.0

ZfNet_model= model.model()
history = ZfNet_model.fit(images, labels, batch_size=10, epochs=20)
ZfNet.save('ZfNet_model.h5')

fig = plt.figure()
plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('model_accuracy.png')

plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig('model_loss.png')



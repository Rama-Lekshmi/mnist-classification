# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset
Digit classification and to verify the response for scanned handwritten images.
We aim to develop a convolutional neural network model capable of accurately classifying handwritten digits into their respective numerical values using the MNIST dataset. This dataset comprises 60,000 images of handwritten digits, each sized 28x28 pixels. The task involves categorizing these digits into one of 10 classes, representing integers from 0 to 9 inclusively.

## Neural Network Model
![dl n3](https://github.com/Rama-Lekshmi/mnist-classification/assets/118541549/dca3a860-e0c0-4739-9425-922e30d5bad0)


## DESIGN STEPS

STEP 1:
Import tensorflow and preprocessing libraries.

STEP 2:
Download and load the dataset

STEP 3:
Scale the dataset between it's min and max values

STEP 4:
Using one hot encode, encode the categorical values

STEP 5:
Split the data into train and test

STEP 6:
Build the convolutional neural network model

STEP 7:
Train the model with the training data

STEP 8:
Plot the performance plot

STEP 9:
Evaluate the model with the testing data

STEP 10:
Fit the model and predict the single input


## PROGRAM

### Name: RAMA E.K. LEKSHMI
### Register Number: 212222240082
```
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[0]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,batch_size=64,validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('img3.png')

img = image.load_img('img3.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),axis=1)

print(x_single_prediction)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![dl 3a](https://github.com/Rama-Lekshmi/mnist-classification/assets/118541549/dafdf8cb-ef81-477e-8c30-9b7e6021eb67)
![dl 3b](https://github.com/Rama-Lekshmi/mnist-classification/assets/118541549/428d86aa-1e7e-4cab-a68a-360493f4a006)
![dl 3c](https://github.com/Rama-Lekshmi/mnist-classification/assets/118541549/9aa5a713-5d6b-4729-b9ce-46d97b7a8bcd)

### Classification Report

![dl 3d](https://github.com/Rama-Lekshmi/mnist-classification/assets/118541549/6cc85677-d032-4e53-9d6f-51fe83d077f3)

### Confusion Matrix

![dl 3e](https://github.com/Rama-Lekshmi/mnist-classification/assets/118541549/278d2ecd-26f2-4c63-8c0a-5f127c68851c)

### New Sample Data Prediction
![dl 3f](https://github.com/Rama-Lekshmi/mnist-classification/assets/118541549/e4da7d85-853f-44a6-a507-c23aa25e78c1)
![dl 3g](https://github.com/Rama-Lekshmi/mnist-classification/assets/118541549/124a26cc-43e8-474d-8a79-37e4b186f89e)



## RESULT
Thus a convolutional deep neural network for digit classification is developed and the response for scanned handwritten images is verified.

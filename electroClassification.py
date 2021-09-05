# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 08:36:25 2021

@author: LANSANA BALDE
"""

import numpy as np
import matplotlib.pyplot as plt
import os 
import random
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from skimage.transform import resize
from sklearn.metrics import confusion_matrix


# the dataset path

DATASET_PATH= "dataset"

target_width=28
target_height=28
INVERT = False
# Set aside 20% for validation and 20% for test
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# You are welcome to change the seed to try a different validation set split
random.seed(2)

labels=[]
X_all=[]
y_all=[]

for label in os.listdir(DATASET_PATH):
    class_path=os.path.join(DATASET_PATH,label)
    labels.append(label)
    
    for i, file in enumerate(os.listdir(class_path)):
        path_file=os.path.join(class_path, file)
        img=PIL.Image.open(path_file).convert('L')
        img_array=np.asarray(img)
        if INVERT:
            img_array=255-img_array
        X_all.append(img_array)
        y_all.append(label)
    print("added", str(i+1), "images from", label)

num_sample=len(X_all)
labels=sorted(labels)

### Convert labels to numbers

# Show the labels before the conversion
print("Before:", y_all)

y_out=[]
for i, label in enumerate(y_all):
    y_out.append(labels.index(label))
    y_all=y_out
print("After:", y_all)



#let's shuffle the sample data

x=list(zip(X_all, y_all))
random.shuffle(x)
X_all, y_all=zip(*x)

num_test_sample=int(TEST_RATIO*num_sample)
num_val_sample=int(VAL_RATIO * num_sample)

x_test=X_all[:num_test_sample]
y_test=y_all[:num_test_sample]

X_val=X_all[num_test_sample:(num_test_sample+num_val_sample)]
y_vall=y_all[num_test_sample:(num_test_sample+num_val_sample)]

X_train=X_all[(num_test_sample+num_val_sample):]
y_train=y_all[(num_test_sample+num_val_sample):]


num_sample_train=len(X_train)

idx=32
y_train[idx]

print("Label ", str(y_train[idx]), labels[y_train[idx]])
plt.imshow(X_train[idx], cmap='gray')

def resize_image(images, height, width, anti_aliasing=True):
    X_out=[]
    for i, image in enumerate(images):
        X_out.append(resize(image, (height, width), anti_aliasing=anti_aliasing))
    return X_out


X_train=resize_image(X_train, target_height, target_width)    
X_val=resize_image(X_val, target_height, target_width)
x_test=resize_image(x_test, target_height, target_width)


#convert sample into array
type(X_train) # it shows this a list

X_train=np.asarray(X_train)
y_train=np.asarray(y_all)

#validation data into array
X_val=np.asarray(X_val)
y_vall=np.asarray(y_vall)

#test data into array

x_test=np.asarray(x_test)
y_test=np.asarray(y_test)

#let's flatten our image
len_vector=target_height*target_width

X_train=X_train.reshape(num_sample_train, len_vector)
X_val=X_val.reshape(num_val_sample, len_vector)
x_test=x_test.reshape(num_test_sample, len_vector)

input_shape=(X_train.shape[1],)


num_class = len(labels)

# Use Keras's np_utils to create one-hot encoding (note the capital 'Y' - 2D array)
Y_train=np_utils.to_categorical(y_train, num_class)
Y_val= np_utils.to_categorical(y_vall, num_class)
Y_test=np_utils.to_categorical(y_test, num_class)

model = Sequential()
model.add(Dense(64, input_shape=input_shape))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_class))
model.add(Activation('softmax'))
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()


history=model.fit(X_train, 
                  Y_train, 
                  batch_size=32, 
                  epochs=200, 
                  verbose=1, 
                  validation_data=(X_val,Y_val))

acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure()
plt.plot(epochs, acc, color='blue', marker='.', label="accuarcy")
plt.plot(epochs, val_acc, color='orange', marker='.', label="val_accuracy")
plt.title("The accuracy of the model")
plt.legend()
plt.figure()
plt.plot(epochs, loss, color='blue', marker='.',label="Loss")
plt.plot(epochs, val_loss, color='orange', marker='.', label="Val_Loss")
plt.legend()
plt.title("Loss")


### Try predicting label with one validation sample (inference)

# Change this to try a different sample from the test set

idx=25

x=np.expand_dims(X_val[idx], 0)
y_pred = model.predict(x)
predicted_label = np.argmax(y_pred)
actual_label = np.argmax(Y_val[idx])

# Display model output, predicted label, actual label
print("Model output:", y_pred)
print("Predicted label:", predicted_label, "-", labels[predicted_label])
print("Actual label:", actual_label, "-", labels[actual_label])



score = model.evaluate(X_test, Y_test)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


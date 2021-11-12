from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = 1e-4 #Learning Rate(The Smaller, The BETTER)
EPOCHS = 20 #No. of times we will be sending data to the model
BS = 32 #Batch Size, after many trials

DIR = r"D:\Bennett\Research\Face Mask Detect My File\dataset" # Calling file location in read mode
CAT = ["with_mask", "without_mask"] #2 categories (same as the folder name amde in the data set)

# Now We need to load the images from our dataset, in form of Data
print("[INFO] Loading Images...")

data = []  #declaring empty lists to append data in form of an array
labels = [] #declaring empty lists to append data in form of an array

for category in CAT:
    path = os.path.join(DIR, category) #giving path to the image folders inside directory
    for img in os.listdir(path): #listdir for all the images
        img_path = os.path.join(path, img) #giving path to the indivisual images
        image = load_img(img_path, target_size=(224, 224)) #loading idivisual images in size 224x224 only
        image = img_to_array(image) #converting image data to arrow, since deep learing models works only with arrays
        image = preprocess_input(image) #preprocessing of image using mobileVnet2 
        
        data.append(image) #appending dta into the earlier made list
        labels.append(category) #appending dta into the earlier made list
        
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32") #converting data list into numpy array
labels = np.array(labels) #converting labels list into numpy array

(trainX, testX, trainY, testY) = train_test_split(data, 
                                                  labels, 
                                                  test_size = 0.20, 
                                                  stratify = labels, 
                                                  random_state=42)
#preprocessing ends here
#we will be using mobilenetes instead of convolution nueral networks as they are faster
#mobilenet are sometimes less accurate
#image data generator many images from single images thus increasing size of our data set!

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# here we start modelling part
basemodel = MobileNetV2(weights="imagenet", include_top = False,
                        input_tensor = Input(shape=(224,224,3)))

headmodel = basemodel.output
headmodel = AveragePooling2D(pool_size=(7, 7))(headmodel) 
headmodel = Flatten(name = "flatten")(headmodel)
headmodel = Dense(128, activation="relu")(headmodel) #relu is the go to activation function for non linear user cases
#we should go with relu whenver dealing with images as dataset
headmodel = Dropout(0.5)(headmodel) #avoiding overinput to model
headmodel = Dense(2, activation="softmax")(headmodel) #need to know more about activation functions

model = Model(inputs=basemodel.input, outputs = headmodel) #calleing model function, 

#we need to free the layers in the basemodels so that they are not updated in the first training
for layer in basemodel.layers:
    layer.trainable = False

#compiling hre model now !

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay = INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuraccy"])

print("[INFO] training head...")
History = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)


#Now evauating our model using predict method

print("[INFO] evaluating Network...")
preIdxs = model.predict(testX, batch_size=BS)
     
preIdxs = np.argmax(preIdxs, axis=1)

#printing classification report with some formatting !
print(classification_report(testY.argmax(axis=1), preIdxs, 
                            target_name = lb.classes_))

#saving model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

#ploting graph
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arrange(0, N), History.history["loss"], Label = "train_loss")
plt.plot(np.arrange(0, N), History.history["val_loss"], Label = "val_loss")
plt.plot(np.arrange(0, N), History.history["accuracy"], Label = "train_acc")
plt.plot(np.arrange(0, N), History.history["val_accuracy"], Label = "val_acc")
plt.tittle("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylable("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savfig("plot.png")
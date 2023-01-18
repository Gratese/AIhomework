import os                       # for working with files
import pandas as pd             # for working with dataframes
import matplotlib.pyplot as plt # for plotting informations on graph and images using tensors
import torchvision.transforms as transforms   # for transforming images into tensors 
from torchvision.datasets import ImageFolder  # for working with classes and images
from  tensorflow import keras
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
from mlxtend.plotting import plot_confusion_matrix
#匯入資料集
lung_dir = "./lung_colon_image_set/lung_image_sets"
lungs = os.listdir(lung_dir)
nums_train = {}
nums_val = {}

for lung in lungs:
    nums_train[lung] = len(os.listdir(lung_dir + '/' + lung))
img_per_class_train = pd.DataFrame(nums_train.values(), index=nums_train.keys(), columns=["no. of images"])

#資料及分為訓練資料集(12000張)和測試資料集(3000張)
train = ImageFolder(lung_dir, transform=transforms.ToTensor())
train_preprocess=keras.preprocessing.image.ImageDataGenerator(rescale=1./255,rotation_range=20,horizontal_flip=True,validation_split=0.2)
valid_preprocess=keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split = 0.2)
train_data=train_preprocess.flow_from_directory(lung_dir, subset='training', target_size=(224,224),batch_size=64, color_mode='rgb',class_mode='categorical',shuffle=True)
val_data=valid_preprocess.flow_from_directory(lung_dir,subset='validation',target_size=(224,224),batch_size=64,color_mode='rgb',class_mode='categorical',shuffle=False)

#建立模型
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)))

model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(64, 3, activation='relu'))
model.add(keras.layers.Dropout(0.15))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Conv2D(128, 3, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.MaxPooling2D())

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_data,
        validation_data=val_data,
        epochs = 10)

#損失函數表，x座標為訓練次數，y座標為損失函數
plt.figure(figsize = (20,5))
plt.subplot(1,2,1)
plt.title("Train and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(history.history['loss'],label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.xlim(0, 10)
plt.ylim(0.0,1.0)
plt.legend()

#準確值表，x座標為訓練次數，y座標為準確值
plt.subplot(1,2,2)
plt.title("Train and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.xlim(0, 9.25)
plt.ylim(0.75,1.0)
plt.legend()
plt.tight_layout()



Y_pred = model.predict(val_data)
y_pred = np.argmax(Y_pred, axis=1)

print(classification_report(val_data.classes, y_pred))


#混淆矩陣
cm1 = confusion_matrix(val_data.classes, y_pred)
plot_confusion_matrix(conf_mat=cm1,show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()

#將訓練完的模型存入
model.save("finalModel")
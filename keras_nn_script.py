import numpy as np
import keras
from keras.datasets import cifar10
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.preprocessing import image
from keras.utils import to_categorical
from pathlib import Path

cifar10_class_names = {
    0: 'Plane',
    1: 'Car',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Boat',
    9: 'Truck'
}

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

"""
import matplotlib.pyplot as plt

for i in range(10):
    sample_image = x_test[i]
    image_class_number = y_train[i][0]
    image_class_name = cifar10_class_names[image_class_number]
    plt.imshow(sample_image)
    plt.title(image_class_name)
    plt.show()
"""

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()

model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(x_test, y_test),
    shuffle=True
)

model_structure = model.to_json()
f = Path('model_structure.json')
f.write_text(model_structure)
model.save_weights('model_weights.h5')

storageAccountName = "techcommunity"
storageAccountAccessKey = "3fmvYhF+1wdZIlYlZYTcyR4Hosib+7kKEl+axFF/NywuUqLJd0dAQiC1hAZpfKCVkWGpJoSs+IT8+AStz42mbg=="
blobContainerName = "images"
mountPoint = "/mnt/data/"
if not any(mount.mountPoint == mountPoint for mount in dbutils.fs.mounts()):
  try:
    dbutils.fs.mount(
      source = "wasbs://{}@{}.blob.core.windows.net".format(blobContainerName, storageAccountName),
      mount_point = mountPoint,
      extra_configs = {'fs.azure.account.key.' + storageAccountName + '.blob.core.windows.net': storageAccountAccessKey}
    )
    print("mount succeeded!")
  except Exception as e:
    print("mount exception", e)
    
dbutils.fs.ls("/mnt/data/")

img = image.load_img("/dbfs/mnt/data/truck.png", target_size=(32,32))
image_to_test = image.img_to_array(img) / 255
list_of_images = np.expand_dims(image_to_test, axis=0)
results = model.predict(list_of_images)
single_result = results[0]
most_likely_class_index = int(np.argmax(single_result))
class_likehood = single_result[most_likely_class_index]
class_label = cifar10_class_names[most_likely_class_index]

print("This is image is a {} - Likelihood: {:2f}".format(class_label, class_likehood))
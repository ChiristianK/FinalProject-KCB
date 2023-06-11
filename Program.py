#-------------------------------------------------------------------------------------------------------------------------------------->
#import library
from google.colab import drive
drive.mount('/content/drive/')
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from google.colab import files
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import zipfile, os
#-------------------------------------------------------------------------------------------------------------------------------------->

#-------------------------------------------------------------------------------------------------------------------------------------->
#unzip file dari Drive google dan membaca file
zip_ref = zipfile.ZipFile('/content/drive/My Drive/archive.zip', 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

base_dir = '/tmp/rps-cv-images'
os.listdir(base_dir)
#-------------------------------------------------------------------------------------------------------------------------------------->

#-------------------------------------------------------------------------------------------------------------------------------------->
#melakukan augmentasi pada gambar dan membuat generator data
datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=25,
    horizontal_flip=True,
    vertical_flip=True,
    shear_range=0.2,
    validation_split=0.4
)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150,150),
    batch_size=5,
    seed=42,
    subset="training"
)

validation_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(150,150),
    batch_size=5,
    seed=42,
    shuffle=False,
    subset="validation"
)

#-------------------------------------------------------------------------------------------------------------------------------------->
#membuat model 
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(nesterov=True),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
#-------------------------------------------------------------------------------------------------------------------------------------->

#-------------------------------------------------------------------------------------------------------------------------------------->
# Training model dan pengetesan model
model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=75,
    validation_data=validation_generator,
    validation_steps=10,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=10,
            restore_best_weights=True,
        )
    ]
)
# Testing model
model.evaluate(validation_generator)
#-------------------------------------------------------------------------------------------------------------------------------------->

#-------------------------------------------------------------------------------------------------------------------------------------->
#prediksi gambar
uploaded = files.upload()

for name in uploaded.keys():
  img = image.load_img(name, target_size=(150,150))
  image_plot = plt.imshow(img)
  image_arr = image.img_to_array(img)
  image_arr = np.expand_dims(image_arr, axis=0)

  images = np.vstack([image_arr]) 
  pred = model.predict(images, batch_size = 10)

  print(name)
  if pred[0][0]==1:
    print("paper")
  elif pred[0][1]==1:
    print("rock")
  else:
    print("scissors")
#-------------------------------------------------------------------------------------------------------------------------------------->

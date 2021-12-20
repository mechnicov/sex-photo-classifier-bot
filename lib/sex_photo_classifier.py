import os

from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing import image

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import kaggle

import tempfile
import requests

import numpy as np

class SexPhotoClassifier():
    MODEL_PATH = f'{os.getcwd()}/data/sex_photo_classifier.h5'
    DATASET_NAME = 'mechnicov/sex-photo-classifier'

    def __init__(self):
        if os.path.isfile(self.MODEL_PATH):
            self.model = load_model(self.MODEL_PATH)
        elif 'Dataset' in next(os.walk(f'{os.getcwd()}/data'))[1]:
            self.model = self.__define_model()
            self.__train()
        else:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(self.DATASET_NAME, path = f'{os.getcwd()}/data', unzip = True)
            self.model = self.__define_model()
            self.__train()

    def classify(self, image_path):
        tfile = tempfile.NamedTemporaryFile()
        tfile.write(requests.get(image_path).content)
        tfile.flush()

        img = image.load_img(tfile.name, target_size = (64, 64))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        images = np.vstack([x])
        classes = self.model.predict(images, batch_size = 1)

        verdict = classes[0][0]

        if verdict > 0.5:
            return 'man'
        else:
            return 'woman'

    def __define_model(self):
        return Sequential([
            # 1st conv
            layers.Conv2D(96, (11, 11),strides = (4, 4), activation = 'relu', input_shape = (64, 64, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, strides = (2, 2)),

            # 2nd conv
            layers.Conv2D(256, (11, 11),strides = (1,1), activation = 'relu', padding = 'same'),
            layers.BatchNormalization(),

            # 3rd conv
            layers.Conv2D(384, (3, 3),strides = (1, 1), activation = 'relu', padding = 'same'),
            layers.BatchNormalization(),

            # 4th conv
            layers.Conv2D(384, (3, 3),strides = (1, 1), activation = 'relu', padding = 'same'),
            layers.BatchNormalization(),

            # 5th Conv
            layers.Conv2D(256, (3, 3), strides = (1, 1), activation='relu', padding = 'same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, strides = (2, 2)),

            # To Flatten layer
            layers.Flatten(),

            # To FC layer 1
            layers.Dense(4096, activation = 'relu'),
            layers.Dropout(0.5),

            #To FC layer 2
            layers.Dense(4096, activation = 'relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation = 'sigmoid')
        ])

    def __train(self):
        train_datagen = ImageDataGenerator(rescale = 1.0 / 255,
            rotation_range = 40,
            width_shift_range = 0.2,
            height_shift_range = 0.2,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            fill_mode = 'nearest'
        )

        test_datagen = ImageDataGenerator(rescale = 1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            f'{os.getcwd()}/data/Dataset/Train',
            batch_size = 256 ,
            class_mode = 'binary',
            target_size = (64, 64)
        )

        validation_generator = test_datagen.flow_from_directory(
            f'{os.getcwd()}/data/Dataset/Validation',
            batch_size  = 256,
            class_mode  = 'binary',
            target_size = (64, 64)
        )

        self.model.compile(
            optimizer = Adam(lr = 0.001),
            loss = 'binary_crossentropy',
            metrics = ['accuracy']
        )

        self.model.fit_generator(generator = train_generator,
            validation_data = validation_generator,
            # steps_per_epoch = 256,
            # validation_steps = 256,
            epochs = 10
        )

        self.model.save(self.MODEL_PATH)

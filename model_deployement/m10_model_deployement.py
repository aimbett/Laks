
#!/usr/bin/python

import pandas as pd
from tensorflow import keras
#from sklearn.externals import joblib
import joblib
import sys
import os
from PIL import Image
from io import BytesIO

batch_size = 32
img_height = 180
img_width = 180
class_names = ['Lice', 'No Lice']


def predict_image(image: Image.Image):
    reconstructed_model = keras.models.load_model(os.path.dirname(__file__) +"/model.h5")

    #img = keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img = np.asarray(image.resize((img_height, img_width)))[..., :3]
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = reconstructed_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)) 
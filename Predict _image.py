# predict_image.py
# Load saved model and predict plant leaf disease

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model("plant_model.h5")

# Class labels
classes = ["Healthy", "Rust", "Angular Leaf Spot"]

def predict_leaf(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    class_idx = np.argmax(prediction)
    return classes[class_idx]

# Example
print(predict_leaf("test_leaf.jpg"))

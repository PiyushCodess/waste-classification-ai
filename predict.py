from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

# Load the trained model
model = load_model("waste_classifier_model.keras")

# Load an image to predict
img_path = "C:\\Users\\PIYUSH PATRIKAR\\Desktop\\waste_data\\DATASET\\TEST\\O\\O_12568.jpg"  # 🔁 Change this to your test image
img = image.load_img(img_path, target_size=(150, 150))  # size must match training input
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

# Predict
prediction = model.predict(img_array)
class_names = ['organic', 'recyclable', 'hazardous']  # 🔁 Replace with your actual classes
predicted_class = class_names[np.argmax(prediction)]

print(f"Predicted Class: {predicted_class}")

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 18:10:36 2025

@author: 91790
"""

import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# File ID and expected output model name
file_id = "1cLIU-JCglubQLzt6zih52V0VajP2mNjL"
model_path = "model.keras"

# Download model if not present
if not os.path.exists(model_path):
    st.write("Downloading model, please wait...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# Load model
model = tf.keras.models.load_model(model_path)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title("Object Recognition App")
st.write("Upload a 32x32 image (CIFAR-10 style) to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Prepare image
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"Prediction: **{predicted_class}**")

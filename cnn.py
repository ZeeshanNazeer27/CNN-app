import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image

model = load_model('E:\Internship\CNN\cnn.h5')

st.title("Cheetah or Hyena Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0 
    image = image.reshape(1, 256, 256, 3)

    prediction = model.predict(image)
    if prediction < 0.5:
        st.write("Prediction: Cheetah")
    else:
        st.write("Prediction: Hyena")



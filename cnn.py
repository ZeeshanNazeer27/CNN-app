import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os


model_path = 'cnn.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error("Model file not found. Please upload the model file.")

st.title("Cheetah or Hyena Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
   
    image = image.resize((256, 256))
    image = np.array(image)
    image = image / 255.0
    
    image = image.reshape(1, 256, 256, 3)

    prediction = model.predict(image)
    if prediction < 0.5:
        st.write("Prediction: Cheetah")
    else:
        st.write("Prediction: Hyena")

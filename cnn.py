import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

# Load model
model_path = 'cnn.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error("Model file not found. Please upload the model file.")

st.title("Cheetah or Hyena Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Resize the image to 256x256 using Pillow
    image = image.resize((256, 256))
    
    # Convert the image to a numpy array and normalize it
    image = np.array(image)
    image = image / 255.0
    
    # Reshape the image for the model
    image = image.reshape(1, 256, 256, 3)

    # Predict using the model
    prediction = model.predict(image)
    if prediction < 0.5:
        st.write("Prediction: Cheetah")
    else:
        st.write("Prediction: Hyena")

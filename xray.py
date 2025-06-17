import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(r'C:\Users\atulb\OneDrive\Documents\Desktop\Model\pneumonia_cnn_model.keras')
    return model

model = load_model()

st.title("🫁 Pneumonia Detection from Chest X-ray Images")

uploaded_file = st.file_uploader("📁 Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)

    if st.button('🔍 Predict'):
        # Preprocess the image
        img = image.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)

        # Display the result
        if prediction[0][0] > 0.5:
            st.error("🩺 **Prediction: Pneumonia Detected**")
        else:
            st.success("✅ **Prediction: Normal**")

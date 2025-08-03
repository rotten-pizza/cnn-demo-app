import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model (update custom_objects if you used GELU or other custom activations)
model = tf.keras.models.load_model("model.keras", custom_objects={"gelu": tf.keras.activations.gelu})

# Title
st.title("🧠 CNN Image Classifier")
st.markdown("Upload an image and the model will predict the class (Real or Fake).")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Class labels — update if yours are different
class_names = ["Real", "Fake"]

# Preprocessing function
def preprocess_image(image):
    image = image.convert("RGB")                # Ensure RGB
    image = image.resize((32, 32))              # Resize to match model input
    # image = np.array(image).astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)       # Add batch dimension
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        input_tensor = preprocess_image(image)
        prediction = model.predict(input_tensor)[0][0]  # For sigmoid output

        predicted_index = int(prediction > 0.5)
        confidence = prediction if predicted_index == 1 else 1 - prediction

        predicted_label = class_names[predicted_index]

        st.success(f"Prediction: **{predicted_label}** ({confidence:.2%} confidence)")

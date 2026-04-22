import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(page_title="Deepfake Detector", page_icon="🔍", layout="centered")

# --- Custom Styling to make it look professional ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { 
        width: 100%; 
        background-color: #4CAF50; 
        color: white; 
        font-weight: bold; 
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔍 Deepfake Image Detector")
st.write("Upload an image to check if the AI detects it as a **Real photo** or a **Deepfake**.")

# --- Model Loading Logic ---
@st.cache_resource
def load_my_model():
    # This looks for the file created by train_model.py
    model_path = 'deepfake_model.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_my_model()

# --- Check if the model exists ---
if model is None:
    st.error("❌ Model file 'deepfake_model.h5' not found!")
    st.info("You must run the training script first: `python3 train_model.py` in your terminal.")
    st.stop()

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Analysis Button
    if st.button("Analyze Image"):
        with st.spinner("AI is analyzing the pixels..."):
            try:
                # 1. Preprocess the image (Must match the training settings)
                img = image.convert('RGB').resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # 2. Make Prediction
                prediction = model.predict(img_array)[0][0]
                
                # 3. Show results
                st.markdown("---")
                # Logic: prediction > 0.5 is 'Real', < 0.5 is 'Fake'
                if prediction > 0.5:
                    confidence = prediction * 100
                    st.success(f"### ✅ Result: REAL\n**Confidence: {confidence:.2f}%**")
                    st.write("The AI believes this is a genuine photograph.")
                else:
                    confidence = (1 - prediction) * 100
                    st.error(f"### 🚨 Result: DEEPFAKE\n**Confidence: {confidence:.2f}%**")
                    st.write("The AI detected patterns common in AI-generated images.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")

st.markdown("---")
st.caption("Developed using CNN & Transfer Learning (MobileNetV2)")
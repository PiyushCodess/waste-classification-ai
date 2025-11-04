import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# ✅ Streamlit Page Config
st.set_page_config(page_title="♻️ Waste Classifier AI", layout="centered")

# ✅ App Title and Description
st.title("♻️ Waste Classification Using AI")
st.write("Upload an image of waste to classify it as **Organic** or **Recyclable** using a trained deep learning model.")

# ✅ Load the model (with caching to improve speed on Render)
@st.cache_resource
def load_waste_model():
    model_path = os.path.join(os.getcwd(), "waste_classifier.h5")
    return load_model(model_path)

model = load_waste_model()

# ✅ File Uploader
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "jpeg", "png"])

# ✅ Prediction Section
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼 Uploaded Image", use_column_width=True)

    if st.button("🔍 Classify"):
        with st.spinner("Analyzing..."):
            # Preprocess image
            img = img.resize((150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make prediction
            prediction = model.predict(img_array)

            # Display Result
            if prediction[0][0] > 0.5:
                st.success("♻️ Predicted: **Recyclable (Non-biodegradable)**")
            else:
                st.success("🌱 Predicted: **Organic (Biodegradable)**")

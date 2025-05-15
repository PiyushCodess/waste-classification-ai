import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = load_model('waste_classifier.h5')

# Streamlit UI
st.set_page_config(page_title="Waste Classifier AI", layout="centered")
st.title("♻️ Waste Classification Using AI")
st.write("Upload an image of waste to classify it as **Organic** or **Recyclable**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        # Preprocess image
        img = img.resize((150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict
        prediction = model.predict(img_array)

        if prediction[0][0] > 0.5:
            st.success("🟢 Predicted: Recyclable (Non-biodegradable)")
        else:
            st.success("🟢 Predicted: Organic (Biodegradable)")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Agri-Vision Diagnostics", layout="centered")
st.title("🌿 Agricultural Vision Diagnostic Tool")
st.write("Upload a clear photo of a plant leaf to check its health status.")

# Load the model and cache it to prevent lag on user clicks
@st.cache_resource
def load_diagnostic_model():
    return tf.keras.models.load_model("plant_model.h5")

model = load_diagnostic_model()

# These match the exact classes your model trained on in Colab
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Sample", use_container_width=True)
    
    # Process image to match what the model expects (224x224 pixels)
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Expand dimensions to create a batch of 1
    
    with st.spinner("Analyzing structural markers..."):
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    st.success(f"Analysis Complete!")
    st.subheader(f"Result: {predicted_class.title()}")
    st.write(f"Match Confidence: {confidence:.2f}%")
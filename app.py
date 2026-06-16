import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image

st.set_page_config(page_title="Agri-Vision Diagnostics", layout="centered")
st.title("🌿 Agricultural Vision Diagnostic Tool")
st.write("Upload a clear photo of a plant leaf to check its health status.")

@st.cache_resource
def load_onnx_session():
    return ort.InferenceSession("plant_model.onnx")

session = load_onnx_session()
input_name = session.get_inputs()[0].name

# Updated 38-class PlantVillage Diagnostic Matrix
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Sample", use_container_width=True)
    
    with st.spinner("Analyzing structural markers..."):
        # 1. Force convert to RGB (handles transparency channels in PNGs safely)
        img = image.convert("RGB")
        # 2. Resize to the exact dimension the model expects
        img = img.resize((224, 224))
        
        # 3. Process into float32 array and add batch dimension (NO manual division here!)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0) # Shape: (1, 224, 224, 3)
        
        # 4. Run inference using ONNX runtime
        probabilities = session.run(None, {input_name: img_array})[0]
        
        # 5. Extract predictions directly (since the model output is already post-softmax probabilities)
        class_idx = np.argmax(probabilities[0])
        predicted_class = CLASS_NAMES[class_idx]
        confidence = 100 * probabilities[0][class_idx]
        
    # Format the folder structure name to look beautiful on the UI
    clean_display_name = predicted_class.replace("___", " - ").replace("_", " ")
    
    st.success(f"Analysis Complete!")
    st.subheader(f"Result: {clean_display_name}")
    st.write(f"Match Confidence: {confidence:.2f}%")
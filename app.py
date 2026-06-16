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

   CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

   uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

   if uploaded_file is not None:
       image = Image.open(uploaded_file)
       st.image(image, caption="Uploaded Sample", use_container_width=True)
       
       # Process image to match what the model expects
       img = image.resize((224, 224))
       img_array = np.array(img, dtype=np.float32)
       img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, 224, 224, 3)
       
       with st.spinner("Analyzing structural markers..."):
           # Run inference using ONNX runtime instead of heavy TensorFlow
           raw_preds = session.run(None, {input_name: img_array})[0]
           # Apply softmax manually
           exp_preds = np.exp(raw_preds - np.max(raw_preds))
           probabilities = exp_preds / exp_preds.sum(axis=1, keepdims=True)
           
       predicted_class = CLASS_NAMES[np.argmax(probabilities[0])]
       confidence = 100 * np.max(probabilities[0])
       
       st.success(f"Analysis Complete!")
       st.subheader(f"Result: {predicted_class.title()}")
       st.write(f"Match Confidence: {confidence:.2f}%")
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function to make predictions
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    
    # Convert uploaded file to a PIL image
    image = Image.open(test_image)
    image = image.resize((128, 128))  # Resize image to match model input
    input_arr = np.array(image) / 255.0  # Normalize pixel values
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("Plant Disease System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])

# Main Image (with reduced height)
img = Image.open('Diseases.jpg')
img = img.resize((img.width, int(img.height * 0.7)))  # Reduce height by 50%
st.image(img)

# Home Page
if app_mode == 'Home':
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", 
                unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == 'Disease Recognition':
    st.header('Plant Disease Detection System For Sustainable Agriculture')

    # Upload Image
    test_image = st.file_uploader('Choose an image:')

    if test_image:
        # Show Image Button
        if st.button('Show Image'):
            st.image(test_image, use_column_width=True)

        # Predict Button
        if st.button('Predict'):
            st.snow()
            result_index = model_prediction(test_image)
            class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            st.success(f'Model is predicting it’s a {class_name[result_index]}')

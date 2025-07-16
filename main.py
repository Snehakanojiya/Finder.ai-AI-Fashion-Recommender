import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Load image features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load ResNet50 model for image feature extraction
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Session state for cart and wishlist
if 'cart' not in st.session_state:
    st.session_state.cart = []
if 'wishlist' not in st.session_state:
    st.session_state.wishlist = []

st.title('🧠 Finder.ai — AI Fashion Recommender')

# File upload
uploaded_file = st.file_uploader("📁 Upload a fashion image")

# Save file locally
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return True
    except:
        return False

# Feature extraction from image
def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# Recommendation logic
def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# Display recommendations
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        st.image(Image.open(uploaded_file), caption='Uploaded Image', use_container_width=True)
        features = feature_extraction(os.path.join("uploads", uploaded_file.name), model)
        indices = recommend(features, feature_list)

        st.subheader("🛍️ Recommended Products")
        cols = st.columns(5)

        for i, col in enumerate(cols):
            file_index = indices[0][i + 1]  # skip the first one (it's the uploaded image)
            product_path = filenames[file_index]
            col.image(product_path, use_container_width=True)

            if col.button("🛒 Add to Cart", key=f"cart{i}"):
                st.session_state.cart.append(product_path)

            if col.button("❤️ Wishlist", key=f"wish{i}"):
                st.session_state.wishlist.append(product_path)

# Cart and Wishlist Display
st.sidebar.title("🧾 Your Selection")
st.sidebar.subheader("🛒 Cart")
for item in st.session_state.cart:
    st.sidebar.image(item, width=100)

st.sidebar.subheader("❤️ Wishlist")
for item in st.session_state.wishlist:
    st.sidebar.image(item, width=100)

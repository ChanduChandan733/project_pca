import streamlit as st
import os
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

# Set up the upload folder
UPLOAD_FOLDER = 'static/uploads/'

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to process and compress the image
def reduce_image(image_path, accuracy_level):
    # Load the original image
    image = io.imread(image_path)
    
    # Convert to grayscale for simplicity (optional for color images)
    gray_image = color.rgb2gray(image)

    # Apply PCA for Dimensionality Reduction with specified accuracy level
    pca = PCA(n_components=accuracy_level)
    transformed_image = pca.fit_transform(gray_image)

    # Reconstruct the compressed image
    reconstructed_image = pca.inverse_transform(transformed_image)

    # Normalize and save the compressed image
    compressed_image_normalized = (reconstructed_image - reconstructed_image.min()) / (reconstructed_image.max() - reconstructed_image.min())
    compressed_image_uint8 = img_as_ubyte(compressed_image_normalized)
    
    # Save the compressed image
    compressed_image_path = os.path.join(UPLOAD_FOLDER, f'compressed_{os.path.basename(image_path)}')
    io.imsave(compressed_image_path, compressed_image_uint8)
    
    return compressed_image_path

# Streamlit App
def main():
    # Custom CSS for ash background and styled text
    st.markdown(
        """
        <style>
        body {
            background-color: #2e2e2e;
            color: white;
        }
        .title {
            font-weight: bold;
            color: #e50914;
            text-align: center;
            margin-bottom: 20px;
        }
        .subtitle {
            font-weight: bold;
            color: #00ffcc;
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #2e2e2e;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h1 class="title">Image Compression with PCA</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subtitle">Upload an image and select compression accuracy level</h3>', unsafe_allow_html=True)

    # Sidebar options
    st.sidebar.markdown('<h2 style="color: #e50914; font-weight: bold;">Select Accuracy Level</h2>', unsafe_allow_html=True)
    accuracy = st.sidebar.radio(
        "Choose a compression accuracy level:",
        ('80%', '90%', '95%', '99%'),
        index=1
    )
    
    # Map accuracy options to numeric values
    accuracy_map = {'80%': 0.8, '90%': 0.9, '95%': 0.95, '99%': 0.99}
    accuracy_level = accuracy_map[accuracy]

    # File uploader
    uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display uploaded image
        st.markdown('<h3 class="subtitle">Uploaded Image</h3>', unsafe_allow_html=True)
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process and compress the image
        compressed_image_path = reduce_image(file_path, accuracy_level)

        # Display the compressed image
        st.markdown('<h3 class="subtitle">Compressed Image</h3>', unsafe_allow_html=True)
        st.image(compressed_image_path, caption="Compressed Image", use_container_width=True)

        # Download option
        with open(compressed_image_path, "rb") as f:
            compressed_image_bytes = f.read()
            st.download_button(
                label="Download Compressed Image",
                data=compressed_image_bytes,
                file_name=os.path.basename(compressed_image_path),
                mime="image/png"
            )

if __name__ == "__main__":
    main()

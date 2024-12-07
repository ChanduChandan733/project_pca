from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from sklearn.decomposition import PCA
from skimage import io, color
from skimage.util import img_as_ubyte

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')  # Renders the initial input form

@app.route('/upload', methods=['POST'])
def upload():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return "No file uploaded", 400

    image = request.files['image']
    accuracy = request.form.get('accuracy')

    # Check if a file was selected
    if image.filename == '':
        return "No file selected", 400

    # Check if accuracy level is provided and valid
    if accuracy is None:
        return "Accuracy level not provided", 400

    try:
        accuracy_level = float(accuracy)
    except ValueError:
        return "Invalid accuracy level format", 400

    # Save the image to the specified folder
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(image_path)

    # Process and compress the image
    compressed_image_path = reduce_image(image_path, accuracy_level)
    return render_template('success.html', compressed_image_path=compressed_image_path)

def reduce_image(file_name, accuracy_level):
    # Load the original image
    image = io.imread(file_name)
    
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
    
    # Save the compressed image in the upload folder
    compressed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed_' + os.path.basename(file_name))
    io.imsave(compressed_image_path, compressed_image_uint8)
    print("Image has been successfully compressed")
    return 'compressed_' + os.path.basename(file_name)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

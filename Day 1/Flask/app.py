from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from io import BytesIO
import numpy as np

app = Flask(__name__)

# Load Model
model = tf.keras.models.load_model('Id Dectector.h5')  # Corrected method and assumed filename (update if different)

@app.route('/')
def home():
    return "Welcome to ID Card Detector! Upload an image to predict"  # Fixed typo in "Detector"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400  # Improved error message

    file = request.files['file'].read()
    img = Image.open(BytesIO(file))  
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    prediction = model.predict(img_array)[0][0]
    result = "ID Card" if prediction > 0.5 else "Not an ID Card"  # Simplified "Not a Real ID Card" for clarity
    confidence = float(prediction)
    
    return jsonify({'result': result, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
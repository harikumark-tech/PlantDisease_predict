from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

MODEL_PATH = 'models/mobilenet_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Change this order to match your training
CLASS_NAMES = ['Blight', 'Common_Rust', 'Grey_Leaf_Spot', 'Healthy']

def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file.stream)
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return render_template('result.html', class_name=predicted_class, confidence=round(confidence * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)

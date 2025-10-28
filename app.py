from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# ======== Load trained model ========
MODEL_PATH = 'best_model.h5'   # ✅ Put your model file in the same folder
model = tf.keras.models.load_model(MODEL_PATH)

# Update with your class names
CLASS_NAMES = ['Healthy', 'Disease_1', 'Disease_2', 'Disease_3']


# ======== Helper: preprocess image ========
def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# ======== Home Page ========
@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Plant Disease Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #f4f9f4; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { text-align: center; color: #2e7d32; }
            .upload-box { border: 2px dashed #4CAF50; padding: 30px; text-align: center; margin: 30px 0; border-radius: 10px; background-color: #fff; }
            button { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            button:hover { background-color: #45a049; }
        </style>
    </head>
    <body>
        <h1>🌱 Plant Disease Predictor</h1>
        <div class="upload-box">
            <h3>Upload a Plant Leaf Image</h3>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required>
                <br><br>
                <button type="submit">Predict Disease</button>
            </form>
        </div>
    </body>
    </html>
    '''


# ======== Prediction Route ========
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    try:
        # Read and preprocess image
        img = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(img)
        
        # Make prediction
        predictions = model.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        # Return result page
        return f'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Prediction Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; background-color: #f0fff0; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2e7d32; }}
                .result-box {{ background-color: #e8f5e9; padding: 25px; border-radius: 10px; margin-top: 20px; }}
                .back-btn {{ background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; text-decoration: none; }}
                .back-btn:hover {{ background-color: #45a049; }}
            </style>
        </head>
        <body>
            <h1>🔍 Prediction Results</h1>
            <div class="result-box">
                <h3>Predicted Disease: {predicted_class}</h3>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <h4>All Class Probabilities:</h4>
                <ul>
                    {"".join([f"<li>{name}: {prob:.2%}</li>" for name, prob in zip(CLASS_NAMES, predictions[0])])}
                </ul>
            </div>
            <a href="/" class="back-btn">Upload Another Image</a>
        </body>
        </html>
        '''
    
    except Exception as e:
        return jsonify({'error': str(e)})


# ======== Run Flask App ========
if __name__ == '__main__':
    # Runs on your local system
    app.run(debug=True, host='0.0.0.0', port=5000)

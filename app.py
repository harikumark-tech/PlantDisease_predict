import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # optional - removes performance warnings

from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from tensorflow.keras import layers

app = Flask(__name__)

# ======== Define Custom Layers for ViT ========
from tensorflow.keras import layers

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded



# ======== Load All Models ========
print("Loading models... please wait")

# Load all TFLite models (CNN, MobileNet, ResNet, ViT)
tflite_models = {
    "CNN": tf.lite.Interpreter(model_path=r"models\plant_disease_cnn_model_fp16.tflite"),
    "MobileNet": tf.lite.Interpreter(model_path=r"models\mobilenet_model_fp16.tflite"),
    "ResNet": tf.lite.Interpreter(model_path=r"models\resnet_model_fp16.tflite"),
    #"ViT": tf.lite.Interpreter(model_path=r"models\vit_model_128x128_fp16.tflite"),
}

# Allocate tensors for all models
for name, interpreter in tflite_models.items():
    interpreter.allocate_tensors()

vit_model = tf.keras.models.load_model(
    r"models\vit_model_128x128.h5",
    custom_objects={"Patches": Patches, "PatchEncoder": PatchEncoder}
)

print("All TFLite models loaded successfully!")


# ======== Class Labels ========
CLASS_NAMES = ['Blight', 'Common_Rust', 'Grey_Leaf_Spot', 'Healthy']


# ======== Image Preprocessing Function ========
def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


# ======== Flask Routes ========
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    model_choice = request.form['model']
    img = Image.open(file.stream)
    img_array = preprocess_image(img)

    # the vision transformer model
    if model_choice == "ViT":
        predictions = vit_model.predict(img_array)

    #TFLite models
    else:
        interpreter = tflite_models[model_choice]
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    return render_template(
        'result.html',
        class_name=predicted_class,
        confidence=round(confidence, 2),
        model_used=model_choice
    )


if __name__ == '__main__':
    app.run(debug=True)


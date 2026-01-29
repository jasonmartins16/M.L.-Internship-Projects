from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the model
model = load_model('MobileNetV2.h5')

#Class labels
class_labels = ['Bear', 'Bird', 'Cat', 'Cow', 'Deer', 'Dog', 'Dolphin', 'Elephant', 'Girrafe', 'Horse', 'Kangaroo', 'Lion', 'Panda', 'Tiger', 'Zebra']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis = 0)
    return img_array

@app.route("/predict", methods = ["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected image"}), 400
    
    #Save and preprocess image
    filepath = os.path.join("temp.jpg")
    file.save(filepath)
    img = preprocess_image(filepath)
    
    #Make prediction
    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]

    #Clean up
    os.remove(filepath)

    return jsonify({
        "animal": predicted_class,
        #"confidence": float(np.max(prediction))
    })

if __name__ == "__main__":
    app.run(debug = True)
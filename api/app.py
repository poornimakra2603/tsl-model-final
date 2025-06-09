from flask import Flask, request, jsonify
import pickle
import numpy as np
from PIL import Image
import io
from flask_cors import CORS

labels = ['Crying', 'Eat', 'Drink', 'Hello', 'Welcome']

app = Flask(__name__)
CORS(app)
# Load model
with open("tsl_model.pkl", "rb") as f:
    model = pickle.load(f)

# Tamil translations mapping
translations = {
    'Crying': 'அழுகை',
    'Eat': 'சாப்பிடு',
    'Drink': 'பானம் குடித்தல்',
    'Hello': 'வணக்கம்',
    'Welcome': 'வரவேற்பு',
}

def preprocess_image(file):
    """Preprocess image from an uploaded file (corrected)"""
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = image.resize((64, 64))
        img_array = np.array(image) / 255.0  # Normalize properly
        img_array = img_array.flatten().reshape(1, -1)  # Ensure correct shape
        return img_array
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

@app.route('/predict', methods=["POST"])
def predict():
    """Handle image predictions"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = preprocess_image(file)

    if image is None:
        return jsonify({"error": "Image processing failed"}), 500

    try:
        index_prediction = int(model.predict(image)[0])  # Convert to integer
        print(f"Predicted Index: {index_prediction}")  # Debugging Step

        # Map index to actual label
        if 0 <= index_prediction < len(labels):
            label_prediction = labels[index_prediction]  # Retrieve actual label
        else:
            label_prediction = "Unknown Sign"  # Handle unexpected index

        print(f"Mapped Label: {label_prediction}")  # Debugging Step

        # Retrieve Tamil translation
        result_text = translations.get(label_prediction, "Unknown Sign")
        print(f"Mapped Tamil Translation: {result_text}")  # Debugging Step

        return jsonify({"prediction": result_text})  # Send Tamil output

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)

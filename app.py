from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Class names (38 classes)
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew',
    'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___healthy', 'Corn___Northern_Leaf_Blight',
    'Grape___Black_rot', 'Grape___Esca', 'Grape___healthy', 'Grape___Leaf_blight',
    'Orange___Haunglongbing', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy',
    'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight',
    'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# Load model
model = None
model_loaded = False

def load_model():
    global model, model_loaded
    model_path = 'models/plant_disease_model.h5'
    
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            model_loaded = True
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    else:
        print(f"⚠️ Model not found at {model_path}")

load_model()

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, None
    except Exception as e:
        return None, str(e)

def predict_disease(image_bytes):
    if not model_loaded:
        return {'success': False, 'error': 'Model not loaded'}
    
    try:
        img_array, error = preprocess_image(image_bytes)
        if error:
            return {'success': False, 'error': error}
        
        predictions = model.predict(img_array, verbose=0)
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        
        results = []
        for idx in top_indices:
            disease_name = class_names[idx]
            confidence = float(predictions[0][idx] * 100)
            parts = disease_name.split('___')
            plant = parts[0]
            condition = parts[1].replace('_', ' ')
            
            results.append({
                'plant': plant,
                'disease': condition,
                'confidence': round(confidence, 2)
            })
        
        return {'success': True, 'predictions': results, 'top_prediction': results[0]}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    try:
        file_bytes = file.read()
        result = predict_disease(file_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'running', 'model_loaded': model_loaded})

if __name__ == '__main__':
    print("\n🌿 PlantGuard AI - Plant Disease Detection System")
    print("="*50)
    print("🚀 Server running at: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)
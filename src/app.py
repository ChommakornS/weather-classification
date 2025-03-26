from model import WeatherClassifier
from flask import Flask, request, render_template, redirect, url_for
import io
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import logging
import uuid
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Update your Flask app configuration
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# Configure upload folder correctly
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In app.py, replace the relevant section
from model import WeatherClassifier

# Initialize model
classifier = WeatherClassifier()
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'weather_classifier.pth')

# Load model
try:
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        classifier.load_model(model_path)
        # Note: Don't call classifier.eval() here - your class doesn't have this method
        # The model inside your class is already set to eval mode in your load_model method
        logger.info("Model loaded successfully")
    else:
        logger.error(f"Model file not found at {model_path}")
        logger.warning("Using untrained model.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.warning("Using untrained model.")

# Weather class labels - replace this with your actual classes
weather_classes = ["rainbow", "lightning", "snow", "sandstorm"]


def preprocess_image(image_bytes):
    """
    Preprocess the image bytes before feeding into the model
    """
    try:
        # Use the preprocess_image method from your WeatherClassifier class
        tensor = classifier.preprocess_image(image_bytes)
        return tensor
    except Exception as e:
        logger.error(f"Error in preprocessing image: {str(e)}")
        raise


# Save the uploaded image and return a path relative to the static folder
def save_uploaded_image(file_data):
    """
    Save the uploaded image to the server
    """
    try:
        # Generate a unique filename
        unique_filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # Path to save the file, relative to the static folder
        relative_path = f"uploads/{unique_filename}"
        
        # Create absolute path for saving
        file_path = os.path.join('static', relative_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        logger.debug(f"Saving image to: {file_path}")
        
        # Save the image
        image = Image.open(io.BytesIO(file_data))
        
        # Convert to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        # Save the image
        image.save(file_path)
        
        logger.debug(f"Image saved successfully at {file_path}")
        return relative_path  # Return path relative to static folder
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        raise


def get_prediction(tensor):
    """
    Generate prediction from model output
    """
    try:
        # Use the predict method from your WeatherClassifier class
        predicted_class, confidence = classifier.predict(tensor)
        
        return {
            'prediction': predicted_class,
            'confidence': f"{confidence:.1f}%"
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        # Fallback to a default prediction if model fails
        return {
            'prediction': 'unknown',
            'confidence': 'N/A (model error)'
        }


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.debug("Received prediction request")
        
        # Check if file part exists in request
        if 'file' not in request.files:
            logger.error("No file part in the request")
            return render_template('index.html', error="No file uploaded")
        
        file = request.files['file']
        
        # Check if the file was selected
        if file.filename == '':
            logger.error("No file selected")
            return render_template('index.html', error="No file selected")
        
        # Read file data
        img_bytes = file.read()
        
        if not img_bytes:
            logger.error("Empty file uploaded")
            return render_template('index.html', error="Empty file uploaded")
            
        logger.debug(f"File received: {file.filename}, size: {len(img_bytes)} bytes")
        
        # Save the uploaded image
        image_path = save_uploaded_image(img_bytes)
        logger.debug(f"Image saved with path: {image_path}")
        
        # Create a fresh copy of the image bytes for preprocessing
        file.seek(0)
        img_bytes = file.read()
        
        # Preprocess the image for the model
        tensor = preprocess_image(img_bytes)
        
        # Make prediction using the model
        prediction_result = get_prediction(tensor)
        
        # Complete the prediction data with the image path
        prediction_result['image_path'] = image_path
        
        # Add debugging for the full URL
        full_url = url_for('static', filename=image_path)
        logger.debug(f"Full image URL: {full_url}")
        
        # Log the complete prediction result
        logger.debug(f"Complete prediction result: {prediction_result}")
        
        # Return result template
        return render_template('result.html', result=prediction_result)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return render_template('index.html', error=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

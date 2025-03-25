import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io

class WeatherClassifier:
    def __init__(self):
        # Initialize device (CPU or GPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize model structure
        self.model = models.resnet34(pretrained=False)
        self.class_names = []
        
    def load_model(self, model_path):
        """
        Load the trained model from a checkpoint file
        """
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get class names from checkpoint
        self.class_names = checkpoint['class_names']
        
        # Set up model structure
        num_classes = len(self.class_names)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded with classes: {self.class_names}")
        
    def predict(self, img_tensor):
        """
        Make a prediction on the input image tensor
        
        Args:
            img_tensor: Preprocessed image tensor
            
        Returns:
            prediction: Class name
            confidence: Prediction confidence percentage
        """
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Move tensor to the right device
        img_tensor = img_tensor.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Convert to class name and confidence percentage
            prediction = self.class_names[predicted_idx.item()]
            confidence_pct = confidence.item() * 100
            
        return prediction, confidence_pct
        
    def preprocess_image(self, image_bytes):
        """
        Process a raw image from bytes
        
        Args:
            image_bytes: Raw image data
            
        Returns:
            tensor: Preprocessed image tensor
        """
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return transform(image).unsqueeze(0)
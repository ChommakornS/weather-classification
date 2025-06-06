# Weather Image Classification System

A complete end-to-end system for classifying weather images into different categories (rainbow, lightning, snow, sandstorm) using deep learning with PyTorch and ResNet34. This project is deployed using Google Cloud Run, which provides serverless container deployment. The live application can be accessed at:
[Link](https://weather-classification-655537561777.us-central1.run.app/)

## Project Structure
```
weather-classification/
├── notebooks/
│   ├── 01_model_training.ipynb
│   └── 02_model_evaluation.ipynb
├── src/
│   ├── app.py
│   ├── model.py
│   ├── utils.py
│   ├── templates/ 
│   │    ├── result.html
│   │    ├── index.html
│   └──static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           └── script.js
├── data/
│   ├── train/
│   │   ├── rainbow/
│   │   ├── lightning/
│   │   ├── snow/
│   │   └── sandstorm/
│   ├── valid/
│   │   ├── rainbow/
│   │   ├── lightning/
│   │   ├── snow/
│   │   └── sandstorm/
│   └── test/
│       ├── rainbow/
│       ├── lightning/
│       ├── snow/
│       └── sandstorm/
├── models/
│   └── weather_classifier.pth
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup and Installation

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- Flask
- Docker (optional)

### Dataset and Model Performance

#### Dataset Size
- Training set: 170 images (split across 4 weather categories)
- Validation set: 30 images
- Test set: 30 images

#### Model Performance
- **Overall Accuracy: 98.33%** on the test set
- **Confusion Matrix Analysis:**
  - Lightning class: 30/30 correctly classified (100% accuracy)
  - Rainbow class: 29/30 correctly classified with 1 misclassification as sandstorm
  - Sandstorm class: 30/30 correctly classified (100% accuracy)
  - Snow class: 29/30 correctly classified with 1 misclassification as sandstorm
  - The model shows excellent performance with only 2 misclassifications out of 120 total test cases
  - The sandstorm class appears to occasionally be confused with rainbow and snow images

## Limitations and Challenges

### Dataset Limitations
- **Small Dataset Size**: With only 170 training images across 4 categories, the model may not generalize well to real-world images that differ significantly from the training distribution.
- **Limited Variety**: Each weather phenomenon can appear in various forms and conditions that might not be fully represented in our dataset.
- **Potential Sampling Bias**: The dataset may overrepresent certain visual patterns or contexts, leading to a falsely high performance on the test set.

### Model Limitations
- **Potentially Overconfident Predictions**: The model occasionally shows 100% confidence in its predictions, which is rarely justified in real-world scenarios with natural variation.
- **Perfect Class Accuracy Concerns**: The 100% accuracy for certain classes (lightning, sandstorm) on the test set should be interpreted with caution and is likely due to:
  - Limited test set size (only 30 images per class)
  - Possible similarities between test and training images
  - Potential data leakage or overfitting to the specific characteristics of the dataset

### Evaluation Considerations
- **Test Set Selection**: The current evaluation method uses a fixed test set. Cross-validation would provide a more robust performance estimate.
- **Confidence Calibration**: The raw model outputs should be properly calibrated to reflect true predictive uncertainty rather than showing overconfident predictions.
- **Real-world Performance Gap**: Performance in controlled test environments often exceeds real-world performance due to distribution shifts and unexpected variations.

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/weather-classification.git
cd weather-classification
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the pre-trained model:
   - Download the model file from [Google Drive](https://drive.google.com/file/d/1HX1cloC4rsTz_qFjtbHKGgPV9uanuiyA/view?usp=sharing)
   - Create the `models` directory if it doesn't exist already: `mkdir -p models`
   - Place the downloaded file in the `models` directory with the filename `weather_classifier.pth`

4. Prepare the dataset:
   - Download the weather dataset from [Kaggle](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset)
   - Run the data preparation notebook to organize the dataset into train/valid/test splits

5. Train the model:
   - Run the model training notebook
   - The trained model will be saved to the `models/` directory

6. Start the Flask application:
```bash
python src/app.py
```

7. Open your browser and navigate to `http://localhost:5000`

### Docker Setup

1. Build the Docker image:
```bash
docker build -t weather-classifier .
```

2. Run the container:
```bash
docker run -p 5000:5000 weather-classifier
```

3. Install Docker:
```bash
# For Amazon Linux 2
sudo yum update -y
sudo amazon-linux-extras install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
# Log out and log back in

# For Ubuntu
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ubuntu
# Log out and log back in
```

4. Clone repository and use Docker image to run:
```bash
git clone https://github.com/yourusername/weather-classification.git
cd weather-classification
docker build -t weather-classifier .
docker run -d -p 80:5000 weather-classifier
```

## Cloud Deployment

### Google Cloud Platform (GCP) Deployment


#### Deployment Steps

1. Set up Google Cloud SDK and authenticate:
```bash
gcloud auth login
gcloud config set project your-project-id
```

2. Build and push the Docker image to Google Container Registry:
```bash
gcloud builds submit --tag gcr.io/your-project-id/weather-classifier
```

3. Deploy to Cloud Run:
```bash
gcloud run deploy weather-classification \
  --image gcr.io/your-project-id/weather-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

4. CI/CD Pipeline with Cloud Build:
   - Created a `cloudbuild.yaml` configuration file for automated deployments
   - Set up triggers to automatically deploy on commits to the main branch
   - Implemented testing steps before deployment to ensure application stability

#### GCP Services Used

- **Cloud Run**: Serverless container platform for the application
- **Cloud Build**: CI/CD pipeline for automated testing and deployment
- **Container Registry**: Storage for Docker images
- **Cloud Storage**: Storage of model weights and static assets
- **Cloud Logging**: Monitoring application performance and errors
- **Cloud IAM**: Managing access and permissions

## Usage

1. Open the web interface in your browser
2. Upload an image of a weather phenomenon by dragging and dropping or using the Browse Files button
3. The system will classify the image and display the result with confidence score

![Weather Image Classifier Interface](images/index.png)
![Classification Result Example](images/result.png)

The web application provides a clean interface with descriptions of each weather phenomenon:
- **Rainbow**: Optical phenomenon caused by reflection, refraction and dispersion of light in water droplets
- **Lightning**: Naturally occurring electrostatic discharge during which two electrically charged regions temporarily equalize
- **Snow**: Precipitation in the form of ice crystals, mainly of intricately branched, hexagonal form
- **Sandstorm**: Meteorological phenomenon in which strong winds blow large amounts of sand, reducing visibility

## Model Information

- Base architecture: ResNet34
- Transfer learning approach:
  - Feature extraction: All layers frozen except the last one
  - Fine-tuning: Last few layers unfrozen and retrained

### Model Robustness Measures

To address concerns about the high accuracy and potential overfitting:

- **Regularization Techniques**: We applied dropout and weight decay during training to prevent overfitting
- **Data Augmentation**: Training images were augmented with random rotations, flips, and color jitter to improve generalization
- **Validation Monitoring**: Early stopping was implemented based on validation loss to prevent memorization of training data

## Dataset

The model is trained on the [Weather Dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset) from Kaggle, which includes images of various weather conditions. For this project, we focus on four categories:
- Rainbow
- Lightning
- Snow
- Sandstorm

## Future Work

Some potential improvements for future versions include:

- **Improved Confidence Calibration**: While the model achieves high accuracy, the confidence scores (sometimes showing 100%) could be better calibrated to reflect true predictive uncertainty
- **Expanded Test Dataset**: Collect more diverse test examples to better evaluate real-world performance
- **External Validation**: Test the model on completely independent datasets from different sources
- **Adding More Weather Classes**: Expand beyond the current four classes to include fog, hail, cloud types, etc.
- **Advanced Data Augmentation**: Increase dataset size through more sophisticated augmentation techniques to improve model robustness
- **Adversarial Testing**: Evaluate model performance on adversarially generated images to identify weaknesses
- **Ensemble Methods**: Implement model ensembles to improve reliability and provide better uncertainty estimates
- **Mobile Application**: Develop a mobile version for on-the-go weather classification
- **Integration with Weather APIs**: Combine image classification with data from weather services for more comprehensive analysis
- **Regional Adaptation**: Fine-tune models for different geographical regions with distinctive weather patterns

## Known Issues and Troubleshooting

- **High Confidence Outputs**: The model may display unrealistically high confidence (e.g., 100%) even when uncertain. This is a known limitation of softmax outputs in neural networks.
- **Limited Generalization**: Performance may degrade significantly on images captured under different conditions than those in the training set.
- **Edge Cases**: Certain mixed weather conditions (e.g., lightning during a sandstorm) may confuse the classifier.

## Acknowledgments

- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Kaggle](https://www.kaggle.com/) for the dataset

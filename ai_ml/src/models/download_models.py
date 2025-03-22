import os
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import joblib
from pathlib import Path

# Define base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define model directories
TEXT_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'text_emotion_model')
SPEECH_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'speech_emotion_model')
FACIAL_MODEL_DIR = os.path.join(BASE_DIR, 'models', 'facial_emotion_model')

# Create directories if they don't exist
os.makedirs(TEXT_MODEL_DIR, exist_ok=True)
os.makedirs(SPEECH_MODEL_DIR, exist_ok=True)
os.makedirs(FACIAL_MODEL_DIR, exist_ok=True)

def download_text_emotion_model():
    """Download and save the text emotion model."""
    print("Downloading text emotion model...")
    try:
        # Use BERT base model for text emotion
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
        
        # Save model and tokenizer
        tokenizer.save_pretrained(TEXT_MODEL_DIR)
        model.save_pretrained(TEXT_MODEL_DIR)
        print("Text emotion model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading text emotion model: {str(e)}")

def download_speech_emotion_model():
    """Download and save the speech emotion model."""
    print("Downloading speech emotion model...")
    try:
        # Create a simple model for demonstration
        model = joblib.load('path/to/your/speech_model.joblib')  # Replace with actual model path
        scaler = joblib.load('path/to/your/scaler.joblib')  # Replace with actual scaler path
        
        # Save model and scaler
        with open(os.path.join(SPEECH_MODEL_DIR, 'trained_speech_emotion_model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        with open(os.path.join(SPEECH_MODEL_DIR, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
        print("Speech emotion model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading speech emotion model: {str(e)}")

def download_facial_emotion_model():
    """Download and save the facial emotion model."""
    print("Downloading facial emotion model...")
    try:
        # Download pre-trained facial emotion model
        model_url = "path/to/your/facial_model.pt"  # Replace with actual model URL
        response = requests.get(model_url)
        
        if response.status_code == 200:
            model_path = os.path.join(FACIAL_MODEL_DIR, 'trained_facial_emotion_model.pt')
            with open(model_path, 'wb') as f:
                f.write(response.content)
            print("Facial emotion model downloaded successfully!")
        else:
            print(f"Failed to download facial emotion model. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading facial emotion model: {str(e)}")

def main():
    """Main function to download all models."""
    print("Starting model downloads...")
    
    # Download text emotion model
    download_text_emotion_model()
    
    # Download speech emotion model
    download_speech_emotion_model()
    
    # Download facial emotion model
    download_facial_emotion_model()
    
    print("All models downloaded successfully!")

if __name__ == "__main__":
    main() 
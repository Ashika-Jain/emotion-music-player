import numpy as np
import cv2
import os
import logging
import traceback
from deepface import DeepFace
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Get the base directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger.debug(f"Base directory: {base_dir}")

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Define emotion mapping to music genres with confidence thresholds
emotion_config = {
    'angry': {'genre': 'metal', 'threshold': 0.35},  # Maps to anger/frustrated
    'disgust': {'genre': 'blues', 'threshold': 0.35},  # Maps directly
    'fear': {'genre': 'sad', 'threshold': 0.35},  # Maps directly
    'happy': {'genre': 'happy', 'threshold': 0.35},  # Maps to happy/joy
    'sad': {'genre': 'sad', 'threshold': 0.35},  # Maps directly
    'surprise': {'genre': 'party', 'threshold': 0.35},  # Maps to surprised/excited
    'neutral': {'genre': 'pop', 'threshold': 0.30}  # Maps directly
}

# Additional mood mappings for reference (these come from facial expressions)
extended_mood_map = {
    # Primary emotions (from model)
    'angry': 'metal',
    'disgust': 'blues',
    'fear': 'sad',
    'happy': 'happy',
    'sad': 'sad',
    'surprise': 'party',
    'neutral': 'pop',
    
    # Extended emotions (for reference)
    'joy': 'hip-hop',
    'love': 'romance',
    'calm': 'chill',
    'excited': 'party',
    'bored': 'pop',
    'tired': 'chill',
    'relaxed': 'chill',
    'stressed': 'chill',
    'anxious': 'chill',
    'depressed': 'sad',
    'lonely': 'sad',
    'energetic': 'hip-hop',
    'nostalgic': 'pop',
    'confused': 'pop',
    'frustrated': 'metal',
    'hopeful': 'romance',
    'proud': 'hip-hop',
    'guilty': 'blues',
    'jealous': 'pop',
    'ashamed': 'blues',
    'disappointed': 'pop',
    'content': 'chill',
    'insecure': 'pop',
    'embarrassed': 'blues',
    'overwhelmed': 'chill',
    'amused': 'party'
}

def infer_facial_emotion(image_path):
    """
    Detect emotion from a facial image using DeepFace.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Detected emotion mapped to a music genre
    """
    try:
        logger.debug(f"Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return None
            
        try:
            # Analyze the image using DeepFace
            result = DeepFace.analyze(
                img_path=image_path,
                actions=['emotion'],
                enforce_detection=False,  # Don't raise error if face not detected
                detector_backend='opencv'  # Use OpenCV for faster detection
            )
            
            # Extract emotion data
            if isinstance(result, list):
                emotion_data = result[0]['emotion']
            else:
                emotion_data = result['emotion']
                
            logger.debug(f"DeepFace emotion analysis: {emotion_data}")
            
            # Find the emotion with highest probability
            dominant_emotion = max(emotion_data.items(), key=lambda x: x[1])[0].lower()
            
            # Map the emotion to our defined emotions
            if dominant_emotion not in emotion_config:
                logger.warning(f"Unmapped emotion detected: {dominant_emotion}, defaulting to neutral")
                dominant_emotion = 'neutral'
            
            # Map emotion to genre
            genre = emotion_config[dominant_emotion]['genre']
            logger.debug(f"Mapped emotion {dominant_emotion} to genre: {genre}")
            
            return genre
            
        except Exception as e:
            logger.warning(f"Error in DeepFace analysis: {str(e)}, falling back to neutral")
            return 'neutral'  # Default to neutral if analysis fails
        
    except Exception as e:
        logger.error(f"Error in facial emotion detection: {str(e)}")
        logger.error(traceback.format_exc())
        return None

import os
from models.facial_emotion import infer_facial_emotion
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_emotion_detection():
    # Get the absolute path to the test image
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_image_path = os.path.join(current_dir, '..', 'test_images', 'cry.png')
    
    logger.info(f"Testing emotion detection with image: {test_image_path}")
    
    # Run emotion detection
    detected_genre = infer_facial_emotion(test_image_path)
    
    logger.info(f"Detected genre: {detected_genre}")
    
    return detected_genre

if __name__ == "__main__":
    test_emotion_detection() 
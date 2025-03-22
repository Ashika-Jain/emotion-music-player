import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define the model directory path
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'text_emotion_model')

def analyze_context(text):
    """
    Analyze the context of the text to better understand the emotion.
    """
    text = text.lower()
    
    # Positive context indicators (expanded)
    positive_indicators = [
        'good', 'great', 'wonderful', 'amazing', 'fantastic', 'excellent',
        'love', 'enjoy', 'like', 'beautiful', 'perfect', 'fun', 'excited',
        'happy', 'joy', 'joyful', 'glad', 'delighted', 'cheerful', 'pleased',
        'content', 'satisfied', 'thrilled', 'ecstatic', 'elated', 'jubilant',
        'smile', 'laughing', 'grinning', 'beaming', 'bright', 'sunny', 'upbeat',
        'positive', 'optimistic', 'hopeful', 'enthusiastic', 'eager', 'keen',
        'grateful', 'thankful', 'appreciative', 'blessed', 'fortunate', 'lucky'
    ]
    
    # Negative context indicators (expanded)
    negative_indicators = [
        'bad', 'terrible', 'horrible', 'awful', 'worst', 'hate',
        'sad', 'unhappy', 'depressed', 'down', 'miserable', 'gloomy',
        'angry', 'mad', 'furious', 'annoyed', 'irritated', 'upset',
        'disappointed', 'frustrated', 'bitter', 'resentful', 'hostile',
        'enraged', 'outraged', 'infuriated', 'livid', 'seething',
        'displeased', 'dissatisfied', 'discontented', 'disgruntled',
        'heartbroken', 'devastated', 'crushed', 'hurt', 'pained',
        'suffering', 'agonizing', 'distressed', 'troubled', 'worried',
        'anxious', 'nervous', 'tense', 'stressed', 'overwhelmed',
        'fearful', 'afraid', 'scared', 'terrified', 'frightened',
        'lonely', 'isolated', 'abandoned', 'rejected', 'unwanted'
    ]
    
    # Count positive and negative indicators with word boundary checks
    positive_count = 0
    negative_count = 0
    
    for word in re.findall(r'\b\w+\b', text):
        if word in positive_indicators:
            positive_count += 1
        elif word in negative_indicators:
            negative_count += 1
    
    # Check for negation patterns that reverse sentiment
    negation_words = ['not', "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", 
                      "haven't", "hasn't", "hadn't", "won't", "wouldn't", "can't", "couldn't", 
                      "shouldn't", "mustn't", "never", "no", "none", "nothing", "nowhere"]
    
    # Simple negation detection
    for neg in negation_words:
        neg_pattern = r'\b' + re.escape(neg) + r'\b\s+\w+\s+(\w+)'
        matches = re.findall(neg_pattern, text)
        for match in matches:
            if match in positive_indicators:
                positive_count -= 1
                negative_count += 1
            elif match in negative_indicators:
                negative_count -= 1
                positive_count += 1
    
    return positive_count, negative_count

def preprocess_text(text):
    """
    Preprocess the text to extract key emotional indicators.
    """
    logger.debug(f"Preprocessing text: {text}")
    
    # Convert to lowercase
    text = text.lower()
    
    # Basic emotion keyword mapping with context
    emotion_keywords = {
        'happy': ['happy', 'joy', 'joyful', 'glad', 'delighted', 'excited', 'cheerful', 'great', 'wonderful', 'amazing',
                 'pleased', 'content', 'satisfied', 'thrilled', 'ecstatic', 'elated', 'jubilant', 'smile', 'laughing',
                 'grinning', 'beaming', 'bright', 'sunny', 'upbeat', 'positive', 'optimistic', 'hopeful', 'enthusiastic'],
        'sadness': ['sad', 'unhappy', 'depressed', 'down', 'miserable', 'gloomy', 'terrible', 'horrible', 'disappointed',
                   'heartbroken', 'devastated', 'crushed', 'hurt', 'pained', 'suffering', 'agonizing', 'distressed',
                   'troubled', 'lonely', 'isolated', 'abandoned', 'rejected', 'unwanted'],
        'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'hate', 'rage', 'upset', 'frustrated', 'bitter',
                 'resentful', 'hostile', 'enraged', 'outraged', 'infuriated', 'livid', 'seething', 'displeased'],
        'love': ['love', 'loving', 'lovely', 'romantic', 'passionate', 'adore', 'cherish', 'affection', 'fond',
               'devoted', 'attachment', 'enamored', 'infatuated', 'smitten', 'enchanted', 'captivated'],
        'fear': ['fear', 'afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous', 'tense', 'stressed',
               'overwhelmed', 'fearful', 'frightened', 'panicked', 'alarmed', 'horrified', 'petrified'],
        'neutral': ['okay', 'fine', 'alright', 'normal', 'average', 'moderate', 'standard', 'regular', 'common',
                  'ordinary', 'typical', 'usual', 'routine', 'everyday', 'plain', 'simple']
    }
    
    # Check for direct emotion mentions with word boundary checks
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text):
                logger.debug(f"Found direct emotion keyword: {keyword} -> {emotion}")
                return emotion
    
    # Check for common phrases
    phrases = {
        'happy': ['feeling good', 'in a good mood', 'having a great day', 'feeling great', 'doing well', 
                 'feeling better', 'on cloud nine', 'over the moon', 'on top of the world'],
        'sadness': ['feeling down', 'feeling blue', 'having a bad day', 'not feeling well', 'under the weather',
                   'feeling low', 'in a bad mood', 'feeling sad', 'heart is heavy', 'down in the dumps'],
        'anger': ['pissed off', 'fed up', 'had enough', 'lost my temper', 'makes me mad', 'drives me crazy',
                 'getting on my nerves', 'ticks me off', 'makes my blood boil'],
        'love': ['in love', 'falling for', 'head over heels', 'deeply attached', 'care about', 'fond of'],
        'fear': ['scared of', 'afraid of', 'worried about', 'concerned about', 'nervous about', 
               'anxious about', 'dreading', 'terrified of', 'frightened by']
    }
    
    for emotion, phrase_list in phrases.items():
        for phrase in phrase_list:
            if phrase in text:
                logger.debug(f"Found emotion phrase: {phrase} -> {emotion}")
                return emotion
    
    # Analyze context if no direct emotion is found
    positive_count, negative_count = analyze_context(text)
    logger.debug(f"Context analysis - positive: {positive_count}, negative: {negative_count}")
    
    # Make decision based on context analysis
    if positive_count > negative_count:
        if positive_count >= 2:  # Stronger positive sentiment
            return 'happy'
        return 'happy'  # Default positive emotion
    elif negative_count > positive_count:
        if negative_count >= 2:  # Stronger negative sentiment
            if any(word in text for word in emotion_keywords['anger']):
                return 'anger'
            elif any(word in text for word in emotion_keywords['fear']):
                return 'fear'
            return 'sadness'  # Default negative emotion
        return 'sadness'  # Default mild negative emotion
    
    return 'neutral'

def infer_text_emotion(text):
    """
    Infer the emotion from the given text using the trained text emotion model.

    :param text: The input text.
    :return: The detected emotion.
    """
    try:
        logger.debug(f"Starting emotion inference for text: {text}")
        
        # First try direct emotion detection
        direct_emotion = preprocess_text(text)
        if direct_emotion:
            logger.debug(f"Direct emotion detected: {direct_emotion}")
            return direct_emotion
            
        # If no direct emotion found, use the model
        logger.debug("No direct emotion found, using model")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        logger.debug("Text tokenized successfully")

        with torch.no_grad():
            outputs = model(**inputs)

        scores = outputs[0][0].numpy()
        emotion_idx = scores.argmax()
        logger.debug(f"Model prediction scores: {scores}")

        # Map model emotions to music recommendation emotions
        emotion_labels = ["sadness", "joy", "love", "anger", "fear"]
        base_emotion = emotion_labels[emotion_idx]
        logger.debug(f"Base emotion detected: {base_emotion}")
        
        # Map 'joy' to 'happy' for consistency
        if base_emotion == "joy":
            logger.debug("Mapping 'joy' to 'happy'")
            return "happy"
            
        return base_emotion
    except Exception as e:
        logger.error(f"Error in text emotion inference: {str(e)}", exc_info=True)
        return "neutral"  # Default fallback emotion

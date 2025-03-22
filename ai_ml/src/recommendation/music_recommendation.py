import requests
import random
import logging
import traceback
import sys
import os

# Add the src directory to the Python path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from utils import get_spotify_access_token

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_music_recommendation(emotion, market=None):
    """
    Get music recommendations based on the detected emotion.

    Args:
        emotion (str): The detected emotion.
        market (str, optional): The market for which the recommendations are to be provided.
        
    Returns:
        list: A list of recommended tracks.
    """
    if not emotion:
        logger.warning("No emotion provided, defaulting to 'neutral'")
        emotion = "neutral"
        
    logger.debug(f"Getting music recommendations for emotion: {emotion}")
    
    try:
        # Normalize emotion to lowercase
        emotion = emotion.lower()
        
        # Get Spotify access token
        try:
            access_token = get_spotify_access_token()
            if not access_token:
                logger.error("Failed to retrieve Spotify access token")
                return []
            logger.debug("Successfully retrieved Spotify access token")
        except Exception as e:
            logger.error(f"Error retrieving Spotify access token: {str(e)}")
            logger.error(traceback.format_exc())
            return []

        # Map emotions to keywords for the search query
        emotion_to_keyword = {
            "joy": "joy",
            "happy": "happy",
            "sadness": "sad",
            "sad": "sad",
            "anger": "angry",
            "angry": "angry",
            "love": "romantic",
            "fear": "calm",
            "neutral": "chill",
            "calm": "peaceful",
            "disgust": "blues",
            "surprised": "party",
            "surprise": "party",
            "excited": "energetic",
            "bored": "relaxing",
            "tired": "calm",
            "relaxed": "calm",
            "stressed": "calm",
            "anxious": "calm",
            "depressed": "sad",
            "lonely": "sad",
            "energetic": "upbeat",
            "nostalgic": "retro",
            "confused": "instrumental",
            "frustrated": "aggressive",
            "hopeful": "uplifting",
            "proud": "epic",
            "guilty": "melancholic",
            "jealous": "dark",
            "ashamed": "melancholic",
            "disappointed": "sad",
            "content": "chill",
            "insecure": "soulful",
            "embarassed": "blues",
            "overwhelmed": "ambient",
            "amused": "fun",
            "metal": "metal",  # Direct genre mapping
            "rock": "rock",    # Direct genre mapping
            "pop": "pop",      # Direct genre mapping
            "dark": "dark",    # Direct genre mapping
            "chill": "chill"   # Direct genre mapping
        }

        # Full list of available markets
        available_markets = [
            "AD", "AE", "AG", "AL", "AM", "AO", "AR", "AT", "AU", "AZ", "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI",
            "BJ", "BN", "BO", "BR", "BS", "BT", "BW", "BY", "BZ", "CA", "CD", "CG", "CH", "CI", "CL", "CM", "CO", "CR",
            "CV", "CW", "CY", "CZ", "DE", "DJ", "DK", "DM", "DO", "DZ", "EC", "EE", "EG", "ES", "ET", "FI", "FJ", "FM",
            "FR", "GA", "GB", "GD", "GE", "GH", "GM", "GN", "GQ", "GR", "GT", "GW", "GY", "HK", "HN", "HR", "HT", "HU",
            "ID", "IE", "IL", "IN", "IQ", "IS", "IT", "JM", "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN", "KR", "KW",
            "KZ", "LA", "LB", "LC", "LI", "LK", "LR", "LS", "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MG", "MH",
            "MK", "ML", "MN", "MO", "MR", "MT", "MU", "MV", "MW", "MX", "MY", "MZ", "NA", "NE", "NG", "NI", "NL", "NO",
            "NP", "NR", "NZ", "OM", "PA", "PE", "PG", "PH", "PK", "PL", "PR", "PS", "PT", "PW", "PY", "QA", "RO", "RS",
            "RW", "SA", "SB", "SC", "SE", "SG", "SI", "SK", "SL", "SM", "SN", "SR", "ST", "SV", "SZ", "TD", "TG", "TH",
            "TJ", "TL", "TN", "TO", "TR", "TT", "TV", "TW", "TZ", "UA", "UG", "US", "UY", "UZ", "VC", "VE", "VN", "VU",
            "WS", "XK", "ZA", "ZM", "ZW"
        ]

        # Determine the keyword for the given emotion
        keyword = emotion_to_keyword.get(emotion, "pop")  # Default to "pop" if emotion isn't recognized
        logger.debug(f"Mapped emotion '{emotion}' to keyword '{keyword}'")

        # Select market randomly if not provided or invalid
        if market and market in available_markets:
            selected_market = market
        else:
            selected_market = random.choice(available_markets)
        logger.debug(f"Selected market: {selected_market}")

        headers = {
            "Authorization": f"Bearer {access_token}"
        }

        # Prepare query parameters
        params = {
            "q": keyword,
            "type": "track",
            "limit": 10,
            "market": selected_market
        }

        logger.debug(f"Parameters sent to Spotify API: {params}")

        # Make the request to Spotify Search API
        try:
            response = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params)
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to Spotify API: {str(e)}")
            return []

        # Handle response statuses
        if response.status_code == 401:
            logger.error("Access token expired. Please refresh the token.")
            return []
        elif response.status_code != 200:
            logger.error(f"Failed to fetch music recommendations. Status code: {response.status_code}")
            return []

        # Parse the response
        try:
            data = response.json()
        except ValueError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            return []

        # Extract tracks from the response
        tracks = data.get("tracks", {}).get("items", [])
        logger.debug(f"Retrieved {len(tracks)} tracks from Spotify API")

        # Format the tracks for the response
        recommendations = []
        for track in tracks:
            try:
                artists = ", ".join([artist["name"] for artist in track.get("artists", [])])
                track_info = {
                    "id": track.get("id", ""),
                    "name": track.get("name", "Unknown Track"),
                    "artist": artists,
                    "album": track.get("album", {}).get("name", "Unknown Album"),
                    "preview_url": track.get("preview_url", ""),
                    "external_url": track.get("external_urls", {}).get("spotify", ""),
                    "image_url": track.get("album", {}).get("images", [{}])[0].get("url", "") if track.get("album", {}).get("images", []) else ""
                }
                recommendations.append(track_info)
            except Exception as e:
                logger.error(f"Error formatting track: {str(e)}")
                continue

        logger.debug(f"Returning {len(recommendations)} formatted track recommendations")
        return recommendations

    except Exception as e:
        logger.error(f"Unexpected error in get_music_recommendation: {str(e)}")
        logger.error(traceback.format_exc())
        return []

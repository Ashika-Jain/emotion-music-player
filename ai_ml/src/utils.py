import base64
import requests
import logging
import traceback
from config import CONFIG

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_spotify_access_token():
    """
    Retrieve the Spotify access token using the client ID and client secret.

    Returns:
        str: The Spotify access token or None if retrieval fails.
    """
    try:
        client_id = CONFIG.get("spotify_client_id")
        client_secret = CONFIG.get("spotify_client_secret")
        
        if not client_id or not client_secret:
            logger.error("Spotify client ID or client secret is missing in the configuration")
            return None
            
        logger.debug("Retrieving Spotify access token")
        token_url = "https://accounts.spotify.com/api/token"

        # Compute the Base64-encoded string
        try:
            encoded_credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
            logger.debug("Successfully encoded client credentials")
        except Exception as e:
            logger.error(f"Error encoding client credentials: {str(e)}")
            logger.error(traceback.format_exc())
            return None

        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        data = {
            "grant_type": "client_credentials"
        }

        try:
            logger.debug("Making request to Spotify token endpoint")
            response = requests.post(token_url, headers=headers, data=data, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"Failed to retrieve Spotify access token. Status code: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
                
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
            logger.debug("Successfully received response from Spotify token endpoint")
        except requests.exceptions.Timeout:
            logger.error("Request to Spotify token endpoint timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve Spotify access token: {str(e)}")
            logger.error(traceback.format_exc())
            return None

        # Extract and return the access token
        try:
            response_data = response.json()
            token = response_data.get("access_token")
            
            if not token:
                logger.error("Access token not found in response")
                logger.error(f"Response data: {response_data}")
                return None
                
            logger.debug("Successfully retrieved Spotify access token")
            return token
        except ValueError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            logger.error(traceback.format_exc())
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error in get_spotify_access_token: {str(e)}")
        logger.error(traceback.format_exc())
        return None

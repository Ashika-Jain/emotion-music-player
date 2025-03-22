from mongoengine import Document, StringField, ListField, DictField, DateTimeField, BooleanField, connect
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class UserProfile(Document):
    """
    This class defines the structure of the user profile document in the MongoDB database.
    """
    username = StringField(required=True, unique=True)
    mood_history = ListField(DictField(), default=list)  # List of {emotion: str, timestamp: datetime}
    listening_history = ListField(DictField(), default=list)  # List of {track_id: str, track_name: str, artist: str, timestamp: datetime}
    recommendations = ListField(DictField(), default=list)  # List of {track_id: str, track_name: str, artist: str, emotion: str}
    is_active = BooleanField(default=True)
    created_at = DateTimeField(default=datetime.utcnow)
    last_login = DateTimeField(default=datetime.utcnow)

    meta = {
        'collection': 'user_profiles',
        'db_alias': 'default',
        'indexes': [
            {'fields': ['username'], 'unique': True}
        ],
        'auto_create_index': True,
        'strict': False
    }

    def save(self, *args, **kwargs):
        try:
            logger.debug(f"Saving UserProfile for {self.username}")
            if not self.created_at:
                self.created_at = datetime.utcnow()
            self.last_login = datetime.utcnow()
            
            # Ensure we're using force_insert for new documents
            if not self.id and 'force_insert' not in kwargs:
                kwargs['force_insert'] = True
                
            result = super(UserProfile, self).save(*args, **kwargs)
            logger.debug(f"Successfully saved UserProfile for {self.username}")
            return result
        except Exception as e:
            logger.error(f"Error saving UserProfile for {self.username}: {str(e)}")
            raise

    def __str__(self):
        return f"UserProfile(username={self.username}, created_at={self.created_at})"

    def add_mood(self, emotion):
        """Add a mood entry to the user's mood history."""
        try:
            logger.debug(f"Adding mood {emotion} for user {self.username}")
            self.mood_history.append({
                'emotion': emotion,
                'timestamp': datetime.utcnow()
            })
            self.save()
            logger.debug(f"Successfully added mood {emotion} for user {self.username}")
        except Exception as e:
            logger.error(f"Error adding mood for {self.username}: {str(e)}")
            raise

    def update_mood_history(self, emotion):
        """Update the user's mood history with a new emotion."""
        try:
            logger.debug(f"Updating mood history with {emotion} for user {self.username}")
            self.mood_history.append({
                'emotion': emotion,
                'timestamp': datetime.utcnow()
            })
            self.save()
            logger.debug(f"Successfully updated mood history for user {self.username}")
        except Exception as e:
            logger.error(f"Error updating mood history for {self.username}: {str(e)}")
            raise

    def add_listening(self, track_id, track_name, artist):
        """Add a track to the user's listening history."""
        try:
            logger.debug(f"Adding track {track_name} to listening history for user {self.username}")
            self.listening_history.append({
                'track_id': track_id,
                'track_name': track_name,
                'artist': artist,
                'timestamp': datetime.utcnow()
            })
            self.save()
            logger.debug(f"Successfully added track to listening history for user {self.username}")
        except Exception as e:
            logger.error(f"Error adding track to listening history for {self.username}: {str(e)}")
            raise

    def add_recommendation(self, track_id, track_name, artist, emotion):
        """Add a track recommendation."""
        try:
            logger.debug(f"Adding recommendation {track_name} for emotion {emotion} for user {self.username}")
            self.recommendations.append({
                'track_id': track_id,
                'track_name': track_name,
                'artist': artist,
                'emotion': emotion
            })
            self.save()
            logger.debug(f"Successfully added recommendation for user {self.username}")
        except Exception as e:
            logger.error(f"Error adding recommendation for {self.username}: {str(e)}")
            raise

    def get_recent_moods(self, limit=5):
        """Get the user's most recent moods."""
        return sorted(self.mood_history, key=lambda x: x['timestamp'], reverse=True)[:limit]

    def get_recent_tracks(self, limit=5):
        """Get the user's most recently played tracks."""
        return sorted(self.listening_history, key=lambda x: x['timestamp'], reverse=True)[:limit]

    def get_recommendations_by_emotion(self, emotion):
        """Get recommendations for a specific emotion."""
        return [rec for rec in self.recommendations if rec['emotion'] == emotion]

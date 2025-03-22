import os
import sys

# Get the absolute path to the ai_ml directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')

# Add the src directory to the Python path
sys.path.insert(0, src_dir)

# Change to the src directory
os.chdir(src_dir)

# Import Flask app and config
from api.emotion_api import app
from config import CONFIG

if __name__ == "__main__":
    # Disable Flask reloader to prevent the file not found error
    app.run(host="0.0.0.0", port=CONFIG["api_port"], debug=True, use_reloader=False) 
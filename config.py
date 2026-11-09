import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

class Config:
    # Flask application settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'identityshield_default_secret_key_2026')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    DEBUG = FLASK_ENV == 'development'
    
    # MongoDB Settings
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/identityshield')
    
    # Upload settings
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', os.path.join(os.getcwd(), 'uploads', 'images'))
    # Ensure upload folder exists when importing config
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Max size allowed for uploaded files (50 MB for video support)
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 50 * 1024 * 1024))
    
    # Allowed media formats for deepfake detection
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'wav', 'mp3', 'flac', 'ogg', 'mp4', 'avi', 'mov', 'webm'}

    # Pre-trained Model Paths
    MODEL_WEIGHTS_PATH = os.path.join(os.getcwd(), 'model', 'DeepfakeWeights.h5')
    URL_MODEL_PATH = os.path.join(os.getcwd(), 'model', 'url_model.pkl')
    URL_VECTORIZER_PATH = os.path.join(os.getcwd(), 'model', 'vectorizer.pkl')

    # SMTP Configuration
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME', '')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', '')

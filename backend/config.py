import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Base configuration"""
    # Flask settings
    DEBUG = True
    TESTING = False
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-please-change-in-production')
    
    # Spotify API settings
    SPOTIFY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', "d7eb111127ba40d083314a2c429caa74")
    SPOTIFY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET', "1deb80662ad24c9186ea557b2d901380")
    SPOTIFY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI', "http://127.0.0.1:3000")
    
    # ML model paths
    ML_SERVICE_PATH = os.getenv('ML_SERVICE_PATH', 
                               r"C:\Project 02\Mood-and-context-based-Spotify-playlist-creation-system\ml service")


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # In production, set this from environment variables
    SECRET_KEY = os.getenv('SECRET_KEY')


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True


# Configuration dictionary
config_by_name = dict(
    dev=DevelopmentConfig,
    test=TestingConfig,
    prod=ProductionConfig
)

# Get configuration from environment or use default
config_name = os.getenv('FLASK_ENV', 'dev')
if config_name not in config_by_name:
    config_name = 'dev'
    
# Active config
Config = config_by_name[config_name]
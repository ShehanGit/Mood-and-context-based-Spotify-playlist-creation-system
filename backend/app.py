from flask import Flask
from flask_cors import CORS
import os

# Import configuration
from config import Config

# Import blueprints
from routes.recommendation_routes import recommendation_bp
from routes.spotify_routes import spotify_bp

def create_app(config=Config):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Enhanced CORS configuration to fix CORS errors
    CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
    
    # Register blueprints
    app.register_blueprint(recommendation_bp)
    app.register_blueprint(spotify_bp)
    
    # Add special CORS headers to all responses
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        return response
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    # Fixed the DEBUG attribute access
    app.run(host='0.0.0.0', port=port, debug=Config.DEBUG)
from flask import Blueprint, request, jsonify, redirect
from services.spotify_service import SpotifyService
from services.recommendation_service import RecommendationService

# Create Blueprint
spotify_bp = Blueprint('spotify', __name__, url_prefix='/api/spotify')

# Initialize services
spotify_service = SpotifyService()
recommendation_service = RecommendationService()

@spotify_bp.route('/login', methods=['GET'])
def login():
    """Redirect to Spotify authorization page"""
    auth_url = spotify_service.get_auth_url()
    return jsonify({"success": True, "auth_url": auth_url})

@spotify_bp.route('/callback', methods=['GET'])
def callback():
    """Handle Spotify OAuth callback"""
    # This would be used if you have a web application frontend
    # In a real application, you would handle the callback and redirect to your frontend
    code = request.args.get('code')
    if not code:
        return jsonify({"success": False, "error": "No authorization code provided"}), 400
    
    # Get token and redirect to frontend
    token_info = spotify_service.get_user_token(code)
    if not token_info:
        return jsonify({"success": False, "error": "Failed to get access token"}), 500
    
    # In a real application, you would send this token to the frontend or store it in a session
    return jsonify({"success": True, "token_info": token_info})

@spotify_bp.route('/token', methods=['POST'])
def get_token():
    """Exchange authorization code for access token"""
    data = request.get_json()
    if not data or 'code' not in data:
        return jsonify({"success": False, "error": "No authorization code provided"}), 400
    
    token_info = spotify_service.get_user_token(data['code'])
    if not token_info:
        return jsonify({"success": False, "error": "Failed to get access token"}), 500
    
    return jsonify({"success": True, "token_info": token_info})

@spotify_bp.route('/profile', methods=['GET'])
def get_profile():
    """Get the current user's Spotify profile"""
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({"success": False, "error": "No authorization header provided"}), 401
    
    token = auth_header.split(' ')[1] if len(auth_header.split(' ')) > 1 else None
    if not token:
        return jsonify({"success": False, "error": "Invalid authorization header"}), 401
    
    profile = spotify_service.get_user_profile(token)
    if not profile:
        return jsonify({"success": False, "error": "Failed to get user profile"}), 500
    
    return jsonify({"success": True, "profile": profile})

@spotify_bp.route('/playlists', methods=['POST'])
def create_playlist():
    """Create a new playlist"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400
    
    # Get authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({"success": False, "error": "No authorization header provided"}), 401
    
    token = auth_header.split(' ')[1] if len(auth_header.split(' ')) > 1 else None
    if not token:
        return jsonify({"success": False, "error": "Invalid authorization header"}), 401
    
    # Get required fields
    user_id = data.get('user_id')
    name = data.get('name')
    description = data.get('description', '')
    
    if not user_id or not name:
        return jsonify({"success": False, "error": "Missing required fields (user_id, name)"}), 400
    
    # Create playlist
    playlist = spotify_service.create_playlist(token, user_id, name, description)
    if not playlist:
        return jsonify({"success": False, "error": "Failed to create playlist"}), 500
    
    return jsonify({"success": True, "playlist": playlist})

@spotify_bp.route('/playlists/<playlist_id>/tracks', methods=['POST'])
def add_tracks_to_playlist(playlist_id):
    """Add tracks to a playlist"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400
    
    # Get authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({"success": False, "error": "No authorization header provided"}), 401
    
    token = auth_header.split(' ')[1] if len(auth_header.split(' ')) > 1 else None
    if not token:
        return jsonify({"success": False, "error": "Invalid authorization header"}), 401
    
    # Get track IDs
    track_ids = data.get('track_ids', [])
    if not track_ids:
        return jsonify({"success": False, "error": "No track IDs provided"}), 400
    
    # Add tracks to playlist
    success = spotify_service.add_tracks_to_playlist(token, playlist_id, track_ids)
    if not success:
        return jsonify({"success": False, "error": "Failed to add tracks to playlist"}), 500
    
    return jsonify({"success": True, "message": f"Added {len(track_ids)} tracks to playlist"})

@spotify_bp.route('/recommendations/playlist', methods=['POST'])
def create_recommendations_playlist():
    """Create a playlist with recommended songs"""
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400
    
    # Get authorization header
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return jsonify({"success": False, "error": "No authorization header provided"}), 401
    
    token = auth_header.split(' ')[1] if len(auth_header.split(' ')) > 1 else None
    if not token:
        return jsonify({"success": False, "error": "Invalid authorization header"}), 401
    
    # Get user profile
    profile = spotify_service.get_user_profile(token)
    if not profile:
        return jsonify({"success": False, "error": "Failed to get user profile"}), 500
    
    user_id = profile['id']
    
    # Get preferences and playlist details
    preferences = data.get('preferences', {})
    playlist_name = data.get('playlist_name', 'Recommended Songs')
    playlist_description = data.get('playlist_description', 'Generated by Music Recommendation System')
    num_recommendations = data.get('num_recommendations', 50)
    
    # Get recommendations
    recommendations, error = recommendation_service.recommend_songs(preferences, num_recommendations)
    if error:
        return jsonify({"success": False, "error": f"Failed to get recommendations: {error}"}), 500
    
    # Extract track IDs from recommendations
    track_ids = []
    for track in recommendations:
        # Try different possible field names for the track ID
        for field in ['id', 'track_id', 'spotify_id']:
            if field in track and track[field]:
                track_ids.append(track[field])
                break
    
    if not track_ids:
        return jsonify({"success": False, "error": "No valid track IDs found in recommendations"}), 500
    
    # Create playlist
    playlist = spotify_service.create_playlist(token, user_id, playlist_name, playlist_description)
    if not playlist:
        return jsonify({"success": False, "error": "Failed to create playlist"}), 500
    
    # Add tracks to playlist
    success = spotify_service.add_tracks_to_playlist(token, playlist['id'], track_ids)
    if not success:
        return jsonify({
            "success": False, 
            "error": "Created playlist but failed to add tracks", 
            "playlist": playlist
        }), 500
    
    return jsonify({
        "success": True, 
        "message": f"Created playlist with {len(track_ids)} recommended tracks",
        "playlist": playlist
    })
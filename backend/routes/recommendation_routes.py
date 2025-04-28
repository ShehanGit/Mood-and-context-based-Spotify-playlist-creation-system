from flask import Blueprint, request, jsonify
from services.recommendation_service import RecommendationService

# Create Blueprint
recommendation_bp = Blueprint('recommendation', __name__, url_prefix='/api')

# Initialize the recommendation service
recommendation_service = RecommendationService()

@recommendation_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Music recommendation API is running"}), 200

@recommendation_bp.route('/dataset/columns', methods=['GET'])
def get_dataset_columns():
    """Get all columns in the dataset (for debugging)"""
    columns = recommendation_service.get_dataset_columns()
    
    if columns:
        return jsonify({
            "success": True,
            "columns": columns
        })
    else:
        return jsonify({
            "success": False,
            "error": "Failed to retrieve dataset columns"
        }), 500

@recommendation_bp.route('/categories', methods=['GET'])
def get_categories():
    """Get all possible values for categorical features"""
    categories = recommendation_service.get_categories()
    
    if categories:
        return jsonify({
            "success": True,
            "categories": categories
        })
    else:
        return jsonify({
            "success": False,
            "error": "Failed to retrieve categories"
        }), 500

@recommendation_bp.route('/recommend', methods=['POST'])
def recommend_songs():
    """Recommend songs based on user preferences"""
    # Get JSON data from request
    data = request.get_json()
    
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400
    
    # Extract user preferences
    user_preferences = data.get("preferences", {})
    num_recommendations = data.get("num_recommendations", 50)
    
    # Get recommendations
    recommendations, error = recommendation_service.recommend_songs(
        user_preferences, 
        num_recommendations
    )
    
    if error:
        return jsonify({
            "success": False,
            "error": error
        }), 500
    
    # Return recommendations
    return jsonify({
        "success": True,
        "count": len(recommendations),
        "recommendations": recommendations
    })

@recommendation_bp.route('/song/<song_id>', methods=['GET'])
def get_song_details(song_id):
    """Get detailed information about a specific song"""
    song_details, error = recommendation_service.get_song_details(song_id)
    
    if error:
        return jsonify({
            "success": False,
            "error": error
        }), 404 if error == "Song not found" else 500
    
    return jsonify({
        "success": True,
        "song": song_details
    })
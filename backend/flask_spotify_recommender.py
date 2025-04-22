from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the path to ML service directory
ML_SERVICE_PATH = r"C:\Project 02\Mood-and-context-based-Spotify-playlist-creation-system\ml service"

# Global variables to store loaded models and data
MODEL = None
SCALER = None
ENCODERS = {}
DATASET = None

def load_resources():
    """Load all necessary resources for prediction"""
    global MODEL, SCALER, ENCODERS, DATASET
    
    try:
        # Load model from the correct directory
        model_path = os.path.join(ML_SERVICE_PATH, "improved_xgb_model.pkl")
        print(f"Loading model from: {model_path}")
        MODEL = joblib.load(model_path)
        
        # Load scaler
        scaler_path = os.path.join(ML_SERVICE_PATH, "numerical_scaler.pkl")
        print(f"Loading scaler from: {scaler_path}")
        SCALER = joblib.load(scaler_path)
        
        # Load encoders
        categorical_features = ["mood", "age_group", "activity", "weather"]
        for col in categorical_features:
            encoder_path = os.path.join(ML_SERVICE_PATH, f"{col}_encoder.pkl")
            print(f"Loading encoder for {col} from: {encoder_path}")
            ENCODERS[col] = joblib.load(encoder_path)
        
        # Load dataset
        dataset_path = os.path.join(ML_SERVICE_PATH, "enhanced_dataset_improved.csv")
        print(f"Loading dataset from: {dataset_path}")
        DATASET = pd.read_csv(dataset_path)
        
        # Print the columns to help with debugging
        print("Available columns in dataset:", DATASET.columns.tolist())
        
        print("All resources loaded successfully")
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Music recommendation API is running"}), 200

@app.route('/api/dataset/columns', methods=['GET'])
def get_dataset_columns():
    """Get all columns in the dataset (for debugging)"""
    global DATASET
    
    if DATASET is None:
        load_resources()
    
    return jsonify({
        "success": True,
        "columns": DATASET.columns.tolist()
    })

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all possible values for categorical features"""
    global DATASET
    
    if DATASET is None:
        load_resources()
    
    categorical_features = ["mood", "age_group", "activity", "weather"]
    categories = {}
    
    for feature in categorical_features:
        categories[feature] = DATASET[feature].unique().tolist()
    
    return jsonify({
        "success": True,
        "categories": categories
    })

@app.route('/api/recommend', methods=['POST'])
def recommend_songs():
    """Recommend songs based on user preferences"""
    global MODEL, SCALER, ENCODERS, DATASET
    
    # Load resources if not already loaded
    if MODEL is None or SCALER is None or not ENCODERS or DATASET is None:
        load_resources()
    
    # Get JSON data from request
    data = request.get_json()
    
    if not data:
        return jsonify({"success": False, "error": "No data provided"}), 400
    
    try:
        # Extract user preferences
        user_preferences = data.get("preferences", {})
        num_recommendations = data.get("num_recommendations", 50)
        
        # Define expected features
        numerical_features = ["valence", "energy", "danceability", "tempo", 
                             "acousticness", "instrumentalness", "popularity", "temperature"]
        categorical_features = ["mood", "age_group", "activity", "weather"]
        
        # Check if required parameters are present
        for feature in numerical_features + categorical_features:
            if feature not in user_preferences:
                return jsonify({
                    "success": False, 
                    "error": f"Missing required parameter: {feature}"
                }), 400
        
        # Create input dataframe
        input_data = pd.DataFrame([user_preferences])
        
        # Scale numerical features
        input_numerical = input_data[numerical_features].values.reshape(1, -1)
        input_data[numerical_features] = SCALER.transform(input_numerical)
        
        # Encode categorical features
        for col in categorical_features:
            encoder = ENCODERS[col]
            # Try to encode the value
            try:
                input_data[col] = encoder.transform([user_preferences[col]])
            except ValueError:
                # If category not in training data, use most common
                most_common = DATASET[col].mode()[0]
                print(f"Warning: '{user_preferences[col]}' not found in training data for '{col}'. Using '{most_common}' instead.")
                input_data[col] = encoder.transform([most_common])
        
        # Prepare original dataset for prediction
        df_encoded = DATASET.copy()
        for col in categorical_features:
            df_encoded[col] = ENCODERS[col].transform(DATASET[col])
        
        # Scale numerical features in original dataset
        df_encoded[numerical_features] = SCALER.transform(DATASET[numerical_features])
        
        # Prepare features for prediction
        X_pred = df_encoded[numerical_features + categorical_features]
        
        # Make predictions
        predicted_scores = MODEL.predict(X_pred)
        
        # Add predictions to original dataset
        DATASET['predicted_suitability'] = predicted_scores
        
        # Sort by predicted suitability and get top recommendations
        recommendations = DATASET.sort_values(
            by='predicted_suitability', 
            ascending=False
        ).head(num_recommendations)
        
        # Convert recommendations to JSON-friendly format
        # Check if 'id' and 'name' columns exist, otherwise use available columns
        output_columns = ['predicted_suitability']
        
        # Map common column names to expected columns
        column_mapping = {
            'id': ['id', 'track_id', 'spotify_id', 'song_id'],
            'name': ['name', 'track_name', 'song_name', 'title'],
            'artists': ['artists', 'artist', 'artist_name', 'artist_names']
        }
        
        for expected_col, possible_cols in column_mapping.items():
            # Find the first matching column that exists in the dataset
            matching_col = next((col for col in possible_cols if col in DATASET.columns), None)
            if matching_col:
                output_columns.append(matching_col)
        
        # Add categorical features to output
        output_columns.extend([col for col in categorical_features if col in DATASET.columns])
        
        # Make sure all output columns exist in the dataset
        valid_columns = [col for col in output_columns if col in DATASET.columns]
        
        # Create result with available columns
        result = recommendations[valid_columns].to_dict(orient='records')
        
        # Return recommendations
        return jsonify({
            "success": True,
            "count": len(result),
            "available_columns": valid_columns,
            "recommendations": result
        })
    
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/song/<song_id>', methods=['GET'])
def get_song_details(song_id):
    """Get detailed information about a specific song"""
    global DATASET
    
    if DATASET is None:
        load_resources()
    
    try:
        # Determine which column contains the song IDs
        id_column = next((col for col in ['id', 'track_id', 'spotify_id', 'song_id'] 
                          if col in DATASET.columns), None)
        
        if not id_column:
            return jsonify({
                "success": False, 
                "error": "No ID column found in dataset", 
                "available_columns": DATASET.columns.tolist()
            }), 500
        
        # Find song by ID
        song = DATASET[DATASET[id_column] == song_id]
        
        if song.empty:
            return jsonify({"success": False, "error": "Song not found"}), 404
        
        # Convert to dictionary
        song_details = song.iloc[0].to_dict()
        
        return jsonify({
            "success": True,
            "song": song_details
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Load resources at startup
    print("Initializing Music Recommendation API")
    print(f"Looking for ML resources in: {ML_SERVICE_PATH}")
    
    try:
        load_resources()
        
        # Get port from environment or use default
        port = int(os.environ.get('PORT', 5000))
        
        print(f"Starting server on port {port}")
        # Run the app
        app.run(host='0.0.0.0', port=port, debug=True)
    except Exception as e:
        print(f"Failed to start server: {str(e)}")
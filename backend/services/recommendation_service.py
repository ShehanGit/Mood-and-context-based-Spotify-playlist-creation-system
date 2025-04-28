import os
import pandas as pd
import numpy as np
import joblib
from config import Config

class RecommendationService:
    """Service for generating music recommendations"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.dataset = None
        self.ml_path = Config.ML_SERVICE_PATH
    
    def load_resources(self):
        """Load all necessary ML resources for prediction"""
        try:
            # Load model
            model_path = os.path.join(self.ml_path, "improved_xgb_model.pkl")
            print(f"Loading model from: {model_path}")
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.ml_path, "numerical_scaler.pkl")
            print(f"Loading scaler from: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            # Load encoders
            categorical_features = ["mood", "age_group", "activity", "weather"]
            for col in categorical_features:
                encoder_path = os.path.join(self.ml_path, f"{col}_encoder.pkl")
                print(f"Loading encoder for {col} from: {encoder_path}")
                self.encoders[col] = joblib.load(encoder_path)
            
            # Load dataset
            dataset_path = os.path.join(self.ml_path, "enhanced_dataset_improved.csv")
            print(f"Loading dataset from: {dataset_path}")
            self.dataset = pd.read_csv(dataset_path)
            
            print("All resources loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading resources: {str(e)}")
            return False
    
    def get_dataset_columns(self):
        """Get all columns in the dataset"""
        if self.dataset is None and not self.load_resources():
            return None
        
        return self.dataset.columns.tolist()
    
    def get_categories(self):
        """Get all possible values for categorical features"""
        if self.dataset is None and not self.load_resources():
            return None
        
        categorical_features = ["mood", "age_group", "activity", "weather"]
        categories = {}
        
        for feature in categorical_features:
            categories[feature] = self.dataset[feature].unique().tolist()
        
        return categories
    
    def recommend_songs(self, preferences, num_recommendations=50):
        """Generate song recommendations based on user preferences"""
        # Load resources if not already loaded
        if self.model is None or self.scaler is None or not self.encoders or self.dataset is None:
            if not self.load_resources():
                return None, "Failed to load ML resources"
        
        try:
            # Define expected features
            numerical_features = ["valence", "energy", "danceability", "tempo", 
                                "acousticness", "instrumentalness", "popularity", "temperature"]
            categorical_features = ["mood", "age_group", "activity", "weather"]
            
            # Check if required parameters are present
            for feature in numerical_features + categorical_features:
                if feature not in preferences:
                    return None, f"Missing required parameter: {feature}"
            
            # Create input dataframe
            input_data = pd.DataFrame([preferences])
            
            # Scale numerical features
            input_numerical = input_data[numerical_features].values.reshape(1, -1)
            input_data[numerical_features] = self.scaler.transform(input_numerical)
            
            # Encode categorical features
            for col in categorical_features:
                encoder = self.encoders[col]
                # Try to encode the value
                try:
                    input_data[col] = encoder.transform([preferences[col]])
                except ValueError:
                    # If category not in training data, use most common
                    most_common = self.dataset[col].mode()[0]
                    print(f"Warning: '{preferences[col]}' not found in training data for '{col}'. Using '{most_common}' instead.")
                    input_data[col] = encoder.transform([most_common])
            
            # Prepare original dataset for prediction
            df_encoded = self.dataset.copy()
            for col in categorical_features:
                df_encoded[col] = self.encoders[col].transform(self.dataset[col])
            
            # Scale numerical features in original dataset
            df_encoded[numerical_features] = self.scaler.transform(self.dataset[numerical_features])
            
            # Prepare features for prediction
            X_pred = df_encoded[numerical_features + categorical_features]
            
            # Make predictions
            predicted_scores = self.model.predict(X_pred)
            
            # Add predictions to original dataset
            df_encoded['predicted_suitability'] = predicted_scores
            
            # Sort by predicted suitability and get top recommendations
            recommendations = df_encoded.sort_values(
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
                matching_col = next((col for col in possible_cols if col in self.dataset.columns), None)
                if matching_col:
                    output_columns.append(matching_col)
            
            # Add categorical features to output
            output_columns.extend([col for col in categorical_features if col in self.dataset.columns])
            
            # Make sure all output columns exist in the dataset
            valid_columns = [col for col in output_columns if col in recommendations.columns]
            
            # Create result with available columns
            result = recommendations[valid_columns].to_dict(orient='records')
            
            return result, None
        
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return None, str(e)
    
    def get_song_details(self, song_id):
        """Get detailed information about a specific song"""
        if self.dataset is None and not self.load_resources():
            return None, "Failed to load dataset"
        
        try:
            # Determine which column contains the song IDs
            id_column = next((col for col in ['id', 'track_id', 'spotify_id', 'song_id'] 
                             if col in self.dataset.columns), None)
            
            if not id_column:
                return None, "No ID column found in dataset"
            
            # Find song by ID
            song = self.dataset[self.dataset[id_column] == song_id]
            
            if song.empty:
                return None, "Song not found"
            
            # Convert to dictionary
            song_details = song.iloc[0].to_dict()
            
            return song_details, None
        
        except Exception as e:
            return None, str(e)
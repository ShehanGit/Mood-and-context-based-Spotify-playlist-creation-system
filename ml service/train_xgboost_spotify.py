import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
import sys

# Set the path to save all model files
SAVE_PATH = os.path.abspath(".")  # Current directory
print(f"Files will be saved to: {SAVE_PATH}")

# Check if enhanced_dataset.csv exists in the current directory
dataset_path = os.path.join(SAVE_PATH, "enhanced_dataset.csv")
if not os.path.exists(dataset_path):
    print(f"Error: {dataset_path} not found. Please make sure this file exists.")
    sys.exit(1)

print("Step 1: Loading and enhancing the dataset...")
# Load the original dataset
df = pd.read_csv(dataset_path)

# Define more nuanced functions for context-based attributes
def assign_mood(valence, energy, acousticness):
    if valence > 0.7 and energy > 0.7:
        return "ecstatic"
    elif valence > 0.5 and energy > 0.5:
        return "happy"
    elif valence > 0.5 and energy < 0.5:
        return "relaxed"
    elif valence < 0.3 and energy > 0.7:
        return "angry"
    elif valence < 0.3 and energy < 0.5:
        return "sad"
    elif acousticness > 0.7:
        return "calm"
    else:
        return "neutral"

def assign_age_group(popularity, instrumentalness, tempo):
    if popularity > 80:
        return "teen"
    elif popularity > 65:
        return "young_adult"
    elif popularity > 45:
        return "adult"
    elif instrumentalness > 0.7 or (tempo < 100 and popularity < 45):
        return "senior"
    else:
        return "all_ages"

def assign_activity(tempo, energy, acousticness, danceability):
    if tempo > 125 and energy > 0.7:
        return "intense_workout"
    elif tempo > 110 and energy > 0.6:
        return "workout"
    elif danceability > 0.7 and energy > 0.6:
        return "party"
    elif acousticness > 0.7 and energy < 0.4:
        return "meditation"
    elif acousticness > 0.5 and energy < 0.5:
        return "study"
    elif energy < 0.3:
        return "sleep"
    elif 0.4 < danceability < 0.7 and energy < 0.6:
        return "commute"
    else:
        return "casual_listening"

def assign_weather(valence, instrumentalness, acousticness, energy):
    if valence < 0.3 and instrumentalness > 0.5:
        return "stormy"
    elif valence < 0.4 and acousticness > 0.6:
        return "rainy"
    elif 0.4 <= valence < 0.6 and energy < 0.5:
        return "cloudy"
    elif valence > 0.7 and energy > 0.6:
        return "sunny"
    else:
        return "mild"

def assign_temperature(energy, valence):
    base_temp = 15
    energy_effect = (energy - 0.5) * 10
    valence_effect = (valence - 0.5) * 6
    return base_temp + energy_effect + valence_effect

# Apply these functions to create more nuanced context features
df["mood"] = df.apply(lambda row: assign_mood(row["valence"], row["energy"], row["acousticness"]), axis=1)
df["age_group"] = df.apply(lambda row: assign_age_group(row["popularity"], row["instrumentalness"], row["tempo"]), axis=1)
df["activity"] = df.apply(lambda row: assign_activity(row["tempo"], row["energy"], row["acousticness"], row["danceability"]), axis=1)
df["weather"] = df.apply(lambda row: assign_weather(row["valence"], row["instrumentalness"], row["acousticness"], row["energy"]), axis=1)
df["temperature"] = df.apply(lambda row: assign_temperature(row["energy"], row["valence"]), axis=1)

# User contexts for suitability calculation
user_contexts = {
    "mood": "happy",
    "age_group": "young_adult",
    "activity": "workout",
    "weather": "sunny",
    "temperature": 24
}

print("Step 2: Calculating improved suitability scores...")
# Calculate suitability using the improved approach
def calculate_suitability(row, user_contexts):
    # Base score starts at 0.5 instead of 0
    score = 0.5
    
    # Context matching with weighted importance
    context_weights = {
        "mood": 0.25,
        "activity": 0.25, 
        "weather": 0.15,
        "age_group": 0.15
    }
    
    for context, weight in context_weights.items():
        # Exact match gives full points
        if row[context] == user_contexts[context]:
            score += weight
        else:
            # Partial matching for similar contexts
            # Define similarity mappings for each context
            similarity_maps = {
                "mood": {
                    "happy": {"ecstatic": 0.8, "relaxed": 0.5, "neutral": 0.3},
                    "ecstatic": {"happy": 0.8, "neutral": 0.2},
                    "relaxed": {"happy": 0.5, "calm": 0.7, "neutral": 0.4},
                    "sad": {"calm": 0.3, "neutral": 0.2},
                    "angry": {"intense_workout": 0.4},
                    "calm": {"relaxed": 0.7, "neutral": 0.4, "sad": 0.3},
                    "neutral": {"relaxed": 0.4, "happy": 0.3, "calm": 0.4}
                },
                "activity": {
                    "workout": {"intense_workout": 0.8, "party": 0.5, "casual_listening": 0.3},
                    "intense_workout": {"workout": 0.8, "party": 0.4},
                    "party": {"workout": 0.5, "casual_listening": 0.4},
                    "meditation": {"sleep": 0.6, "study": 0.4},
                    "study": {"meditation": 0.4, "commute": 0.3, "casual_listening": 0.4},
                    "sleep": {"meditation": 0.6},
                    "commute": {"casual_listening": 0.7, "study": 0.3},
                    "casual_listening": {"commute": 0.7, "party": 0.4, "workout": 0.3}
                },
                "weather": {
                    "sunny": {"mild": 0.6, "cloudy": 0.3},
                    "cloudy": {"mild": 0.7, "rainy": 0.5, "sunny": 0.3},
                    "rainy": {"cloudy": 0.5, "stormy": 0.6},
                    "stormy": {"rainy": 0.6},
                    "mild": {"sunny": 0.6, "cloudy": 0.7}
                },
                "age_group": {
                    "teen": {"young_adult": 0.7, "all_ages": 0.5},
                    "young_adult": {"teen": 0.7, "adult": 0.7, "all_ages": 0.8},
                    "adult": {"young_adult": 0.7, "senior": 0.5, "all_ages": 0.8},
                    "senior": {"adult": 0.5, "all_ages": 0.6},
                    "all_ages": {"teen": 0.5, "young_adult": 0.8, "adult": 0.8, "senior": 0.6}
                }
            }
            
            # Look up similarity score if available
            if context in similarity_maps and user_contexts[context] in similarity_maps[context]:
                if row[context] in similarity_maps[context][user_contexts[context]]:
                    similarity = similarity_maps[context][user_contexts[context]][row[context]]
                    score += weight * similarity
    
    # Temperature matching - use a bell curve for temperature
    temp_diff = abs(row["temperature"] - user_contexts["temperature"])
    # Temperature gets a 0.2 weight, with exponential decay for differences
    temp_score = 0.2 * np.exp(-0.5 * (temp_diff/5)**2)  # Standard deviation of 5°C
    score += temp_score
    
    # Audio feature matching
    activity_profiles = {
        "workout": {"energy": 0.8, "tempo": 0.8, "danceability": 0.7, "valence": 0.7},
        "intense_workout": {"energy": 0.9, "tempo": 0.9, "danceability": 0.6, "valence": 0.6},
        "party": {"energy": 0.8, "tempo": 0.7, "danceability": 0.9, "valence": 0.8},
        "meditation": {"energy": 0.2, "tempo": 0.3, "acousticness": 0.8, "instrumentalness": 0.6},
        "study": {"energy": 0.3, "tempo": 0.5, "acousticness": 0.6, "instrumentalness": 0.4},
        "sleep": {"energy": 0.1, "tempo": 0.2, "acousticness": 0.8, "instrumentalness": 0.6},
        "commute": {"energy": 0.5, "tempo": 0.5, "danceability": 0.5, "valence": 0.6},
        "casual_listening": {"energy": 0.5, "tempo": 0.5, "danceability": 0.5, "valence": 0.6}
    }
    
    # If we have a profile for the requested activity
    if user_contexts["activity"] in activity_profiles:
        profile = activity_profiles[user_contexts["activity"]]
        # Create feature vectors
        profile_vector = []
        song_vector = []
        
        for feature, target_value in profile.items():
            if feature in row:
                profile_vector.append(target_value)
                # Normalize tempo to 0-1 range for comparison
                if feature == "tempo":
                    # Assuming tempo ranges from 50-200 BPM
                    song_vector.append(min(1.0, max(0.0, (row[feature] - 50) / 150)))
                else:
                    song_vector.append(row[feature])
        
        # Calculate similarity using cosine similarity if we have features to compare
        if profile_vector and len(song_vector) > 0:
            # Transform to avoid division by zero in cosine similarity
            profile_vector = np.array(profile_vector) + 0.0001
            song_vector = np.array(song_vector) + 0.0001
            
            # Calculate cosine similarity (1 - cosine distance) using numpy
            profile_norm = np.linalg.norm(profile_vector)
            song_norm = np.linalg.norm(song_vector)
            dot_product = np.dot(profile_vector, song_vector)
            
            if profile_norm > 0 and song_norm > 0:
                feature_similarity = dot_product / (profile_norm * song_norm)
                # Add feature similarity with a weight of 0.25
                score += 0.25 * feature_similarity
    
    # Ensure score is between 0 and 1
    return min(1.0, max(0.0, score))

# Apply the suitability calculation
df["suitability"] = df.apply(lambda row: calculate_suitability(row, user_contexts), axis=1)

# Save the enhanced dataset
enhanced_dataset_path = os.path.join(SAVE_PATH, "enhanced_dataset_improved.csv")
df.to_csv(enhanced_dataset_path, index=False)
print(f"Enhanced dataset saved to: {enhanced_dataset_path}")

print("Step 3: Preprocessing data for model training...")
# Define feature columns
numerical_features = ["valence", "energy", "danceability", "tempo", "acousticness", 
                      "instrumentalness", "popularity", "temperature"]
categorical_features = ["mood", "age_group", "activity", "weather"]
target = "suitability"

# Standard scaling for numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
scaler_path = os.path.join(SAVE_PATH, "numerical_scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"Numerical scaler saved to: {scaler_path}")

# Encode categorical variables
encoders = {}
df_encoded = df[numerical_features + [target]].copy()
for col in categorical_features:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    encoders[col] = le
    # Save the encoder
    encoder_path = os.path.join(SAVE_PATH, f"{col}_encoder.pkl")
    joblib.dump(le, encoder_path)
    print(f"{col} encoder saved to: {encoder_path}")

# Combine all features
X = df_encoded[numerical_features + categorical_features]
y = df_encoded[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

print("Step 4: Training XGBoost model...")
# Train the XGBoost model with optimized parameters - Fix for eval_metric error
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Simple fit without eval_metric parameter
xgb_model.fit(X_train, y_train)
print("Model training completed successfully")

# Evaluate the model
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance Metrics:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save the model
model_path = os.path.join(SAVE_PATH, "improved_xgb_model.pkl")
joblib.dump(xgb_model, model_path)
print(f"Model saved to: {model_path}")

# Save feature map
feature_map_path = os.path.join(SAVE_PATH, "feature_map.pkl")
joblib.dump(dict(zip(X.columns, range(len(X.columns)))), feature_map_path)
print(f"Feature map saved to: {feature_map_path}")

print("\nAll necessary files have been generated. You can now run the Flask API.")
# Mood and Context-Based Spotify Playlist Generator

## Overview
This Python-based application generates personalized Spotify playlists based on user context (activity, weather, age group) and mood. It uses an XGBoost machine learning model to predict track features (tempo, energy, valence) and the Spotify API to recommend and create playlists. Ideal for music enthusiasts and developers exploring ML-driven recommendation systems.

## Features
- **Contextual Recommendations**: Creates playlists tailored to user inputs like activity (e.g., workout, study), weather (e.g., sunny, rainy), age group, and mood (e.g., energetic, calm).
- **Machine Learning**: Employs XGBoost to predict optimal track features for personalized recommendations.
- **Spotify Integration**: Fetches track recommendations and creates playlists using the Spotify API.
- **Extensible**: Easily adaptable to include additional context features or genres.

## Prerequisites
- Python 3.8+
- Spotify Developer Account (for API credentials)
- Required Python packages: `spotipy`, `pandas`, `xgboost`, `scikit-learn`

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   ```bash
   pip install spotipy pandas xgboost scikit-learn
   ```

3. **Configure Spotify API**:
   - Create a Spotify Developer account at [developer.spotify.com](https://developer.spotify.com).
   - Create an app to obtain `client_id`, `client_secret`, and set `redirect_uri` to `http://localhost:5000/callback`.
   - Update `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, and `SPOTIFY_REDIRECT_URI` in `spotify_playlist_generator.py`.

4. **Run the Application**:
   ```bash
   python spotify_playlist_generator.py
   ```

## Usage
1. **Run the Script**: Execute `spotify_playlist_generator.py` to authenticate with Spotify and generate a playlist.
2. **Provide Context Input**: Modify the `context_input` dictionary in the script with desired values, e.g.:
   ```python
   context_input = {
       "activity": "workout",
       "weather": "sunny",
       "age_group": "20-30",
       "mood": "energetic"
   }
   ```
3. **Output**: The script creates a Spotify playlist and prints its URL.

## Example
```python
context_input = {
    "activity": "study",
    "weather": "rainy",
    "age_group": "20-30",
    "mood": "calm"
}
```
Running the script generates a playlist URL: `https://open.spotify.com/playlist/<playlist_id>`.

## Technologies
- **Python**: Core programming language.
- **XGBoost**: Machine learning model for feature prediction.
- **Spotify API (Spotipy)**: For track recommendations and playlist creation.
- **pandas**: Data preprocessing and handling.
- **scikit-learn**: Categorical encoding with LabelEncoder.

## Future Enhancements
- Add a Flask web interface for user input.
- Incorporate real-time weather data via an API (e.g., OpenWeatherMap).
- Expand training data for more accurate predictions.

## License
MIT License. See `LICENSE` for details.

## Contact
For questions or contributions, contact [Your Name] at [your.email@example.com].
"""

import requests
import base64
import json
import urllib.parse
from config import Config

class SpotifyService:
    """Service for interacting with the Spotify API"""
    
    @staticmethod
    def get_auth_url():
        """Generate the authorization URL for user to grant permission"""
        client_id = Config.SPOTIFY_CLIENT_ID
        redirect_uri = Config.SPOTIFY_REDIRECT_URI
        scope = "playlist-modify-public playlist-modify-private"
        
        auth_url = "https://accounts.spotify.com/authorize"
        auth_url += f"?client_id={client_id}"
        auth_url += f"&response_type=code"
        auth_url += f"&redirect_uri={urllib.parse.quote(redirect_uri)}"
        auth_url += f"&scope={urllib.parse.quote(scope)}"
        
        return auth_url
    
    @staticmethod
    def get_client_credentials_token():
        """Get an access token using client credentials flow (no user login required)"""
        client_id = Config.SPOTIFY_CLIENT_ID
        client_secret = Config.SPOTIFY_CLIENT_SECRET
        
        auth_string = f"{client_id}:{client_secret}"
        auth_bytes = auth_string.encode("utf-8")
        auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
        
        url = "https://accounts.spotify.com/api/token"
        headers = {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}
        
        response = requests.post(url, headers=headers, data=data)
        json_result = response.json()
        
        if "access_token" in json_result:
            return json_result["access_token"]
        else:
            print(f"Error getting token: {json_result.get('error', 'Unknown error')}")
            return None
    
    @staticmethod
    def get_user_token(auth_code):
        """Exchange authorization code for access token"""
        client_id = Config.SPOTIFY_CLIENT_ID
        client_secret = Config.SPOTIFY_CLIENT_SECRET
        redirect_uri = Config.SPOTIFY_REDIRECT_URI
        
        auth_string = f"{client_id}:{client_secret}"
        auth_bytes = auth_string.encode("utf-8")
        auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
        
        url = "https://accounts.spotify.com/api/token"
        headers = {
            "Authorization": f"Basic {auth_base64}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "redirect_uri": redirect_uri
        }
        
        response = requests.post(url, headers=headers, data=data)
        json_result = response.json()
        
        if "access_token" in json_result:
            return json_result
        else:
            print(f"Error getting user token: {json_result.get('error', 'Unknown error')}")
            return None
    
    @staticmethod
    def get_user_profile(token):
        """Get the current user's Spotify profile"""
        url = "https://api.spotify.com/v1/me"
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting user profile: {response.status_code}")
            print(response.text)
            return None
    
    @staticmethod
    def create_playlist(token, user_id, name, description=""):
        """Create a new Spotify playlist"""
        url = f"https://api.spotify.com/v1/users/{user_id}/playlists"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        data = {
            "name": name,
            "description": description,
            "public": True
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code == 201:
            return response.json()
        else:
            print(f"Error creating playlist: {response.status_code}")
            print(response.text)
            return None
    
    @staticmethod
    def add_tracks_to_playlist(token, playlist_id, track_ids):
        """Add tracks to a Spotify playlist"""
        url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        # Convert track IDs to full URIs if needed
        uris = []
        for track_id in track_ids:
            if not track_id.startswith("spotify:track:"):
                uris.append(f"spotify:track:{track_id}")
            else:
                uris.append(track_id)
        
        # Add tracks in batches of 100 (Spotify API limit)
        batch_size = 100
        success = True
        
        for i in range(0, len(uris), batch_size):
            batch = uris[i:i+batch_size]
            data = {"uris": batch}
            
            response = requests.post(url, headers=headers, data=json.dumps(data))
            
            if response.status_code not in [200, 201]:
                print(f"Error adding tracks (batch {i//batch_size + 1}): {response.status_code}")
                print(response.text)
                success = False
        
        return success
    
    @staticmethod
    def get_track_details(track_id, token):
        """Get details for a specific track"""
        url = f"https://api.spotify.com/v1/tracks/{track_id}"
        headers = {"Authorization": f"Bearer {token}"}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting track details: {response.status_code}")
            print(response.text)
            return None
import requests
import base64
import json
import webbrowser
import time

# Your Spotify API credentials
CLIENT_ID = "d7eb111127ba40d083314a2c429caa74"
CLIENT_SECRET = "1deb80662ad24c9186ea557b2d901380"
REDIRECT_URI = "http://127.0.0.1:3000"

def get_client_credentials_token():
    """Get an access token using client credentials flow"""
    auth_string = f"{CLIENT_ID}:{CLIENT_SECRET}"
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

def get_auth_code():
    """Get the authorization code using user authorization"""
    # Scopes for playlist modification
    scope = "playlist-modify-public playlist-modify-private"
    
    # Create the authorization URL
    auth_url = f"https://accounts.spotify.com/authorize"
    auth_url += f"?client_id={CLIENT_ID}"
    auth_url += f"&response_type=code"
    auth_url += f"&redirect_uri={REDIRECT_URI}"
    auth_url += f"&scope={scope}"
    
    # Open the authorization URL in the default web browser
    print("\n=== Spotify Authorization ===")
    print("1. Opening your browser to authorize the application.")
    print("2. Login to Spotify and approve the permissions.")
    print("3. After approval, you'll be redirected to a page that can't be reached.")
    print("4. Copy the FULL URL from your browser's address bar and paste it here.")
    
    # Open the URL in the browser
    webbrowser.open(auth_url)
    time.sleep(2)  # Give the browser some time to open
    
    # Wait for the user to authenticate and get redirected
    print("\nWaiting for authorization...")
    redirect_url = input("\nPaste the redirect URL here: ")
    
    # Extract the code from the URL
    try:
        code = redirect_url.split("code=")[1].split("&")[0]
        return code
    except IndexError:
        print("Error: Could not extract authorization code from URL.")
        return None

def get_user_token(auth_code):
    """Exchange authorization code for access token"""
    auth_string = f"{CLIENT_ID}:{CLIENT_SECRET}"
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
        "redirect_uri": REDIRECT_URI
    }
    
    response = requests.post(url, headers=headers, data=data)
    json_result = response.json()
    
    if "access_token" in json_result:
        return json_result["access_token"]
    else:
        print(f"Error getting user token: {json_result.get('error', 'Unknown error')}")
        return None

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

def get_recommended_track_ids():
    """Get recommended track IDs from local API or manual input"""
    try:
        # Try to get recommendations from your local API
        preferences = {
            "valence": 0.7,
            "energy": 0.8,
            "danceability": 0.75,
            "tempo": 120,
            "acousticness": 0.2,
            "instrumentalness": 0.1,
            "popularity": 70,
            "temperature": 25,
            "mood": "happy",
            "age_group": "young_adult",
            "activity": "workout",
            "weather": "sunny"
        }
        
        print("Attempting to get recommendations from local API...")
        response = requests.post(
            "http://localhost:5000/api/recommend", 
            json={"preferences": preferences, "num_recommendations": 50},
            timeout=5  # 5 second timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                recommendations = data.get("recommendations", [])
                
                track_ids = []
                for track in recommendations:
                    # Try different possible field names for the track ID
                    for field in ['id', 'track_id', 'spotify_id']:
                        if field in track and track[field]:
                            track_ids.append(track[field])
                            break
                
                if track_ids:
                    print(f"✅ Successfully retrieved {len(track_ids)} recommended tracks from API!")
                    return track_ids
    except Exception as e:
        print(f"Could not connect to recommendation API: {str(e)}")
    
    # Fallback to manual input
    print("\nEnter track IDs manually:")
    print("(Enter one ID per line, press Enter on an empty line to finish)")
    
    track_ids = []
    while True:
        track_id = input("> ").strip()
        if not track_id:
            break
        track_ids.append(track_id)
    
    return track_ids

def main():
    print("=== Spotify Playlist Creator ===")
    
    # Step 1: Get the authorization code
    auth_code = get_auth_code()
    if not auth_code:
        print("Failed to get authorization code.")
        return
    
    # Step 2: Exchange the code for a user access token
    user_token = get_user_token(auth_code)
    if not user_token:
        print("Failed to get user access token.")
        return
    
    print("✅ Successfully authenticated with Spotify!")
    
    # Step 3: Get the user's Spotify profile
    user_profile = get_user_profile(user_token)
    if not user_profile:
        print("Failed to get user profile.")
        return
    
    user_id = user_profile["id"]
    print(f"✅ Retrieved user profile for {user_profile['display_name']} (ID: {user_id})")
    
    # Step 4: Get track IDs for the playlist
    track_ids = get_recommended_track_ids()
    if not track_ids:
        print("No tracks provided. Exiting.")
        return
    
    # Step 5: Create a new playlist
    playlist_name = input("\nEnter a name for your playlist: ")
    playlist_description = input("Enter a description (optional): ")
    
    print("\nCreating playlist...")
    playlist = create_playlist(user_token, user_id, playlist_name, playlist_description)
    if not playlist:
        print("Failed to create playlist.")
        return
    
    playlist_id = playlist["id"]
    print(f"✅ Created playlist: {playlist_name} (ID: {playlist_id})")
    
    # Step 6: Add tracks to the playlist
    print(f"\nAdding {len(track_ids)} tracks to playlist...")
    success = add_tracks_to_playlist(user_token, playlist_id, track_ids)
    
    if success:
        print(f"\n✅ SUCCESS! Your playlist '{playlist_name}' is ready with {len(track_ids)} tracks.")
        print(f"Open it in Spotify: {playlist['external_urls']['spotify']}")
    else:
        print("\n❌ There were some errors adding tracks to the playlist.")

if __name__ == "__main__":
    main()
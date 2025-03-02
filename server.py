from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import os
import random
import requests
import base64
from clustering import ArtistCluster

app = Flask(__name__)

data_path = "data/dataset.csv"
clusterer = ArtistCluster(data_path)
clusterer.fit_clusters()

# spotify API credentials
SPOTIFY_CLIENT_ID = os.environ['SPOTIFY_CLIENT_ID']
SPOTIFY_CLIENT_SECRET = os.environ['SPOTIFY_CLIENT_SECRET']

try:
    df = pd.read_csv(data_path)
    has_track_data = 'track_preview_url' in df.columns
except Exception as e:
    print(f"Error loading dataset: {e}")
    has_track_data = False

@app.route('/')
def index():
    return render_template('index.html')

def get_spotify_token():
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("Spotify credentials not found in environment variables")
        return None
        
    auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    auth_bytes = auth_string.encode('utf-8')
    auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')
    
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        json_result = response.json()
        return json_result.get("access_token")
    except Exception as e:
        print(f"Error getting Spotify token: {e}")
        return None

def search_spotify_track(artist_name, token):
    if not token:
        return None
        
    url = "https://api.spotify.com/v1/search"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    params = {
        "q": f"artist:{artist_name}",
        "type": "track",
        "limit": 5
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        json_result = response.json()
        
        tracks = json_result.get("tracks", {}).get("items", [])
        if tracks:
            track = random.choice(tracks)
            track_id = track["id"]
            track_name = track["name"]
            return {
                "track_id": track_id,
                "track_name": track_name,
                "artist": artist_name
            }
    except Exception as e:
        print(f"Error searching Spotify for {artist_name}: {e}")
    
    return None

@app.route('/predict', methods=['POST'])
def predict_artists():
    try:
        artist_name = request.form.get('artist_name').lower()
        
        if not artist_name:
            return jsonify({"error": "Please enter an artist name"})
        
        similar_artists_with_scores = clusterer.find_similar_artists(artist_name)
        
        if similar_artists_with_scores is None:
            return jsonify({"error": f"Artist '{artist_name}' not found in our database"})
        
        top_similar_artists = [artist.capitalize() for artist, _ in similar_artists_with_scores[:5]] #show top 5 artits, you can change it
        
        artist_preview = get_artist_preview(artist_name)
    
        preview_tracks = []
        preview_tracks.append(artist_preview) if artist_preview else None
        
        for similar_artist in top_similar_artists:
            artist_preview = get_artist_preview(similar_artist)
            if artist_preview:
                preview_tracks.append(artist_preview)
        
        spotify_tracks = []
        spotify_token = get_spotify_token()
        
        input_artist_track = search_spotify_track(artist_name, spotify_token)
        if input_artist_track:
            spotify_tracks.append(input_artist_track)
        
        for similar_artist in top_similar_artists:
            track = search_spotify_track(similar_artist, spotify_token)
            if track:
                spotify_tracks.append(track)
        
        return jsonify({
            "similar_artists": top_similar_artists,
            "preview_tracks": preview_tracks,
            "spotify_tracks": spotify_tracks
        })
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"})

def get_artist_preview(artist_name):
    if not has_track_data:
        return None
        
    artist_tracks = df[df['track_artist'] == artist_name]
    if not artist_tracks.empty and 'track_preview_url' in artist_tracks.columns:
        valid_tracks = artist_tracks[artist_tracks['track_preview_url'].notna()]
        if not valid_tracks.empty:
            random_track = valid_tracks.sample(1).iloc[0]
            return {
                "url": random_track['track_preview_url'],
                "name": random_track.get('track_name', 'Sample Track'),
                "artist": artist_name
            }
    return None

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/visualize')
def visualize_clusters():
    clusterer.plot_clusters(perplexity=30)
    return jsonify({"message": "Visualization displayed"})

if __name__ == "__main__":
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5050)
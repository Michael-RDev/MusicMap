# Artist Recommendation Flask App

![image](https://github.com/user-attachments/assets/f10ce754-188b-4b34-ac05-8f829546822e)



This is a Flask-based web application that provides artist recommendations using clustering techniques. It integrates with Spotify's API to fetch track previews and related artist information.

## Features

- Predict similar artists based on clustering techniques.
- Fetch track previews from a dataset (if available).
- Retrieve tracks from Spotify's API using client credentials authentication.
- Visualize artist clusters.
- Serve static files for frontend rendering.

## Installation

### Prerequisites
- Python 3.x
- Flask
- Pandas
- Requests
- Spotipy (for Spotify API interaction)
- Matplotlib (for visualization, if needed)

### Setup
1. Clone this repository:
   ```sh
   git clone https://github.com/Michael-RDev/MusicMap.git
   cd MusicMap
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up your Spotify API credentials:
   - Create a `.env` file or set the following environment variables:
     ```sh
     export SPOTIFY_CLIENT_ID="your_spotify_client_id"
     export SPOTIFY_CLIENT_SECRET="your_spotify_client_secret"
     ```
   - Alternatively, add them to your system’s environment variables.

## Usage

1. Ensure the dataset (`data/dataset.csv`) is available and formatted correctly.
2. Start the Flask application:
   ```sh
   python app.py
   ```
3. Access the web interface by navigating to:
   ```
   http://localhost:5050
   ```
4. Enter an artist's name in the input field to find similar artists (MAKE SURE YOU'RE ARTIST IS IN THE DATASET)
5. View and listen to Spotify track previews and recommendations.
6. Visualize clusters by visiting:
   ```
   http://localhost:5050/visualize
   ```

## API Endpoints

### `/` (Home Page)
Renders the main index page for user interaction.

### `/predict` (POST)
Predicts similar artists and fetches track previews.
#### Request Body:
```json
{
  "artist_name": "artist_name_here"
}
```
#### Response Example:
```json
{
  "similar_artists": ["Artist1", "Artist2", "Artist3"],
  "preview_tracks": [
    {"url": "preview_url", "name": "Track Name", "artist": "Artist1"}
  ],
  "spotify_tracks": [
    {"track_id": "id", "track_name": "Track Name", "artist": "Artist1"}
  ]
}
```

### `/visualize`
Generates and displays a cluster visualization.

### `/static/<path>`
Serves static files for frontend resources.

## Spotify API Integration

- The app uses the Spotify Web API to fetch track previews and artist data.
- Requires a valid `SPOTIFY_CLIENT_ID` and `SPOTIFY_CLIENT_SECRET`.

## Troubleshooting

- **Error fetching Spotify token?** Ensure your client ID and secret are set correctly.
- **Dataset missing?** Make sure `data/dataset.csv` exists and contains necessary columns.
- **Flask not starting?** Check port conflicts and ensure all dependencies are installed.
  
## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Author

Developed by Michael R. Feel free to reach out for any improvements or suggestions!


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


class ArtistCluster:
    def __init__(self, data_path, random_state=42):
        self.feature_columns = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                               'liveness', 'loudness', 'speechiness', 
                               'tempo', 'valence', 'time_signature', 'mode', 'key', "track_artist", "playlist_genre"]

        self.data_path = data_path
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        
        self.artist_features = None
        self.scaled_features = None
        self.num_genres = None
        self.kmeans = None
        self.cluster_labels = None
        self.cluster_artists = None
        self.input_artist = None
        
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        df.dropna(subset=self.feature_columns, inplace=True)
        df = df[self.feature_columns]

        df['track_artist'] = df['track_artist'].apply(lambda x: str(x).lower())
        df["playlist_genre"] = self.encoder.fit_transform(df["playlist_genre"])
        self.num_genres = df["playlist_genre"].nunique()
        self.artist_features = df.groupby('track_artist').mean()
        self.scaled_features = self.scaler.fit_transform(self.artist_features)

    def fit_clusters(self):
        self.kmeans = KMeans(n_clusters=self.num_genres, random_state=self.random_state) 
        self.cluster_labels = self.kmeans.fit_predict(self.scaled_features)
        return self.cluster_labels

    def plot_clusters(self, perplexity=30):
        if self.scaled_features is None or self.cluster_labels is None:
            raise ValueError("Clusters have not been fitted. Call fit_clusters first.")
            
        tsne = TSNE(n_components=2, random_state=self.random_state, perplexity=perplexity)
        reduced_features = tsne.fit_transform(self.scaled_features)
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                    c=self.cluster_labels, cmap='viridis', alpha=0.7)
        plt.title("Artist Clusters Visualization via TSNE")
        plt.xlabel("TSNE Component 1")
        plt.ylabel("TSNE Component 2")
        plt.colorbar(label='Cluster Label')
        plt.show()

    def find_similar_artists(self, input_artist):
        if self.artist_features is None or self.kmeans is None or self.cluster_labels is None:
            raise ValueError("Clusters have not been fitted. Please run fit_clusters() first.")
        
        if input_artist not in self.artist_features.index:
            print("Artist not found")
            return None

        self.input_artist = input_artist
        input_features = self.artist_features.loc[input_artist].values.reshape(1, -1)
        input_scaled = self.scaler.transform(input_features)
        cluster_index = self.kmeans.predict(input_scaled)[0]
        cluster_mask = self.cluster_labels == cluster_index
        self.cluster_artists = self.artist_features.index[cluster_mask].tolist()
        if input_artist in self.cluster_artists:
            self.cluster_artists.remove(input_artist)
            
        return self.calculate_artists_probability(cluster_index)
    
    def calculate_artists_probability(self, cluster_index):
        print("cluster #: ", cluster_index)

        if not self.cluster_artists:
            print("No similar artists found in the cluster")
            return None
            
        cluster_features = self.artist_features.loc[self.cluster_artists].values
        input_features = self.artist_features.loc[self.input_artist].values.reshape(1, -1)
        similarities = cosine_similarity(input_features, cluster_features)[0]
        probabilities = (similarities - np.min(similarities)) / (np.max(similarities) - np.min(similarities)) if len(similarities) > 1 else [1.0]
        artist_probabilities = dict(zip(self.cluster_artists, probabilities))
        sorted_artists = sorted(artist_probabilities.items(), key=lambda x: x[1], reverse=True)
        return sorted_artists


if __name__ == "__main__":
    data_path = "data/dataset.csv"
    clusterer = ArtistCluster(data_path)
    clusterer.fit_clusters()
    clusterer.plot_clusters(perplexity=30)
    example_artist = "Soundgarden"  # listening to them right now
    similar_artists = clusterer.find_similar_artists(example_artist.lower())
    if similar_artists:
        print(f"Artists similar to {example_artist}:")
        for artist, prob in similar_artists[:10]:
            print(f"Artist: {artist}, Similarity: {prob}")
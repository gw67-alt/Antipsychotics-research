import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import sys
import random
import json

# Library import checks
try:
    import librosa
except ImportError:
    print("Error: librosa library not found. Please install using: pip install librosa")
    sys.exit(1)

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
except ImportError:
    print("Error: scikit-learn library not found. Please install using: pip install scikit-learn")
    sys.exit(1)

class AdvancedSignalProcessor:
    def __init__(self, sample_rate=22050):
        """
        Initialize advanced signal processor with enhanced capabilities
        """
        self.sample_rate = sample_rate
        self.frequencies = self._create_frequency_mapping()

    def _create_frequency_mapping(self):
        """
        Create a more sophisticated frequency mapping for letters
        """
        # Expanded frequency mapping with harmonic considerations
        return {
            'A': (220, 440, 660),   # Fundamental, 2nd, 3rd harmonics
            'B': (247, 494, 741),
            'C': (262, 524, 786),
            'D': (294, 588, 882),
            'E': (330, 660, 990),
            'F': (349, 698, 1047),
            'G': (392, 784, 1176),
            'H': (440, 880, 1320),
            'I': (494, 988, 1482),
            'J': (523, 1046, 1569),
            'K': (587, 1174, 1761),
            'L': (659, 1318, 1977),
            'M': (698, 1396, 2094),
            'N': (784, 1568, 2352),
            'O': (880, 1760, 2640),
            'P': (988, 1976, 2964),
            'Q': (1047, 2094, 3141),
            'R': (1175, 2350, 3525),
            'S': (1319, 2638, 3957),
            'T': (1397, 2794, 4191),
            'U': (1568, 3136, 4704),
            'V': (1760, 3520, 5280),
            'W': (1976, 3952, 5928),
            'X': (2093, 4186, 6279),
            'Y': (2349, 4698, 7047),
            'Z': (2637, 5274, 7911)
        }

    def generate_signal(self, letter):
        """
        Generate an advanced signal for a single letter
        """
        duration = 0.5  # Half a second per letter
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)

        # Get frequency tuple for the letter
        freq_tuple = self.frequencies.get(letter.upper(), (0, 0, 0))

        # If no frequency found, generate silence
        if freq_tuple[0] == 0:
            return np.zeros(int(self.sample_rate * duration)), self.sample_rate

        # Generate complex signal with multiple harmonics
        signal_data = np.zeros_like(t)
        for i, freq in enumerate(freq_tuple):
            # Reduce amplitude for higher harmonics
            amplitude = 1.0 / (i + 1)
            signal_data += amplitude * np.sin(2 * np.pi * freq * t)

        # Apply sophisticated envelope (Hann window as a fallback)
        envelope = np.hanning(len(signal_data))
        signal_data *= envelope

        # Normalize and add slight noise
        signal_data = signal_data / np.max(np.abs(signal_data))
        signal_data += 0.05 * np.random.normal(0, 0.1, len(signal_data))

        return signal_data, self.sample_rate

    def generate_word_signal(self, word):
        """
        Generate a comprehensive signal for a complete word
        """
        word_signal_parts = []
        pause_duration = 0.1  # Short pause between letters

        # Filter and process only alphabetic characters
        valid_letters = [letter.upper() for letter in word if letter.upper() in self.frequencies]

        if not valid_letters:
            return np.array([]), self.sample_rate

        for letter in valid_letters:
            # Generate letter signal
            letter_signal = self.generate_signal(letter)[0]
            word_signal_parts.append(letter_signal)
            
            # Add short pause between letters
            pause = np.zeros(int(self.sample_rate * pause_duration))
            word_signal_parts.append(pause)

        # Concatenate all parts
        full_word_signal = np.concatenate(word_signal_parts)
        return full_word_signal, self.sample_rate

    def extract_comprehensive_features(self, signal_data):
        """
        Extract advanced, multi-dimensional features
        """
        if signal_data.size == 0:
            return None

        features = {}

        try:
            # Time domain features
            features['zero_crossings'] = np.sum(librosa.zero_crossings(signal_data))
            features['rms_energy'] = librosa.feature.rms(y=signal_data)[0][0]
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=signal_data, sr=self.sample_rate)[0][0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal_data, sr=self.sample_rate)[0][0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=signal_data, sr=self.sample_rate)[0][0]
            
            features['spectral_centroid'] = spectral_centroid
            features['spectral_bandwidth'] = spectral_bandwidth
            features['spectral_rolloff'] = spectral_rolloff
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=signal_data, sr=self.sample_rate, n_mfcc=20)
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=signal_data, sr=self.sample_rate)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            
            # Tonnetz (tonal centroid) features
            tonnetz = librosa.feature.tonnetz(y=signal_data, sr=self.sample_rate)
            features['tonnetz_mean'] = np.mean(tonnetz, axis=1)

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

        return features

    def categorize_words(self, words, num_clusters=10, clustering_method='kmeans'):
        """
        Categorize words using advanced feature extraction and clustering
        """
        # Prepare feature matrix
        features_matrix = []
        word_labels = []

        # Extract features for words
        for word in words:
            word_signal, _ = self.generate_word_signal(word)
            features = self.extract_comprehensive_features(word_signal)
            
            if features is None:
                continue

            # Flatten features into a single vector
            feature_vector = []
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    feature_vector.extend(value)
                else:
                    feature_vector.append(value)
            
            features_matrix.append(feature_vector)
            word_labels.append(word)

        # Convert to numpy array
        features_matrix = np.array(features_matrix)

        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_matrix)

        # Dimensionality reduction
        pca = PCA(n_components=min(10, scaled_features.shape[1]))
        reduced_features = pca.fit_transform(scaled_features)

        # Clustering
        if clustering_method == 'kmeans':
            # Find optimal number of clusters
            best_score = -1
            best_n_clusters = 2
            max_clusters = min(num_clusters, len(word_labels))

            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(reduced_features)
                
                try:
                    score = silhouette_score(reduced_features, labels)
                    if score > best_score:
                        best_score = score
                        best_n_clusters = n_clusters
                except:
                    continue

            # Final clustering
            kmeans = KMeans(n_clusters=best_n_clusters, random_state=42, n_init=10)
            final_labels = kmeans.fit_predict(reduced_features)
            
            # Organize results
            word_groups = {}
            for word, label in zip(word_labels, final_labels):
                if label not in word_groups:
                    word_groups[label] = []
                word_groups[label].append(word)

            return word_groups, best_n_clusters

        elif clustering_method == 'dbscan':
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=3)
            labels = dbscan.fit_predict(reduced_features)
            
            # Organize results
            word_groups = {}
            for word, label in zip(word_labels, labels):
                if label != -1:  # Ignore noise points
                    if label not in word_groups:
                        word_groups[label] = []
                    word_groups[label].append(word)

            return word_groups, len(set(labels) - {-1})

def main():
    """
    Main function for word signal categorization
    """
    # Input file setup
    input_word_file = "words_alpha.txt"
    max_words_to_process = 500  # Increased sample size
    output_filename = "advanced_word_categories.txt"

    # Read words
    try:
        with open(input_word_file, 'r', encoding='utf-8') as file:
            all_words = [word.strip().upper() for word in file.readlines() 
                         if word.strip() and word.strip().isalpha()]
    except FileNotFoundError:
        print(f"Error: Input word file '{input_word_file}' not found.")
        sys.exit(1)

    if not all_words:
        print("Error: No valid words found in the input file.")
        sys.exit(1)

    print(f"Found {len(all_words)} alphabetic words.")

    # Select words to process
    if max_words_to_process is not None and len(all_words) > max_words_to_process:
        print(f"Taking a random sample of {max_words_to_process} words for processing...")
        word_dictionary = random.sample(all_words, max_words_to_process)
    else:
        print("Processing all words found in the file.")
        word_dictionary = all_words

    # Initialize processor
    processor = AdvancedSignalProcessor()

    # Perform categorization
    try:
        categorized_groups, num_clusters = processor.categorize_words(
            word_dictionary, 
            num_clusters=15,  # Maximum clusters to consider
            clustering_method='kmeans'  # Can also use 'dbscan'
        )

        # Save results
        print(f"\nSaving categorized groups to '{output_filename}'...")
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("--- Advanced Word Signal Categorization ---\n")
            f.write(f"Processed {len(word_dictionary)} words (sampled from {len(all_words)} total).\n")
            f.write(f"Number of clusters: {num_clusters}\n\n")

            for cluster_id, words_in_group in sorted(categorized_groups.items()):
                sorted_words = sorted(words_in_group)
                f.write(f"Cluster {cluster_id + 1} ({len(words_in_group)} words):\n")
                
                # Write words with line wrapping
                max_words_per_line = 10
                for i in range(0, len(sorted_words), max_words_per_line):
                    line_words = sorted_words[i:i+max_words_per_line]
                    f.write(f"  {', '.join(line_words)}\n")
                f.write("\n")

        print(f"Successfully saved categories to '{output_filename}'.")

    except Exception as e:
        print(f"Categorization error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
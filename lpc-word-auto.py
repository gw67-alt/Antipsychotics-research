import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import sys
import random

try:
    import librosa
    import librosa.display
except ImportError:
    print("Error: librosa library not found.")
    print("Please install it using: pip install librosa")
    sys.exit(1)

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Error: scikit-learn library not found.")
    print("Please install it using: pip install scikit-learn")
    sys.exit(1)

# --- Signal Generation Functions (Unchanged) ---
def generate_signal(letter):
    """
    Generate a basic signal for a single letter.
    Returns the signal data and the sample rate used.
    """
    # Basic parameters
    sample_rate = 4410  # Reduced sample rate
    duration = 0.5  # Half a second per letter
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Base frequencies for each letter
    frequencies = {
        'A': 220, 'B': 240, 'C': 260, 'D': 280, 'E': 300,
        'F': 320, 'G': 340, 'H': 360, 'I': 380, 'J': 400,
        'K': 420, 'L': 440, 'M': 460, 'N': 480, 'O': 500,
        'P': 520, 'Q': 540, 'R': 560, 'S': 580, 'T': 600,
        'U': 620, 'V': 640, 'W': 660, 'X': 680, 'Y': 700, 'Z': 720
    }

    # Get frequency for the letter (uppercase)
    freq = frequencies.get(letter.upper(), 0) # Default to 0Hz (silence) if not a letter

    # If frequency is 0 (not a standard letter), generate silence
    if freq == 0:
        signal_data = np.zeros(int(sample_rate * duration))
    else:
        # Generate signal with some complexity
        signal_data = np.sin(2 * np.pi * freq * t)
        signal_data += 0.5 * np.sin(2 * np.pi * (freq * 2) * t) # Add a harmonic

        # Apply envelope (Hanning window) to smooth transitions
        envelope = np.hanning(len(signal_data))
        signal_data = signal_data * envelope

    return signal_data, sample_rate

def generate_word_signal(word):
    """
    Generates a concatenated signal for a given word by joining letter signals.
    Filters out non-alphabetic characters.
    Returns the full word signal and the sample rate.
    """
    word_signal_parts = []
    sample_rate = 4410 # Default, will be confirmed by generate_signal
    valid_letters = 0

    for letter in word:
        if 'A' <= letter.upper() <= 'Z':
            signal_part, sr = generate_signal(letter.upper())
            word_signal_parts.append(signal_part)
            sample_rate = sr # Ensure consistent sample rate
            valid_letters += 1

    if not word_signal_parts:
        return np.array([]), sample_rate # Return empty signal

    # Concatenate all parts
    full_word_signal = np.concatenate(word_signal_parts)
    return full_word_signal, sample_rate

# --- ENHANCED: Feature Extraction Function for 8 Groups ---
def extract_signal_features(signal_data, sample_rate):
    """
    Extract a comprehensive set of features for more detailed signal categorization.
    """
    if signal_data is None or signal_data.size == 0:
        return None

    # Ensure signal is normalized
    signal_data = signal_data / (np.max(np.abs(signal_data)) + 1e-10)

    # --- 1. Peak Analysis Features ---
    # Find peaks with different height thresholds to capture different peak types
    peaks_high, _ = signal.find_peaks(np.abs(signal_data), height=0.5)  # High peaks
    peaks_med, _ = signal.find_peaks(np.abs(signal_data), height=0.3)   # Medium peaks
    peaks_low, _ = signal.find_peaks(np.abs(signal_data), height=0.1)   # Low peaks
    
    # Peak counts at different heights
    peak_count_high = len(peaks_high)
    peak_count_med = len(peaks_med)
    peak_count_low = len(peaks_low)
    
    # Peak density (peaks per time unit)
    signal_duration = len(signal_data) / sample_rate
    peak_density = peak_count_med / signal_duration if signal_duration > 0 else 0
    
    # If we have peaks, calculate width features
    if peak_count_med > 0:
        # Calculate peak widths at different heights
        widths_50 = signal.peak_widths(np.abs(signal_data), peaks_med, rel_height=0.5)[0]
        widths_25 = signal.peak_widths(np.abs(signal_data), peaks_med, rel_height=0.75)[0]
        widths_75 = signal.peak_widths(np.abs(signal_data), peaks_med, rel_height=0.25)[0]
        
        # Peak width statistics
        mean_width = np.mean(widths_50)
        width_std = np.std(widths_50) if len(widths_50) > 1 else 0
        width_ratio = np.mean(widths_25 / widths_75) if len(widths_75) > 0 and np.all(widths_75 > 0) else 1.0
    else:
        mean_width = 0
        width_std = 0
        width_ratio = 1.0
    
    # --- 2. Spectral Features ---
    try:
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=signal_data, sr=sample_rate)[0].mean()
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal_data, sr=sample_rate)[0].mean()
        
        # Spectral flatness (tonal vs. noisy)
        spectral_flatness = librosa.feature.spectral_flatness(y=signal_data)[0].mean()
        
        # Spectral rolloff (frequency below which 85% of spectrum energy)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=signal_data, sr=sample_rate)[0].mean()
    except Exception:
        spectral_centroid = 0
        spectral_bandwidth = 0
        spectral_flatness = 0
        spectral_rolloff = 0
    
    # --- 3. Waveform Shape Features ---
    # Zero crossing rate
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(signal_data).astype(int))))
    zero_crossing_rate = zero_crossings / len(signal_data)
    
    # RMS energy
    rms = np.sqrt(np.mean(signal_data**2))
    
    # Crest factor (peak to RMS ratio)
    crest_factor = np.max(np.abs(signal_data)) / (rms + 1e-10)
    
    # --- 4. Rhythm and Envelope Features ---
    # Compute signal envelope
    envelope = np.abs(signal.hilbert(signal_data))
    
    # Envelope statistics
    env_mean = np.mean(envelope)
    env_std = np.std(envelope)
    env_max = np.max(envelope)
    
    # Envelope roughness (higher = more irregular)
    env_roughness = np.std(np.diff(envelope)) * 100
    
    # --- 5. Harmonic Features ---
    try:
        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(signal_data)
        harmonic_ratio = np.sum(harmonic**2) / (np.sum(signal_data**2) + 1e-10)
        percussive_ratio = np.sum(percussive**2) / (np.sum(signal_data**2) + 1e-10)
    except Exception:
        harmonic_ratio = 0.5
        percussive_ratio = 0.5
    
    # Combine all features into one vector
    features = np.array([
        # Peak features
        peak_count_high,
        peak_count_med / max(1, peak_count_low),  # Ratio of medium to low peaks
        peak_density,
        mean_width,
        width_std,
        width_ratio,
        
        # Spectral features
        spectral_centroid,
        spectral_bandwidth,
        spectral_flatness,
        spectral_rolloff,
        
        # Waveform features
        zero_crossing_rate,
        rms,
        crest_factor,
        
        # Envelope features
        env_mean,
        env_std / (env_mean + 1e-10),  # Coefficient of variation of envelope
        env_roughness,
        
        # Harmonic features
        harmonic_ratio,
        percussive_ratio
    ])
    
    return features

def compute_mfcc_features(signal_data, sample_rate, n_mfcc=13, n_fft=512, hop_length=256):
    """
    Compute mean MFCC features for a given signal.
    Returns a 1D numpy array (the mean MFCC vector).
    """
    if signal_data is None or signal_data.size == 0:
        return None

    # Ensure signal is float32
    signal_data = signal_data.astype(np.float32)

    # Check for NaN/Inf before processing
    if np.any(np.isnan(signal_data)) or np.any(np.isinf(signal_data)):
        print(f"Warning: Signal contains NaN or Inf values before MFCC. Length={len(signal_data)}. Returning None.")
        return None

    try:
        # Compute MFCCs (returns shape: [n_mfcc, n_frames])
        mfccs = librosa.feature.mfcc(y=signal_data, sr=sample_rate, n_mfcc=n_mfcc,
                                     n_fft=n_fft, hop_length=hop_length)

        # Check for NaN/Inf *after* MFCC computation (can happen with silent frames etc.)
        if not np.all(np.isfinite(mfccs)):
             print(f"Warning: MFCC computation resulted in non-finite values. Signal length={len(signal_data)}. Returning None.")
             return None

        # Compute the mean of coefficients across frames (axis=1)
        # Results in a single vector of shape [n_mfcc]
        mean_mfccs = np.mean(mfccs, axis=1)

        return mean_mfccs

    except Exception as e:
        print(f"Error computing MFCC: {e}")
        print("Signal stats: len={}, min={:.2f}, max={:.2f}, mean={:.2f}".format(
            len(signal_data), np.min(signal_data), np.max(signal_data), np.mean(signal_data)
        ))
        return None

# --- Updated Categorization Logic for 8 Groups ---
def categorize_words(words_to_process, num_clusters=8, use_comprehensive=True):
    """
    Generates signals, computes features, clusters them into 8 groups, and returns groups.
    
    Parameters:
    - words_to_process: List of words to categorize
    - num_clusters: Number of clusters to form (default 8)
    - use_comprehensive: If True, use comprehensive signal features; if False, use MFCC features
    """
    audio_features = {}
    word_list_for_clustering = []
    
    print(f"Processing {len(words_to_process)} words using {'comprehensive' if use_comprehensive else 'MFCC'} features...")

    processed_count = 0
    for word in words_to_process:
        word_signal, sample_rate = generate_word_signal(word)

        if word_signal.size == 0:
            continue

        # Compute features based on selected method
        if use_comprehensive:
            features = extract_signal_features(word_signal, sample_rate)
        else:
            features = compute_mfcc_features(word_signal, sample_rate)

        # Check if features were computed successfully
        if features is not None and not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
            audio_features[word] = features
            word_list_for_clustering.append(word)

        processed_count += 1
        if (processed_count % 100 == 0):
             print(f"  Processed {processed_count}/{len(words_to_process)} words...")

    if not audio_features:
        print("No valid audio features could be computed for any word. Cannot categorize.")
        return {}

    print(f"\nSuccessfully computed features for {len(audio_features)} words.")

    # --- Prepare data for clustering ---
    feature_matrix = np.array([audio_features[word] for word in word_list_for_clustering])

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # --- Perform Clustering into 8 groups ---
    actual_num_clusters = min(num_clusters, len(word_list_for_clustering))
    if actual_num_clusters < 2:
        print(f"Only {len(word_list_for_clustering)} words with features, cannot form multiple clusters.")
        return {0: word_list_for_clustering} if len(word_list_for_clustering) == 1 else {}

    print(f"\nClustering features into {actual_num_clusters} groups using K-Means...")
    kmeans = KMeans(n_clusters=actual_num_clusters, random_state=42, n_init=10)
    kmeans.fit(scaled_features)

    # Get cluster assignments
    labels = kmeans.labels_

    # --- Group words based on clusters ---
    word_groups = {}
    for i, word in enumerate(word_list_for_clustering):
        cluster_label = labels[i]
        if cluster_label not in word_groups:
            word_groups[cluster_label] = []
        word_groups[cluster_label].append(word)

    # --- Analyze key characteristics of each cluster ---
    if use_comprehensive:
        # Get the cluster centers
        centers = kmeans.cluster_centers_
        
        # Map back to original feature space for interpretation
        centers_original = scaler.inverse_transform(centers)
        
        # Extract key characteristics for each cluster
        cluster_characteristics = {}
        
        for cluster_id in range(len(centers_original)):
            center = centers_original[cluster_id]
            
            # Extract key characteristics from the center vector
            peak_width = center[3]  # mean_width
            peak_count = center[1]  # peak count ratio
            spectral_flatness = center[8]
            harmonic_ratio = center[16]
            env_roughness = center[15]
            
            # Categorize peak width
            if peak_width < np.percentile(centers_original[:, 3], 25):
                width_category = "very thin peaks"
            elif peak_width < np.percentile(centers_original[:, 3], 50):
                width_category = "thin peaks"
            elif peak_width < np.percentile(centers_original[:, 3], 75):
                width_category = "thick peaks"
            else:
                width_category = "very thick peaks"
            
            # Categorize harmonic content
            if harmonic_ratio > np.percentile(centers_original[:, 16], 75):
                harmonic_category = "highly harmonic"
            elif harmonic_ratio > np.percentile(centers_original[:, 16], 50):
                harmonic_category = "moderately harmonic"
            else:
                harmonic_category = "less harmonic"
            
            # Combine characteristics
            characteristic = f"{width_category}, {harmonic_category}"
            
            # Add peak density if it's distinctive
            if peak_count > np.percentile(centers_original[:, 1], 75):
                characteristic += ", many peaks"
            elif peak_count < np.percentile(centers_original[:, 1], 25):
                characteristic += ", few peaks"
                
            # Add envelope roughness if it's distinctive
            if env_roughness > np.percentile(centers_original[:, 15], 75):
                characteristic += ", rough envelope"
            elif env_roughness < np.percentile(centers_original[:, 15], 25):
                characteristic += ", smooth envelope"
                
            cluster_characteristics[cluster_id] = characteristic
            
        return word_groups, cluster_characteristics
    
    return word_groups

# --- Main function ---
def main():
    input_word_file = "words_alpha.txt"
    max_words_to_process = 1000  # Limit sample size

    print(f"Reading words from '{input_word_file}'...")
    try:
        with open(input_word_file, 'r', encoding='utf-8') as file:
            all_words = [word.strip() for word in file.readlines() if word.strip()]
            all_words = [word.upper() for word in all_words if word.isalpha()]
    except FileNotFoundError:
        print(f"Error: Input word file '{input_word_file}' not found.")
        print("Please make sure 'words_alpha.txt' is in the same directory as the script.")
        print("You can often find this file online (search 'words_alpha.txt dictionary').")
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

    # --- Parameters ---
    num_clusters = 8    # Classify into 8 groups
    output_filename = "word_categories_8_groups.txt"
    
    # Use comprehensive feature set
    use_comprehensive = True

    # --- Perform categorization ---
    result = categorize_words(word_dictionary, num_clusters=num_clusters, use_comprehensive=use_comprehensive)
    
    if use_comprehensive:
        categorized_groups, cluster_characteristics = result
    else:
        categorized_groups = result
        cluster_characteristics = {}

    # --- Output the results to Text File ---
    print(f"\nSaving categorized groups to '{output_filename}'...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"--- Word Categories (k={num_clusters}) ---\n")
            f.write(f"Processed {len(word_dictionary)} words (sampled from {len(all_words)} total).\n")

            if not categorized_groups:
                f.write("\nNo categories were formed.\n")
            else:
                # Sort clusters by size (largest first)
                sorted_clusters = sorted(categorized_groups.items(), 
                                        key=lambda x: len(x[1]), reverse=True)
                
                for cluster_id, words_in_group in sorted_clusters:
                    sorted_words = sorted(words_in_group)
                    
                    # Include characteristics if available
                    characteristic = ""
                    if cluster_id in cluster_characteristics:
                        characteristic = f" - {cluster_characteristics[cluster_id]}"
                    
                    f.write(f"\nGroup {cluster_id + 1}{characteristic} ({len(words_in_group)} words):\n")
                    f.write(f"  {', '.join(sorted_words)}\n")
        print(f"Successfully saved categories to '{output_filename}'.")
    except IOError as e:
        print(f"\nError: Could not write to file '{output_filename}'.")
        print(f"Reason: {e}")


    print("\nCategorization finished.")

if __name__ == "__main__":
    main()

# Make sure to install librosa and scikit-learn:
# pip install librosa scikit-learn
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import sys # To exit gracefully if libraries are not found
import random # For sampling words

# --- Library Import Checks ---
try:
    import librosa
    import librosa.display # For potential future visualization
except ImportError:
    print("Error: librosa library not found.")
    print("Please install it using: pip install librosa")
    sys.exit(1)

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler # Optional but recommended
except ImportError:
    print("Error: scikit-learn library not found.")
    print("Please install it using: pip install scikit-learn")
    sys.exit(1)
# ---------------------------

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
        # print(f"Warning: No valid alphabetic characters found in the input word: '{word}'.") # Less verbose
        return np.array([]), sample_rate # Return empty signal

    # Concatenate all parts
    full_word_signal = np.concatenate(word_signal_parts)
    return full_word_signal, sample_rate

# --- >>> NEW: Feature Extraction using MFCC <<< ---
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

# --- >>> UPDATED: Main Categorization Logic using MFCC <<< ---
def categorize_words(words_to_process, n_mfcc=13, num_clusters=5):
    """
    Generates signals, computes MEAN MFCC features, clusters them, and returns groups.
    """
    audio_features = {}
    word_list_for_clustering = [] # Keep track of words corresponding to feature vectors
    feature_dimension = n_mfcc # Expected dimension of the feature vector

    print(f"Processing {len(words_to_process)} words...")

    processed_count = 0
    for i, word in enumerate(words_to_process):
        word_signal, sample_rate = generate_word_signal(word)

        if word_signal.size == 0:
            # print(f"Skipping '{word}' due to no valid signal.") # Less verbose
            continue

        # Compute MFCC features
        # Pass sample_rate which is needed for MFCC calculation
        mfcc_coeffs = compute_mfcc_features(word_signal, sample_rate, n_mfcc=n_mfcc)

        # Check if features were computed successfully and have the correct dimension
        if mfcc_coeffs is not None and mfcc_coeffs.size == feature_dimension :
            audio_features[word] = mfcc_coeffs
            word_list_for_clustering.append(word)
        # else:
            # print(f"Skipping '{word}' due to MFCC computation failure.") # Less verbose

        processed_count += 1
        if (processed_count % 100 == 0): # Print progress update every 100 words
             print(f"  Processed {processed_count}/{len(words_to_process)} words...")


    if not audio_features:
        print("No valid audio features could be computed for any word. Cannot categorize.")
        return {}

    print(f"\nSuccessfully computed MFCC features for {len(audio_features)} words.")

    # --- Prepare data for clustering ---
    # Create a matrix where each row is a mean MFCC feature vector
    feature_matrix = np.array([audio_features[word] for word in word_list_for_clustering])

    # Scale features (Recommended for K-Means, especially with MFCCs)
    print("Scaling features...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # --- Perform Clustering ---
    actual_num_clusters = min(num_clusters, len(word_list_for_clustering))
    if actual_num_clusters < 2:
         print(f"Only {len(word_list_for_clustering)} words with features, cannot form multiple clusters.")
         if len(word_list_for_clustering) == 1:
             return {0: word_list_for_clustering}
         else:
             return {}

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

    print("Clustering complete.")
    return word_groups

# --- >>> UPDATED: Main function with word sampling <<< ---
def main():
    """
    Main function to define words (sampling from file), run categorization, and save groups.
    """
    input_word_file = "words_alpha.txt"
    # --- >>> Limit the number of words processed <<< ---
    # Set to None to process all words (can be VERY slow and memory intensive)
    max_words_to_process = 1000  # Example: Process a sample of 1000 words

    print(f"Reading words from '{input_word_file}'...")
    try:
        with open(input_word_file, 'r', encoding='utf-8') as file:
            all_words = [word.strip() for word in file.readlines() if word.strip()] # Read lines, strip whitespace
            all_words = [word.upper() for word in all_words if word.isalpha()] # Keep only alpha words, convert to upper
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
    num_mfcc_coeffs = 13 # Number of MFCCs to compute (standard value)
    num_clusters = 10    # Desired number of groups (adjust as needed - maybe more needed for larger sample?)
    output_filename = "word_categories_mfcc.txt" # Output filename changed

    # --- Perform categorization using MFCC features ---
    categorized_groups = categorize_words(word_dictionary,
                                          n_mfcc=num_mfcc_coeffs,
                                          num_clusters=num_clusters)

    # --- Output the results to Console (Optional) ---
    # (Consider commenting out for large numbers of words/groups)
    # print("\n--- Word Categories (Console Output) ---")
    # if not categorized_groups:
    #     print("No categories were formed.")
    # else:
    #     for cluster_id, words_in_group in sorted(categorized_groups.items()):
    #         sorted_words = sorted(words_in_group)
    #         # Limit printing if group is too large
    #         display_limit = 20
    #         words_display = ', '.join(sorted_words[:display_limit])
    #         if len(sorted_words) > display_limit:
    #             words_display += f", ... ({len(sorted_words) - display_limit} more)"
    #         print(f"\nGroup {cluster_id + 1} ({len(words_in_group)} words):")
    #         print(f"  {words_display}")


    # --- Output the results to Text File ---
    print(f"\nSaving categorized groups to '{output_filename}'...")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(f"--- Word Categories based on Mean MFCC (k={num_clusters}) ---\n")
            f.write(f"Processed {len(word_dictionary)} words (sampled from {len(all_words)} total).\n")
            f.write(f"Number of features (MFCCs): {num_mfcc_coeffs}\n")

            if not categorized_groups:
                f.write("\nNo categories were formed.\n")
            else:
                # Sort clusters by ID
                for cluster_id, words_in_group in sorted(categorized_groups.items()):
                    sorted_words = sorted(words_in_group)
                    f.write(f"\nGroup {cluster_id + 1} ({len(words_in_group)} words):\n")
                    # Write all words, perhaps line-wrapped for large groups
                    f.write(f"  {', '.join(sorted_words)}\n") # Simple comma-separated for now
        print(f"Successfully saved categories to '{output_filename}'.")
    except IOError as e:
        print(f"\nError: Could not write to file '{output_filename}'.")
        print(f"Reason: {e}")

    print("\nCategorization finished.")


if __name__ == "__main__":
    main()

# Make sure to install librosa and scikit-learn:
# pip install librosa scikit-learn
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import sys # To exit gracefully if libraries are not found

# --- Library Import Checks ---
try:
    import librosa
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

    # print(f"Generating signal for word: '{word}'") # Reduced verbosity
    for letter in word:
        if 'A' <= letter.upper() <= 'Z':
            # print(f"  - Generating signal for letter: {letter.upper()}") # Reduced verbosity
            signal_part, sr = generate_signal(letter.upper())
            word_signal_parts.append(signal_part)
            sample_rate = sr # Ensure consistent sample rate
            valid_letters += 1
        # else:
            # print(f"  - Skipping non-alphabetic character: {letter}") # Reduced verbosity

    if not word_signal_parts:
        print(f"Warning: No valid alphabetic characters found in the input word: '{word}'.")
        return np.array([]), sample_rate # Return empty signal

    # Concatenate all parts
    full_word_signal = np.concatenate(word_signal_parts)
    # print(f"Generated signal of length {len(full_word_signal)} samples for {valid_letters} letters.") # Reduced verbosity
    return full_word_signal, sample_rate


def compute_lpc_coefficients(signal_data, order=16):
    """
    Compute LPC coefficients using librosa's implementation.
    Returns coefficients a[1] through a[p]. Checks for non-finite values.
    """
    if signal_data is None or len(signal_data) <= order:
        # print(f"Warning: Signal length ({len(signal_data) if signal_data is not None else 0}) is too short for LPC order ({order}). Returning None.") # Less verbose
        return None # Return None to indicate failure

    # Ensure signal is float32, librosa expects this
    signal_data = signal_data.astype(np.float32)

    # Check for NaNs or Infs *before* LPC computation
    if np.any(np.isnan(signal_data)) or np.any(np.isinf(signal_data)):
         print(f"Warning: Signal contains NaN or Inf values before LPC. Length={len(signal_data)}. Returning None.")
         return None

    # Normalize signal amplitude (helps stability)
    max_abs_val = np.max(np.abs(signal_data))
    if max_abs_val > 1e-9: # Avoid division by zero for silence
         signal_data = signal_data / max_abs_val
    else:
        # If signal is essentially silent, LPC is likely problematic/undefined
        # print(f"Warning: Signal is near silent (max_abs={max_abs_val}). LPC may fail. Returning None.")
        return None # Treat silent signals as LPC failure

    # Compute LPC coefficients using librosa
    try:
        # librosa.lpc returns the coefficients including a_0=1.
        lpc_coeffs_full = librosa.lpc(signal_data, order=order)

        # --- >>> ADDED CHECK FOR FINITE VALUES <<< ---
        # Check if LPC returned NaNs OR Infs
        if not np.all(np.isfinite(lpc_coeffs_full)):
            print(f"Warning: LPC computation resulted in non-finite (NaN/Inf) values. Signal length={len(signal_data)}. Returning None.")
            # You could optionally print the problematic coefficients here for debugging:
            # print(f"Problematic coeffs: {lpc_coeffs_full}")
            return None
        # --- >>> END OF ADDED CHECK <<< ---

        # Return only a_1 to a_order.
        return lpc_coeffs_full[1:]

    except Exception as e:
        # Catch potential errors within librosa.lpc itself
        print(f"Error computing LPC: {e}")
        print("Signal stats: len={}, min={:.2f}, max={:.2f}, mean={:.2f}".format(
            len(signal_data), np.min(signal_data), np.max(signal_data), np.mean(signal_data)
        ))
        return None # Return None on error

# --- Main Categorization Logic ---
def categorize_words(words_to_process, lpc_order=16, num_clusters=5):
    """
    Generates signals, computes LPC, clusters them, and returns groups.
    """
    lpc_features = {}
    word_list_for_clustering = [] # Keep track of words corresponding to feature vectors

    print(f"Processing {len(words_to_process)} words...")

    for word in words_to_process:
        word_signal, sample_rate = generate_word_signal(word)

        if word_signal.size == 0:
            print(f"Skipping '{word}' due to no valid signal.")
            continue

        # Compute LPC coefficients
        lpc_coeffs = compute_lpc_coefficients(word_signal, order=lpc_order)

        if lpc_coeffs is not None and lpc_coeffs.size == lpc_order : # Ensure valid coefficients were returned
            lpc_features[word] = lpc_coeffs
            word_list_for_clustering.append(word)
        else:
            print(f"Skipping '{word}' due to LPC computation failure.")

    if not lpc_features:
        print("No valid LPC features could be computed for any word. Cannot categorize.")
        return {}

    print(f"\nSuccessfully computed LPC features for {len(lpc_features)} words.")

    # --- Prepare data for clustering ---
    # Create a matrix where each row is an LPC feature vector
    feature_matrix = np.array([lpc_features[word] for word in word_list_for_clustering])

    # Optional: Scale features (often improves K-Means performance)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    # --- Perform Clustering ---
    # Determine the number of clusters (can be tricky, might need tuning)
    # If num_clusters > number of samples, KMeans will fail.
    actual_num_clusters = min(num_clusters, len(word_list_for_clustering))
    if actual_num_clusters < 2:
         print(f"Only {len(word_list_for_clustering)} words with features, cannot form multiple clusters.")
         # Return a single group if only one word/feature vector exists
         if len(word_list_for_clustering) == 1:
             return {0: word_list_for_clustering}
         else:
             return {} # Or handle as needed if zero

    print(f"\nClustering features into {actual_num_clusters} groups using K-Means...")
    kmeans = KMeans(n_clusters=actual_num_clusters, random_state=42, n_init=10) # Set n_init explicitly
    # Use scaled_features if you used the scaler, otherwise feature_matrix
    kmeans.fit(scaled_features) # or kmeans.fit(feature_matrix)

    # Get cluster assignments for each word
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

def main():
    """
    Main function to define words and run the categorization.
    """
    # --- Define your dictionary of words here ---

    with open("words_alpha.txt", 'r', encoding='utf-8') as file:
        word_dictionary = file.read().upper().split()
    lpc_order = 16  # LPC analysis order (feature vector dimension)
    num_clusters = 5 # Desired number of groups (adjust as needed)
    output_filename = "word_categories.txt" # <<< Name of the output file

    # Perform categorization
    categorized_groups = categorize_words(word_dictionary, lpc_order, num_clusters)

    # --- Output the results to Console (Optional - you can comment this out if desired) ---
    print("\n--- Word Categories (Console Output) ---")
    if not categorized_groups:
        print("No categories were formed.")
    else:
        # Loop through sorted cluster IDs
        for cluster_id, words_in_group in sorted(categorized_groups.items()):
            # Sort words within the group alphabetically
            sorted_words = sorted(words_in_group)
            print(f"\nGroup {cluster_id + 1}:")
            print(f"  {', '.join(sorted_words)}")

    # --- >>> Output the results to Text File <<< ---
    print(f"\nSaving categorized groups to '{output_filename}'...")
    try:
        # Use 'with open' to ensure the file is properly closed
        # 'w' mode overwrites the file if it exists
        # encoding='utf-8' is good practice for text files
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("--- Word Categories ---\n") # Write a header to the file
            if not categorized_groups:
                f.write("\nNo categories were formed.\n")
            else:
                # Loop through sorted cluster IDs (same as console output)
                for cluster_id, words_in_group in sorted(categorized_groups.items()):
                    # Sort words within the group alphabetically
                    sorted_words = sorted(words_in_group)
                    # Write the group information to the file
                    # Add '\n' for newlines where needed
                    f.write(f"\nGroup {cluster_id + 1}:\n")
                    f.write(f"  {', '.join(sorted_words)}\n")
        print(f"Successfully saved categories to '{output_filename}'.") # Confirmation message
    except IOError as e:
        # Basic error handling if the file cannot be written
        print(f"\nError: Could not write to file '{output_filename}'.")
        print(f"Reason: {e}")
    # --- >>> End of File Writing Section <<< ---

    print("\nCategorization finished.")

if __name__ == "__main__":
    main()

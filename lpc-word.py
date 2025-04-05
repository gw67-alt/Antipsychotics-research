# Make sure to install librosa: pip install librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa # Import librosa for LPC
import sys # To exit gracefully if librosa is not found

# Check for librosa installation
try:
    import librosa
except ImportError:
    print("Error: librosa library not found.")
    print("Please install it using: pip install librosa")
    sys.exit(1)


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

    print(f"Generating signal for word: '{word}'")
    for letter in word:
        if 'A' <= letter.upper() <= 'Z':
            print(f"  - Generating signal for letter: {letter.upper()}")
            signal_part, sr = generate_signal(letter.upper())
            word_signal_parts.append(signal_part)
            sample_rate = sr # Ensure consistent sample rate
            valid_letters += 1
        else:
            print(f"  - Skipping non-alphabetic character: {letter}")
            # Optionally add silence for non-letters:
            # signal_part, sr = generate_signal('!%') # Will generate silence
            # word_signal_parts.append(signal_part)
            # sample_rate = sr

    if not word_signal_parts:
        print("Warning: No valid alphabetic characters found in the input.")
        return np.array([]), sample_rate # Return empty signal

    # Concatenate all parts
    full_word_signal = np.concatenate(word_signal_parts)
    print(f"Generated signal of length {len(full_word_signal)} samples for {valid_letters} letters.")
    return full_word_signal, sample_rate


def compute_lpc_coefficients(signal_data, order=16):
    """
    Compute LPC coefficients using librosa's implementation.
    Returns coefficients a[1] through a[p].
    """
    if len(signal_data) <= order:
        print(f"Warning: Signal length ({len(signal_data)}) is too short for LPC order ({order}). Returning zeros.")
        return np.zeros(order)

    # Normalize signal amplitude (optional but often helpful)
    # signal_data = signal_data / np.max(np.abs(signal_data) + 1e-9)

    # Compute LPC coefficients using librosa
    try:
        # librosa.lpc returns the coefficients including a_0.
        lpc_coeffs_full = librosa.lpc(signal_data.astype(np.float32), order=order)
        # Return only a_1 to a_order.
        return lpc_coeffs_full[1:]
    except Exception as e:
        # Common errors include signal being too short or containing NaNs/Infs
        print(f"Error computing LPC: {e}")
        print("Signal stats: len={}, min={:.2f}, max={:.2f}, mean={:.2f}".format(
            len(signal_data), np.min(signal_data), np.max(signal_data), np.mean(signal_data)
        ))
        # Check for NaNs or Infs which can cause errors
        if np.any(np.isnan(signal_data)) or np.any(np.isinf(signal_data)):
            print("Signal contains NaN or Inf values.")
        return np.zeros(order) # Return zeros or handle error appropriately

def visualize_word_lpc(word, lpc_coeffs, sample_rate, lpc_order):
    """
    Visualize the magnitude spectrum derived from LPC coefficients for a word.
    """
    if lpc_coeffs is None or len(lpc_coeffs) != lpc_order:
         print("Cannot visualize: Invalid LPC coefficients provided.")
         return

    plt.figure(figsize=(10, 6))

    # Construct the full denominator polynomial A(z) coefficients [1, a_1, ..., a_p]
    a_coeffs = np.concatenate(([1], lpc_coeffs))

    # Compute frequency response of the LPC filter H(z) = 1 / A(z)
    w, h = signal.freqz(1, a_coeffs, worN=4096, fs=sample_rate) # Use fs for Hz axis

    # Compute magnitude spectrum in dB
    magnitudes_db = 20 * np.log10(np.abs(h) + 1e-9) # Add epsilon for stability

    # Plot
    plt.plot(w, magnitudes_db)
    plt.title(f"LPC Spectrum for '{word}' (Order {lpc_order})")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)
    plt.ylim(bottom=np.percentile(magnitudes_db, 1) - 10 if len(magnitudes_db)>0 else -80,
             top=np.percentile(magnitudes_db, 99) + 10 if len(magnitudes_db)>0 else 40) # Dynamic Y limits
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to get user input and run the analysis.
    """
    lpc_order = 16 # You can change the LPC order here

    while True:
        try:
            user_word = input("Enter a word to analyze (or press Enter to quit): ")
            if not user_word:
                break # Exit loop if user presses Enter

            # Generate the combined signal for the word
            word_signal, sample_rate = generate_word_signal(user_word)

            if word_signal.size == 0:
                print("Cannot proceed without a valid signal.")
                continue # Ask for input again

            # Compute LPC coefficients for the entire word signal
            print(f"Computing LPC coefficients (order={lpc_order})...")
            lpc_coeffs = compute_lpc_coefficients(word_signal, order=lpc_order)

            # Check if coefficients were computed successfully
            if np.any(lpc_coeffs): # Check if not all zeros (or None)
                # Visualize the results
                visualize_word_lpc(user_word, lpc_coeffs, sample_rate, lpc_order)
            else:
                print("LPC computation failed or resulted in zeros. Cannot visualize.")

        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Please try again.")

    print("\nExiting program.")

if __name__ == "__main__":
    main()
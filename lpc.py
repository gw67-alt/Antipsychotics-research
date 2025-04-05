# Make sure to install librosa: pip install librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import librosa # Import librosa for LPC

def generate_signal(letter):
    """
    Generate a basic signal for each letter
    Returns the signal data and the sample rate used.
    """
    # Basic parameters
    sample_rate = 4410  # Reduced sample rate
    duration = 0.5  # Half a second
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Base frequencies for each letter
    frequencies = {
        'A': 220, 'B': 240, 'C': 260, 'D': 280, 'E': 300,
        'F': 320, 'G': 340, 'H': 360, 'I': 380, 'J': 400,
        'K': 420, 'L': 440, 'M': 460, 'N': 480, 'O': 500,
        'P': 520, 'Q': 540, 'R': 560, 'S': 580, 'T': 600,
        'U': 620, 'V': 640, 'W': 660, 'X': 680, 'Y': 700, 'Z': 720
    }

    # Get frequency for the letter
    freq = frequencies.get(letter.upper(), 220) # Default to 'A' if not found

    # Generate signal with some complexity
    signal_data = np.sin(2 * np.pi * freq * t)
    signal_data += 0.5 * np.sin(2 * np.pi * (freq * 2) * t) # Add a harmonic

    # Apply envelope (Hanning window)
    envelope = np.hanning(len(signal_data))
    signal_data = signal_data * envelope

    return signal_data, sample_rate

def compute_lpc_coefficients(signal_data, order=16):
    """
    Compute LPC coefficients using librosa's implementation.
    Note: librosa.lpc returns coefficients a[1] through a[p].
          The full polynomial A(z) = 1 + a[1]z^-1 + ... + a[p]z^-p.
    """
    # Normalize signal (optional but often good practice for LPC)
    # signal_data = signal_data / np.max(np.abs(signal_data))

    # Compute LPC coefficients using librosa
    # librosa.lpc returns the coefficients starting from a_1
    try:
        lpc_coeffs = librosa.lpc(signal_data, order=order)
        # The coefficients returned by librosa.lpc are [a_1, a_2, ..., a_order].
        # We usually want the coefficients including a_0=1 for freqz,
        # but librosa returns excluding a_0. We'll handle this in visualize function.
        # Return only a_1 to a_order.
        return lpc_coeffs[1:]
    except Exception as e:
        print(f"Error computing LPC: {e}")
        # Return None or an empty array if LPC fails
        return np.zeros(order)


def visualize_alphabet_lpc():
    """
    Visualize the magnitude spectrum derived from LPC coefficients for the alphabet.
    """
    plt.figure(figsize=(20, 15))
    plt.suptitle("LPC-Derived Spectrum Magnitude for the Alphabet", fontsize=18, y=0.98)

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lpc_order = 16 # Define LPC order

    for idx, letter in enumerate(alphabet, 1):
        # Generate signal
        signal_data, sample_rate = generate_signal(letter)

        # Compute LPC coefficients (a_1 to a_p)
        lpc_coeffs_partial = compute_lpc_coefficients(signal_data, order=lpc_order)

        # Construct the full denominator polynomial A(z) coefficients [1, a_1, ..., a_p]
        # Note the negative sign convention difference:
        # Some LPC definitions use A(z) = 1 - sum(a_k * z^-k)
        # `librosa.lpc` and `scipy.signal.freqz` use A(z) = 1 + sum(a_k * z^-k)
        # The coefficients from librosa match the convention needed by freqz.
        a_coeffs = np.concatenate(([1], lpc_coeffs_partial))

        # Compute frequency response of the LPC filter H(z) = 1 / A(z)
        # We pass [1] as the numerator (b) and a_coeffs as the denominator (a)
        w, h = signal.freqz(1, a_coeffs, worN=2048, fs=sample_rate) # Use fs for Hz axis

        # Compute magnitude spectrum in dB
        # Add small epsilon to avoid log10(0)
        magnitudes_db = 20 * np.log10(np.abs(h) + 1e-9)

        # Plot
        plt.subplot(6, 5, idx)
        plt.plot(w, magnitudes_db) # w is now in Hz because we used fs in freqz
        plt.title(f'Letter {letter}')
        if idx > 21: # Add x-label only to bottom plots
             plt.xlabel('Frequency [Hz]')
        if idx % 5 == 1: # Add y-label only to left-most plots
            plt.ylabel('Magnitude [dB]')
        plt.grid(True)
        plt.ylim(bottom=np.min(magnitudes_db[np.isfinite(magnitudes_db)]) - 10 if np.any(np.isfinite(magnitudes_db)) else -60,
                 top=np.max(magnitudes_db[np.isfinite(magnitudes_db)]) + 10 if np.any(np.isfinite(magnitudes_db)) else 40) # Adjust y-limits dynamically

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    # plt.subplots_adjust(top=0.92) # Keep if needed with rect adjustment
    plt.show()

def main():
    visualize_alphabet_lpc()

if __name__ == "__main__":
    main()
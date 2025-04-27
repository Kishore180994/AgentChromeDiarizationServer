import numpy as np
import soundfile as sf
from scipy.io import wavfile


def create_test_audio(filename="wave_16k.wav", duration=60, sample_rate=16000):
    """
    Create a test audio file with different speakers.

    Args:
        filename: Output filename
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
    """
    print(f"Creating test audio file: {filename}")
    # Total number of samples
    total_samples = duration * sample_rate

    # Create silence
    audio = np.zeros(total_samples, dtype=np.float32)

    # Create two distinct voice patterns (simple sine waves with different frequencies)

    # Speaker 1: Lower pitch voice (200 Hz fundamental)
    speaker1_freq = 200
    # Speaker 2: Higher pitch voice (320 Hz fundamental)
    speaker2_freq = 320

    # Generate time array
    t = np.arange(total_samples) / sample_rate

    # Define speaker segments
    segments = [
        # (start_time, end_time, speaker)
        (0, 5, 1),      # Speaker 1: 0-5s
        (6, 10, 2),     # Speaker 2: 6-10s
        (11, 15, 1),    # Speaker 1: 11-15s
        (16, 20, 2),    # Speaker 2: 16-20s
        (21, 25, 1),    # Speaker 1: 21-25s
        (26, 30, 2),    # Speaker 2: 26-30s
        (31, 38, 1),    # Speaker 1: 31-38s
        (39, 46, 2),    # Speaker 2: 39-46s
        (47, 52, 1),    # Speaker 1: 47-52s
        (53, 59, 2)     # Speaker 2: 53-59s
    ]

    # Add the speaker segments to the audio
    for start, end, speaker in segments:
        start_idx = start * sample_rate
        end_idx = end * sample_rate

        # Choose frequency based on speaker
        freq = speaker1_freq if speaker == 1 else speaker2_freq

        # Create a simple sine wave with harmonics
        segment = np.zeros(end_idx - start_idx, dtype=np.float32)

        # Add fundamental frequency
        segment += 0.5 * np.sin(2 * np.pi * freq * t[start_idx:end_idx])

        # Add some harmonics
        segment += 0.3 * np.sin(2 * np.pi * freq * 2 * t[start_idx:end_idx])
        segment += 0.2 * np.sin(2 * np.pi * freq * 3 * t[start_idx:end_idx])

        # Add some formants (to make it sound more voice-like)
        segment += 0.1 * \
            np.sin(2 * np.pi * (freq * 4 + (speaker * 100))
                   * t[start_idx:end_idx])
        segment += 0.05 * \
            np.sin(2 * np.pi * (freq * 5 + (speaker * 150))
                   * t[start_idx:end_idx])

        # Add some noise to simulate speech variations
        segment += np.random.normal(0, 0.05, size=len(segment))

        # Apply envelope to simulate speech amplitude variations
        envelope = np.ones_like(segment)
        # Fade in
        fade_samples = min(int(0.1 * sample_rate), len(segment) // 10)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        # Fade out
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        # Apply some amplitude modulation to simulate syllables
        syllable_rate = 3  # syllables per second
        envelope *= 0.7 + 0.3 * \
            np.sin(2 * np.pi * syllable_rate * t[start_idx:end_idx])

        # Apply envelope to segment
        segment *= envelope

        # Add to main audio
        audio[start_idx:end_idx] = segment

    # Normalize audio
    audio = audio / np.max(np.abs(audio)) * 0.9

    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    # Write to file
    wavfile.write(filename, sample_rate, audio_int16)
    print(
        f"Created test audio file: {filename} ({duration} seconds, {sample_rate} Hz)")

    # Also create a copy as wave.wav
    wavfile.write("wave.wav", sample_rate, audio_int16)
    print(f"Also created wave.wav as a copy")


if __name__ == "__main__":
    print("Starting test audio file creation...")
    try:
        create_test_audio()
        print("Audio file creation completed successfully")

        # Check if the files were created
        import os
        for filename in ["wave_16k.wav", "wave.wav"]:
            if os.path.exists(filename):
                print(
                    f"File exists: {filename}, size: {os.path.getsize(filename)} bytes")
            else:
                print(f"File does not exist: {filename}")
    except Exception as e:
        print(f"Error creating audio files: {e}")

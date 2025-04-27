import os
import sys

# Redirect stdout to a file
log_file = open("audio_check.log", "w")
sys.stdout = log_file

print("Checking for audio files...")

# Check for wave_16k.wav
if os.path.exists("wave_16k.wav"):
    print(
        f"wave_16k.wav exists, size: {os.path.getsize('wave_16k.wav')} bytes")
else:
    print("wave_16k.wav does not exist")

# Check for wave.wav
if os.path.exists("wave.wav"):
    print(f"wave.wav exists, size: {os.path.getsize('wave.wav')} bytes")
else:
    print("wave.wav does not exist")

# Try to create a simple test audio file
try:
    import numpy as np
    from scipy.io import wavfile

    print("Attempting to create a simple test audio file...")

    # Create a simple sine wave
    sample_rate = 16000
    duration = 5  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

    # Normalize to 16-bit range and convert to int16
    tone = (tone * 32767).astype(np.int16)

    # Write to file
    test_file = "test_tone.wav"
    wavfile.write(test_file, sample_rate, tone)

    if os.path.exists(test_file):
        print(
            f"Successfully created {test_file}, size: {os.path.getsize(test_file)} bytes")
    else:
        print(f"Failed to create {test_file}")

except Exception as e:
    print(f"Error creating test audio: {e}")

# List all files in the current directory
print("\nListing all files in current directory:")
for file in os.listdir("."):
    print(f"- {file} ({os.path.getsize(file)} bytes)")

# Close the log file
log_file.close()

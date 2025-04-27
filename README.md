# Speaker Diarization with Transcription Backend

This project provides a backend service for speaker diarization with transcription capabilities. It processes audio streams to identify different speakers and transcribe the spoken content.

## Features

- Real-time speaker diarization using PyAnnote Audio
- Speech-to-text transcription using Whisper
- WebSocket-based API for streaming audio processing
- Support for real-time audio streaming from clients
- Debug audio storage for troubleshooting and analysis

## Setup Instructions

### Prerequisites

```
brew install python@3.10
brew install cmake
brew install coreutils (for installing nproc)
brew install pkg-config
```

### Installation

1. Create a virtual environment with Python 3.10:

   ```
   python3.10 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```
   pip install fastapi uvicorn websockets numpy torch
   pip install pyannote.audio
   pip install transformers
   pip install python-dotenv
   pip install sentence-transformers  # This will compile and install sentencepiece successfully
   ```

3. Create a `.env` file with your Hugging Face token and debug configuration:

   ```
   HUGGINGFACE_TOKEN="your_huggingface_token"

   # Debug audio storage configuration
   DEBUG_AUDIO_STORAGE=true
   DEBUG_AUDIO_DIR=debug_audio
   ```

## Running the Server

Start the server with:

```
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Testing

Use the included test client to test the service:

```
python test_client.py
```

Make sure to update the `AUDIO_FILE_PATH` in the test client to point to a valid WAV file.

## API Response Format

The service returns JSON responses with the following structures:

### Diarization Update

Sent when speaker segments are identified in a small audio chunk:

```json
{
  "type": "diarization_update",
  "segments": [
    {
      "speaker": "SPEAKER_01",
      "start": 0.123,
      "end": 2.456
    },
    ...
  ],
  "transcription": ""  // Usually empty as transcription is sent separately
}
```

### Speaker Transcripts

Sent when a larger audio buffer has been processed for speech recognition and matched with speaker segments:

```json
{
  "type": "speaker_transcripts",
  "full_transcription": "This is the full transcribed text from the larger audio buffer.",
  "start_time": 0.0,
  "end_time": 30.0,
  "speaker_segments": [
    {
      "speaker": "SPEAKER_01",
      "text": "This is the",
      "start": 0.5,
      "end": 1.2
    },
    {
      "speaker": "SPEAKER_02",
      "text": "full transcribed text",
      "start": 1.3,
      "end": 2.7
    },
    {
      "speaker": "SPEAKER_01",
      "text": "from the larger audio buffer.",
      "start": 2.8,
      "end": 4.5
    }
  ],
  "speaker_transcripts": {
    "SPEAKER_01": {
      "text": "This is the from the larger audio buffer.",
      "segments": [
        {
          "speaker": "SPEAKER_01",
          "text": "This is the",
          "start": 0.5,
          "end": 1.2
        },
        {
          "speaker": "SPEAKER_01",
          "text": "from the larger audio buffer.",
          "start": 2.8,
          "end": 4.5
        }
      ]
    },
    "SPEAKER_02": {
      "text": "full transcribed text",
      "segments": [
        {
          "speaker": "SPEAKER_02",
          "text": "full transcribed text",
          "start": 1.3,
          "end": 2.7
        }
      ]
    }
  }
}
```

This response format provides:

- The full transcription of the audio
- A list of all segments with speaker, text, and timestamps
- A grouped view of transcriptions by speaker, with each speaker's complete text and individual segments

### Transcription Only

Sent if no speaker segments are found but transcription is available:

```json
{
  "type": "transcription_only",
  "transcription": "This is the transcribed text from the audio."
}
```

The service processes audio in two ways:

1. Small chunks (1-2 seconds) for real-time speaker diarization
2. Larger chunks (30 seconds) for more accurate speech recognition

This dual-buffer approach provides both timely speaker identification and high-quality transcription. The larger buffer for speech recognition ensures more complete and accurate transcriptions by providing more context to the speech recognition model.

## Implementation Notes

- The speech recognition model (Whisper) is configured to always transcribe in English
- The service handles warnings and potential issues with the Whisper model by:
  - Using the correct parameter format for the Whisper model
  - Setting the language explicitly to English via the generate_kwargs parameter
  - Setting the task to "transcribe" rather than translate

## Debug Audio Storage

The service includes a feature to store received audio chunks for debugging purposes:

- Audio chunks are saved as WAV files in the configured debug directory
- Each file is named with the client ID and timestamp for easy identification
- This feature can be enabled/disabled via the `DEBUG_AUDIO_STORAGE` environment variable
- The storage directory can be configured via the `DEBUG_AUDIO_DIR` environment variable

To disable debug audio storage, set `DEBUG_AUDIO_STORAGE=false` in your `.env` file.

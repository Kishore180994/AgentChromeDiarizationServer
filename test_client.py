import asyncio
import websockets
import numpy as np
import time
import logging
import json
import soundfile as sf  # Import soundfile

# Configure logging for the client
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebSocketClient")

# --- Configuration ---
SERVER_URI = "ws://localhost:8000/ws/diarize"
# Use the properly formatted mono audio file
AUDIO_FILE_PATH = "wave_16k.wav"  # Using the properly resampled 16kHz mono file
# Must match server's expected rate (16000 Hz as defined in main.py)
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2     # Must match server (2 bytes for 16-bit)
CHANNELS = 1         # Must match server (mono) AND your audio file
CHUNK_DURATION_S = 30  # Duration of each audio chunk to send
# SIMULATION_DURATION_S is no longer needed, we send the whole file

# Calculate chunk size in samples
CHUNK_SAMPLES = int(CHUNK_DURATION_S * SAMPLE_RATE)


async def send_receive(uri):
    """Connects, streams audio from a file, and prints received messages."""
    logger.info(f"Attempting to connect to {uri}...")
    logger.info(f"DEBUG_AUDIO_STORAGE should be enabled in .env")
    try:
        # --- Read Audio File ---
        try:
            logger.info(f"Reading audio file: {AUDIO_FILE_PATH}")
            audio_data, file_sample_rate = sf.read(
                AUDIO_FILE_PATH, dtype='int16')

            # Validate audio format
            if file_sample_rate != SAMPLE_RATE:
                logger.error(
                    f"Audio file sample rate ({file_sample_rate}Hz) doesn't match expected rate ({SAMPLE_RATE}Hz). Please convert the file.")
                return
            # Check if mono (if stereo, convert to mono by averaging channels)
            if len(audio_data.shape) > 1 and audio_data.shape[1] != CHANNELS:
                logger.warning(
                    f"Audio file has {audio_data.shape[1]} channels, expected {CHANNELS}. Converting to mono by averaging channels.")
                if len(audio_data.shape) > 1:
                    # Average all channels instead of just taking the first one
                    audio_data = np.mean(audio_data, axis=1).astype(np.int16)
                    # Increase volume by 50% to make speaker detection easier
                    audio_data = np.clip(
                        audio_data * 1.5, -32768, 32767).astype(np.int16)
                    logger.info(
                        f"Converted to mono and increased volume. New shape: {audio_data.shape}")
                else:
                    logger.error("Cannot determine channel count correctly.")
                    return
            elif len(audio_data.shape) == 1 and CHANNELS != 1:
                logger.error(
                    f"Audio file is mono, but {CHANNELS} channels expected.")
                return

            logger.info(
                f"Successfully read audio file. Duration: {len(audio_data) / SAMPLE_RATE:.2f}s")

        except FileNotFoundError:
            logger.error(f"Audio file not found: {AUDIO_FILE_PATH}")
            return
        except Exception as e:
            logger.error(f"Error reading audio file: {e}", exc_info=True)
            return

        # --- Connect and Stream ---
        async with websockets.connect(uri) as websocket:
            logger.info("Connection established!")

            async def receive_messages():
                # (Keep the receive_messages function as before)
                try:
                    async for message in websocket:
                        # Parse the message to handle different response types
                        try:
                            msg_data = json.loads(message)
                            msg_type = msg_data.get("type", "unknown")

                            if msg_type == "diarization_update":
                                segments = msg_data.get("segments", [])
                                logger.info(
                                    f"<<< Received diarization update with {len(segments)} segments")
                                logger.info(f"<<< Segments: {segments}")
                            elif msg_type == "transcription_only":
                                transcription = msg_data.get(
                                    "transcription", "")
                                logger.info(
                                    f"<<< Received transcription only: {transcription}")
                            elif msg_type == "transcription_update":
                                transcription = msg_data.get(
                                    "transcription", "")
                                start_time = msg_data.get("start_time", 0)
                                end_time = msg_data.get("end_time", 0)
                                logger.info(
                                    f"<<< Received transcription update for time range {start_time:.2f}s - {end_time:.2f}s")
                                logger.info(
                                    f"<<< Full transcription: {transcription}")
                            elif msg_type == "speaker_transcripts":
                                full_transcription = msg_data.get(
                                    "full_transcription", "")
                                start_time = msg_data.get("start_time", 0)
                                end_time = msg_data.get("end_time", 0)
                                speaker_segments = msg_data.get(
                                    "speaker_segments", [])
                                speaker_transcripts = msg_data.get(
                                    "speaker_transcripts", {})

                                logger.info(
                                    f"<<< Received speaker transcripts for time range {start_time:.2f}s - {end_time:.2f}s")
                                logger.info(
                                    f"<<< Full transcription: {full_transcription}")

                                # Print detailed information about each segment
                                logger.info(f"<<< Speaker segments with text:")
                                for segment in speaker_segments:
                                    speaker = segment.get("speaker", "UNKNOWN")
                                    text = segment.get("text", "")
                                    start = segment.get("start", 0)
                                    end = segment.get("end", 0)
                                    logger.info(
                                        f"<<< {speaker} ({start:.2f}s - {end:.2f}s): {text}")

                                # Print transcriptions by speaker
                                logger.info(
                                    f"<<< Speaker transcripts summary:")
                                for speaker, data in speaker_transcripts.items():
                                    speaker_text = data.get("text", "")
                                    segments = data.get("segments", [])
                                    logger.info(
                                        f"<<< {speaker}: {speaker_text}")

                                    # Print each segment for this speaker
                                for segment in segments:
                                    if isinstance(segment, dict):
                                        seg_text = segment.get("text", "")
                                        seg_start = segment.get("start", 0)
                                        seg_end = segment.get("end", 0)
                                        logger.info(
                                            f"<<<   - ({seg_start:.2f}s - {seg_end:.2f}s): {seg_text}")
                            elif msg_type == "speaker_transcription_update":
                                segments = msg_data.get("segments", [])
                                logger.info(
                                    f"<<< Received speaker transcription update with {len(segments)} segments")

                                # Print detailed information about each segment
                                for segment in segments:
                                    speaker = segment.get("speaker", "UNKNOWN")
                                    text = segment.get("text", "")
                                    start = segment.get("start", 0)
                                    end = segment.get("end", 0)
                                    logger.info(
                                        f"<<< {speaker} ({start:.2f}s - {end:.2f}s): {text}")
                            else:
                                logger.info(
                                    f"<<< Received unknown message type: {msg_type}")
                                logger.info(f"<<< Full message: {message}")
                        except json.JSONDecodeError:
                            logger.warning(
                                f"<<< Received non-JSON message: {message}")
                        except Exception as e:
                            logger.error(f"<<< Error processing message: {e}")
                            logger.info(f"<<< Original message: {message}")
                except websockets.exceptions.ConnectionClosedOK:
                    logger.info("Server closed the connection normally.")
                except websockets.exceptions.ConnectionClosedError as e:
                    logger.error(f"Connection closed with error: {e}")
                except Exception as e:
                    logger.error(
                        f"Error receiving message: {e}", exc_info=True)

            receive_task = asyncio.create_task(receive_messages())

            # --- Send audio file in chunks ---
            total_samples = len(audio_data)
            start_sample = 0
            while start_sample < total_samples:
                end_sample = start_sample + CHUNK_SAMPLES
                chunk_data = audio_data[start_sample:end_sample]
                chunk_bytes = chunk_data.tobytes()  # Convert numpy slice to bytes

                logger.info(
                    f">>> Sending {len(chunk_bytes)} bytes of audio...")
                logger.info(
                    f">>> This should trigger debug audio storage on the server")
                await websocket.send(chunk_bytes)

                start_sample = end_sample
                # Simulate real-time streaming delay
                await asyncio.sleep(CHUNK_DURATION_S)

            logger.info("Finished sending audio file.")
            await asyncio.sleep(3)  # Wait longer for final processing/results
            await websocket.close()
            logger.info("WebSocket closed by client.")

            await receive_task

    except websockets.exceptions.InvalidURI:
        logger.error(f"Invalid WebSocket URI: {uri}")
    except ConnectionRefusedError:
        logger.error(f"Connection refused for {uri}. Is the server running?")
    except Exception as e:
        logger.error(f"Failed to connect or communicate: {e}", exc_info=True)


if __name__ == "__main__":
    # IMPORTANT: Replace with the actual path to your test audio file
    if AUDIO_FILE_PATH == "path/to/your/test_audio_16k_mono.wav":
        logger.error(
            "Please update AUDIO_FILE_PATH in the script with a real file path.")
    else:
        asyncio.run(send_receive(SERVER_URI))

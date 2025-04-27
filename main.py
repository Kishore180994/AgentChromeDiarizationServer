import json
import asyncio
import logging
import os
import time
import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import torch
import numpy as np
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from transformers import pipeline as hf_pipeline
import traceback  # For detailed error logging
import wave  # For saving audio files
from collections import defaultdict
import uuid
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file BEFORE accessing them
load_dotenv()

# --- Configuration ---
# Debug audio storage configuration
DEBUG_AUDIO_STORAGE = False
DEBUG_AUDIO_DIR = "debug_audio"

# Create debug audio directory if it doesn't exist and storage is enabled
if DEBUG_AUDIO_STORAGE:
    os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
    logging.info(
        f"Debug audio storage enabled. Files will be saved to {DEBUG_AUDIO_DIR}")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
SAMPLE_RATE = 16000  # Hz
SAMPLE_WIDTH = 2     # Bytes per sample (16-bit)
CHANNELS = 1         # Mono
BYTES_PER_SECOND = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS
# Process audio buffer every X seconds
PROCESSING_CHUNK_S = 15.0  # How much audio to process at once
MIN_BUFFER_S = PROCESSING_CHUNK_S  # Need at least this much to process

# --- Model Initialization ---
pipeline = None
asr_pipeline = None
DEVICE = None

try:
    logger.info("Attempting to load Pyannote pipeline...")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        logger.warning("HUGGINGFACE_TOKEN environment variable not set.")

    # Load the pipeline with default configuration
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        pipeline.to(DEVICE)
        logger.info(
            f"Pyannote pipeline loaded successfully on GPU ({DEVICE}).")
    else:
        DEVICE = torch.device("cpu")
        logger.warning("GPU not available. Pyannote loaded on CPU.")

    # Initialize speech recognition model (Whisper)
    logger.info("Loading speech recognition model...")
    asr_pipeline = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",  # Or tiny for faster, less accurate
        device=0 if torch.cuda.is_available() else -1,
        chunk_length_s=30,  # Larger chunk for ASR context if processing per segment
        # We don't need ASR timestamps as much if processing segment by segment
        # return_timestamps=False, # Or True if you still want word timestamps within the segment
        model_kwargs={
            "use_flash_attention_2": True if torch.cuda.is_available() else False
        }
    )
    # Configure for transcription
    asr_pipeline.model.config.forced_decoder_ids = asr_pipeline.tokenizer.get_decoder_prompt_ids(
        language="en", task="transcribe"
    )
    logger.info("Speech recognition model loaded successfully.")

except ImportError as ie:
    logger.error(
        f"ImportError: {ie}. Make sure 'pyannote.audio' installed.", exc_info=True)
    pipeline = None
    asr_pipeline = None
except Exception as e:
    logger.error(f"Failed to load models: {e}", exc_info=True)
    pipeline = None
    asr_pipeline = None

app = FastAPI(title="Speaker Diarization Service")

# --- Speaker Tracking System ---


class SpeakerTracker:
    """
    Maintains speaker identity across audio chunks by tracking speaker embeddings.
    Uses cosine similarity to match new speakers with previously identified ones.
    """

    def __init__(self, similarity_threshold=0.7, max_speakers=20):
        self.speaker_embeddings = {}  # Global speaker ID -> embedding
        self.speaker_last_seen = {}   # Global speaker ID -> timestamp
        self.similarity_threshold = similarity_threshold
        self.max_speakers = max_speakers
        self.next_speaker_id = 1
        self.embedding_model = None
        self.load_embedding_model()

    def load_embedding_model(self):
        """Load the speaker embedding model."""
        try:
            # We'll use the segmentation model from pyannote for embeddings
            # This is already loaded as part of the pipeline
            logger.info("Using pyannote embedding model for speaker tracking")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")

    def extract_embedding(self, audio_segment, sample_rate=SAMPLE_RATE):
        """
        Extract speaker embedding from audio segment.

        Args:
            audio_segment: numpy array of audio samples
            sample_rate: sample rate of the audio

        Returns:
            embedding vector (numpy array)
        """
        try:
            # Ensure audio segment is long enough (pad if necessary)
            min_samples = int(0.5 * sample_rate)  # At least 0.5 seconds
            if len(audio_segment) < min_samples:
                padding = np.zeros(
                    min_samples - len(audio_segment), dtype=audio_segment.dtype)
                audio_segment = np.concatenate([audio_segment, padding])

            # Normalize audio
            audio_segment = audio_segment / \
                (np.max(np.abs(audio_segment)) + 1e-10)

            # Convert to torch tensor
            waveform = torch.from_numpy(audio_segment).unsqueeze(0).to(DEVICE)

            # Since we can't access the embedding model directly, we'll use a more
            # sophisticated feature extraction approach to better distinguish speakers

            # 1. Extract more detailed spectral features
            # Split into frames
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.010 * sample_rate)    # 10ms hop
            n_frames = 1 + (len(audio_segment) - frame_length) // hop_length

            features = []
            for i in range(n_frames):
                start = i * hop_length
                end = start + frame_length
                if end <= len(audio_segment):
                    frame = audio_segment[start:end]

                    # Apply pre-emphasis to enhance high frequencies
                    pre_emphasis = 0.97
                    emphasized_frame = np.append(
                        frame[0], frame[1:] - pre_emphasis * frame[:-1])

                    # Apply window function
                    window = np.hamming(len(emphasized_frame))
                    windowed_frame = emphasized_frame * window

                    # Compute FFT
                    fft = np.abs(np.fft.rfft(windowed_frame))

                    # Extract more detailed spectral features
                    # More frequency bands for better speaker discrimination
                    bands = []
                    band_edges = [0, 5, 10, 15, 20, 30, 40,
                                  60, 80, 100, 120, 150, 180, 220]
                    for j in range(len(band_edges)-1):
                        band_sum = np.sum(fft[band_edges[j]:band_edges[j+1]])
                        bands.append(band_sum)

                    # Compute pitch-related features
                    # Simple zero-crossing rate as pitch indicator
                    zero_crossings = np.sum(np.abs(np.diff(np.signbit(frame))))

                    # Energy features
                    energy = np.sum(frame**2)
                    log_energy = np.log(energy + 1e-10)

                    # Combine all features
                    frame_features = np.concatenate(
                        [[energy, log_energy, zero_crossings], bands])
                    features.append(frame_features)

            if not features:
                # Fallback if no frames
                return np.random.randn(64) / np.sqrt(64)

            # Convert to numpy array and get statistics across frames
            features = np.array(features)

            # Get more detailed statistics across frames
            mean_features = np.mean(features, axis=0)
            std_features = np.std(features, axis=0)
            max_features = np.max(features, axis=0)
            min_features = np.min(features, axis=0)

            # Compute deltas (first derivatives) for dynamic information
            if features.shape[0] > 1:
                deltas = features[1:] - features[:-1]
                delta_mean = np.mean(deltas, axis=0)
                delta_std = np.std(deltas, axis=0)
            else:
                delta_mean = np.zeros_like(mean_features)
                delta_std = np.zeros_like(std_features)

            # Combine into a single feature vector
            embedding = np.concatenate([
                mean_features, std_features,
                max_features, min_features,
                delta_mean, delta_std
            ])

            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            # Return a random embedding as fallback (normalized)
            random_emb = np.random.randn(32)  # Smaller embedding size
            return random_emb / np.linalg.norm(random_emb)

    def match_speaker(self, embedding, timestamp):
        """
        Match a speaker embedding with existing speakers.

        Args:
            embedding: numpy array of speaker embedding
            timestamp: current timestamp for updating last seen time

        Returns:
            global_speaker_id: string identifier for the speaker
        """
        # Log the number of speakers currently tracked
        logger.debug(
            f"Currently tracking {len(self.speaker_embeddings)} speakers")

        if not self.speaker_embeddings:
            # First speaker
            global_id = "SPEAKER_01"
            self.speaker_embeddings[global_id] = embedding
            self.speaker_last_seen[global_id] = timestamp
            self.next_speaker_id = 2
            logger.info(f"First speaker registered as {global_id}")
            return global_id

        # Calculate similarity with existing speakers
        similarities = {}
        for speaker_id, spk_embedding in self.speaker_embeddings.items():
            similarity = cosine_similarity(
                embedding.reshape(1, -1),
                spk_embedding.reshape(1, -1)
            )[0][0]
            similarities[speaker_id] = similarity
            logger.debug(f"Similarity with {speaker_id}: {similarity:.4f}")

        # Find the most similar speaker
        best_match = max(similarities.items(), key=lambda x: x[1])
        best_speaker_id, best_similarity = best_match

        # Time-based adjustment: lower threshold for recently seen speakers
        time_since_last_seen = timestamp - \
            self.speaker_last_seen.get(best_speaker_id, 0)
        adjusted_threshold = self.similarity_threshold

        # If speaker was seen recently (within 30 seconds), lower the threshold significantly
        if time_since_last_seen < 30:
            adjusted_threshold = max(0.1, self.similarity_threshold - 0.1)
            logger.debug(
                f"Speaker {best_speaker_id} seen recently, adjusted threshold: {adjusted_threshold:.2f}")

        # Use a very low threshold to match speakers
        if best_similarity >= adjusted_threshold:
            # Update the embedding with a weighted average
            combined = 0.7 * embedding + 0.3 * \
                self.speaker_embeddings[best_speaker_id]
            # Normalize
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm

            self.speaker_embeddings[best_speaker_id] = combined
            self.speaker_last_seen[best_speaker_id] = timestamp
            logger.info(
                f"Matched with existing speaker {best_speaker_id} (similarity: {best_similarity:.4f})")
            return best_speaker_id
        else:
            # New speaker
            global_id = f"SPEAKER_{self.next_speaker_id:02d}"
            self.speaker_embeddings[global_id] = embedding
            self.speaker_last_seen[global_id] = timestamp
            self.next_speaker_id += 1
            logger.info(
                f"New speaker registered as {global_id} (best similarity was {best_similarity:.4f} with {best_speaker_id})")

            # Prune old speakers if we exceed the maximum
            self._prune_old_speakers()

            return global_id

    def _prune_old_speakers(self):
        """Remove oldest speakers if we exceed the maximum number."""
        if len(self.speaker_embeddings) <= self.max_speakers:
            return

        # Sort by last seen time
        sorted_speakers = sorted(
            self.speaker_last_seen.items(),
            key=lambda x: x[1]
        )

        # Remove oldest speakers
        speakers_to_remove = len(self.speaker_embeddings) - self.max_speakers
        for i in range(speakers_to_remove):
            speaker_id = sorted_speakers[i][0]
            del self.speaker_embeddings[speaker_id]
            del self.speaker_last_seen[speaker_id]
            logger.info(f"Pruned old speaker {speaker_id} due to limit")


# Dictionary to store speaker trackers for each client
client_speaker_trackers = {}

# Dictionary to store accumulated audio for each client
# This allows us to maintain context across chunks
client_audio_buffers = {}
client_speaker_maps = {}  # Maps local speaker IDs to global ones

# --- Helper Function: Process Audio Chunk ---


async def process_audio_chunk(audio_chunk_np: np.ndarray, chunk_start_time_s: float, client_id: str):
    """
    Runs diarization and then ASR on each detected segment.
    Maintains speaker identity across chunks using the SpeakerTracker.

    Args:
        audio_chunk_np: Numpy array (float32) of audio samples for the chunk.
        chunk_start_time_s: The absolute start time (in seconds) of this audio chunk.
        client_id: Unique identifier for the client connection.

    Returns:
        List of dictionaries, each containing:
        {'speaker': str, 'text': str, 'start': float, 'end': float}
        Returns None if processing fails.
    """
    if pipeline is None or asr_pipeline is None or DEVICE is None:
        logger.error("Pipelines not loaded, cannot process audio chunk.")
        return None

    loop = asyncio.get_running_loop()
    combined_results = []

    try:
        # --- 1. Run Diarization ---
        logger.info(
            f"Running diarization on {len(audio_chunk_np)/SAMPLE_RATE:.2f}s chunk starting at {chunk_start_time_s:.2f}s")
        logger.info(
            f"Client ID: {client_id}, Audio shape: {audio_chunk_np.shape}")
        diarization_start_time = time.time()

        # Prepare waveform for Pyannote
        waveform = torch.from_numpy(audio_chunk_np).unsqueeze(0).to(DEVICE)
        inputs = {"waveform": waveform, "sample_rate": SAMPLE_RATE}

        try:
            # Run diarization in executor
            logger.info(f"Starting PyAnnote diarization pipeline...")
            diarization = await loop.run_in_executor(None, pipeline, inputs)
            diarization_time = time.time() - diarization_start_time
            logger.info(f"Diarization complete in {diarization_time:.3f}s")

            # Log diarization results summary
            num_speakers = len(diarization.labels())
            num_segments = len(list(diarization.itertracks(yield_label=True)))
            logger.info(
                f"Diarization found {num_speakers} speaker(s) across {num_segments} segment(s)")

            # Log individual speaker labels detected
            speakers = sorted(diarization.labels())
            logger.info(f"Detected speakers: {', '.join(speakers)}")
        except Exception as e:
            logger.error(
                f"Diarization failed for chunk starting at {chunk_start_time_s:.2f}s: {e}", exc_info=True)
            return None  # Cant proceed without diarization

        # --- 2. Get or create speaker tracker for this client ---
        if client_id not in client_speaker_trackers:
            logger.info(f"Creating new speaker tracker for client {client_id}")
            client_speaker_trackers[client_id] = SpeakerTracker()

        speaker_tracker = client_speaker_trackers[client_id]

        # --- 3. Run ASR on each segment and track speakers ---
        asr_tasks = []
        segment_info = []  # Store speaker and times to match with ASR results
        speaker_embeddings = {}  # Local speaker ID -> embedding

        # Log detailed information about each segment
        logger.info("Processing individual diarization segments:")

        for turn, _, local_speaker_id in diarization.itertracks(yield_label=True):
            # Timestamps relative to the chunk start
            local_start_s = turn.start
            local_end_s = turn.end
            duration = local_end_s - local_start_s
            abs_start = chunk_start_time_s + local_start_s
            abs_end = chunk_start_time_s + local_end_s

            # Log segment details
            logger.info(f"  Segment: {local_speaker_id}, Duration: {duration:.2f}s, " +
                        f"Time: {local_start_s:.2f}s-{local_end_s:.2f}s (relative), " +
                        f"{abs_start:.2f}s-{abs_end:.2f}s (absolute)")

            # Skip very short segments (potential noise)
            if duration < 0.1:  # Adjust threshold as needed
                logger.info(
                    f"    Skipping segment: too short ({duration:.2f}s < 0.1s)")
                continue

            # Extract audio segment
            start_sample = int(local_start_s * SAMPLE_RATE)
            end_sample = int(local_end_s * SAMPLE_RATE)
            segment_audio_np = audio_chunk_np[start_sample:end_sample]

            if segment_audio_np.size == 0:
                logger.warning(
                    f"Empty audio segment for {local_speaker_id} at {local_start_s:.2f}s, skipping ASR.")
                continue

            # Extract speaker embedding for this segment
            embedding = speaker_tracker.extract_embedding(segment_audio_np)

            # Match with global speaker ID using the tracker
            timestamp = chunk_start_time_s + local_start_s

            # SIMPLEST APPROACH: Just use PyAnnote's speaker IDs directly
            # Convert the PyAnnote speaker ID to our format (SPEAKER_XX)

            # Extract the speaker number from PyAnnote's label (e.g., "SPEAKER_00" -> "00")
            speaker_num = local_speaker_id.split("_")[1]

            # Convert to our format (e.g., "00" -> "SPEAKER_01")
            # Add 1 to the speaker number since PyAnnote uses 0-based indexing and we want 1-based
            speaker_num_int = int(speaker_num) + 1
            global_speaker_id = f"SPEAKER_{speaker_num_int:02d}"

            logger.info(
                f"Using PyAnnote speaker: {local_speaker_id} -> {global_speaker_id} ({local_start_s:.2f}s - {local_end_s:.2f}s)")

            # Prepare data for ASR pipeline
            # Whisper expects int16 if passing raw
            # segment_audio_int16 = (segment_audio_np * 32768.0).astype(np.int16)
            # Or directly use float32 numpy array
            asr_input = {"raw": segment_audio_np, "sampling_rate": SAMPLE_RATE}

            # Add ASR task to run in parallel
            asr_tasks.append(
                loop.run_in_executor(
                    None,
                    # Use lambda with default argument to capture CURRENT value of asr_input
                    # Pass language='en' nested within generate_kwargs
                    lambda inp=asr_input: asr_pipeline(
                        inp, generate_kwargs={"language": "en"})
                )
            )
            # Store corresponding info with global speaker ID
            segment_info.append({
                "speaker": global_speaker_id,  # Use the global ID from speaker tracker
                "local_start": local_start_s,
                "local_end": local_end_s
            })

        # Run all ASR tasks concurrently
        if asr_tasks:
            logger.info(f"Running ASR for {len(asr_tasks)} segments...")
            asr_results = await asyncio.gather(*asr_tasks, return_exceptions=True)
            logger.info("ASR for segments completed.")

            # Combine results
            for i, result in enumerate(asr_results):
                info = segment_info[i]
                if isinstance(result, Exception):
                    logger.error(
                        f"ASR failed for segment {info['speaker']} ({info['local_start']:.2f}s): {result}")
                    transcript = "[ASR Error]"
                elif result and isinstance(result, dict) and "text" in result:
                    transcript = result["text"].strip()
                    if not transcript:
                        logger.info(
                            f"ASR produced empty transcript for segment {info['speaker']} ({info['local_start']:.2f}s)")
                        transcript = ""  # Represent silence or non-speech
                else:
                    logger.warning(
                        f"Unexpected ASR result format for segment {info['speaker']}: {result}")
                    transcript = "[ASR Format Error]"

                # Add the combined result with ABSOLUTE timestamps
                combined_results.append({
                    "speaker": info["speaker"],
                    "text": transcript,
                    "start": round(chunk_start_time_s + info["local_start"], 3),
                    "end": round(chunk_start_time_s + info["local_end"], 3)
                })
        else:
            logger.info("No valid segments found for ASR in this chunk.")

        # Sort results by start time before returning
        combined_results.sort(key=lambda x: x["start"])
        return combined_results

    except Exception as e:
        logger.error(
            f"Error processing chunk starting at {chunk_start_time_s:.2f}s: {e}", exc_info=True)
        logger.error(traceback.format_exc())  # Log detailed traceback
        return None


# --- WebSocket Endpoint ---
@app.websocket("/ws/diarize")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"WebSocket connection established: {client_id}")

    if pipeline is None or asr_pipeline is None:
        logger.warning(
            f"Pipelines not available for client {client_id}. Closing.")
        await websocket.close(code=1011, reason="Required services unavailable")
        return

    # Per-Connection State
    audio_buffer = bytearray()
    total_bytes_received = 0
    last_process_time = time.time()

    try:
        while True:
            data = await websocket.receive_bytes()
            if not data:  # Handle case where client might send empty bytes
                continue
            logger.debug(f"Received {len(data)} bytes from {client_id}")

            # Store audio for debugging if enabled
            if DEBUG_AUDIO_STORAGE:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                debug_filename = f"{DEBUG_AUDIO_DIR}/{client_id.replace(':', '_')}_{timestamp}.wav"

                try:
                    # Create a WAV file with the received audio chunk
                    with wave.open(debug_filename, 'wb') as wf:
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(SAMPLE_WIDTH)
                        wf.setframerate(SAMPLE_RATE)
                        wf.writeframes(data)
                    logger.debug(
                        f"Saved debug audio chunk to {debug_filename}")
                except Exception as e:
                    logger.error(
                        f"Failed to save debug audio: {e}", exc_info=True)

            audio_buffer.extend(data)
            total_bytes_received += len(data)

            current_time = time.time()
            buffer_duration_s = len(audio_buffer) / BYTES_PER_SECOND

            # Check if enough audio is buffered for processing
            if buffer_duration_s >= MIN_BUFFER_S:
                bytes_to_process = int(PROCESSING_CHUNK_S * BYTES_PER_SECOND)
                # Ensure alignment with sample width (should be okay with BYTES_PER_SECOND)
                bytes_to_process = (bytes_to_process //
                                    SAMPLE_WIDTH) * SAMPLE_WIDTH

                if bytes_to_process > len(audio_buffer):
                    # This shouldn't happen if MIN_BUFFER_S >= PROCESSING_CHUNK_S
                    # but as a safeguard, process what we have if it's enough
                    bytes_to_process = (len(audio_buffer) //
                                        SAMPLE_WIDTH) * SAMPLE_WIDTH
                    # Don't process tiny leftovers
                    if bytes_to_process < int(MIN_BUFFER_S * BYTES_PER_SECOND * 0.8):
                        continue  # Wait for more data
                    logger.warning(
                        f"Processing slightly less than {PROCESSING_CHUNK_S}s as buffer isn't full yet.")

                process_chunk_bytes = audio_buffer[:bytes_to_process]

                # Calculate absolute start time of this chunk
                # Byte offset where the *entire current buffer* started
                buffer_start_byte = total_bytes_received - len(audio_buffer)
                # Start time of the specific chunk we are processing
                chunk_start_time_s = buffer_start_byte / BYTES_PER_SECOND

                logger.info(
                    f"Processing {len(process_chunk_bytes) / BYTES_PER_SECOND:.2f}s audio chunk for {client_id} starting at stream time {chunk_start_time_s:.2f}s")

                # Prepare audio data for processing
                try:
                    samples_int16 = np.frombuffer(
                        process_chunk_bytes, dtype=np.int16)
                    samples_float32 = samples_int16.astype(
                        np.float32) / 32768.0
                except ValueError as ve:
                    logger.error(
                        f"Buffer conversion error for {client_id}: {ve}. Processing {bytes_to_process} bytes.")
                    # Discard bad chunk
                    audio_buffer = audio_buffer[bytes_to_process:]
                    continue

                # --- Run Processing ---
                last_process_time = current_time  # Update time before async task
                processing_results = await process_audio_chunk(samples_float32, chunk_start_time_s, client_id)

                # --- Send Results ---
                if processing_results is not None:
                    if processing_results:
                        response_data = {
                            "type": "speaker_transcription_update",
                            "segments": processing_results  # Already contains speaker, text, start, end
                        }

                        # Log the exact JSON payload being sent to client with high visibility
                        print(
                            "\n\n=================== DATA SENT TO CLIENT ===================")
                        print(json.dumps(response_data, indent=2))
                        print(
                            "=================== END OF CLIENT DATA ===================\n\n")
                        logger.info("DATA SENT TO CLIENT >>>")
                        logger.info(json.dumps(response_data, indent=2))
                        logger.info("<<< END OF DATA SENT TO CLIENT")

                        # Send the data to the client
                        await websocket.send_json(response_data)
                    else:
                        logger.info(
                            f"No speaker segments with transcription found in the current chunk for {client_id}.")
                else:
                    logger.warning(
                        f"Processing failed for chunk of {client_id}, no results sent.")

                # --- Buffer Management ---
                # Remove the processed part from the buffer
                audio_buffer = audio_buffer[bytes_to_process:]
                logger.debug(
                    f"Buffer reduced, remaining duration: {len(audio_buffer) / BYTES_PER_SECOND:.2f}s")

            # Optional: Add a small sleep to prevent tight loop if no data received
            # await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed: {client_id}")
    except Exception as e:
        logger.error(
            f"Unhandled error in WebSocket loop for {client_id}: {e}", exc_info=True)
        logger.error(traceback.format_exc())  # Log detailed traceback
        try:
            # Try to inform the client about the error
            await websocket.send_json({"type": "error", "message": f"Internal server error: {e}"})
        except Exception:
            pass  # Websocket might already be closed
        # Ensure connection is closed on server error
        await websocket.close(code=1011, reason=f"Internal Server Error: {e}")


# --- Basic HTTP Route ---
@app.get("/")
async def read_root():
    return {"message": "Speaker Diarization & Transcription Service is running"}

# Reminder: Run with HUGGINGFACE_TOKEN="hf_..." uvicorn main:app --reload --host 0.0.0.0 --port 8000

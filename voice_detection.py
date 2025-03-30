import webrtcvad
import logging
import numpy as np
import struct
import pyaudio
import wave
import whisper
import warnings
from collections import deque
import time

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load the Whisper Tiny model
model = whisper.load_model("tiny")


def transcribe_audio(model, filename):

    # Load the audio file and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(filename)

    # Make a prediction
    result = model.transcribe(audio, word_timestamps=True)

    return result

class VoiceDetector:
    def __init__(self, sample_rate=16000, frame_duration=30):
        self.vad = webrtcvad.Vad(3)  # Reduced aggressiveness for better continuous speech detection
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self.logger = logging.getLogger(__name__)
        self.silence_frames = 0
        self.max_silence_frames = 200  # Allow more silence between words
        self.min_speech_frames = 1  # Require minimum speech frames to avoid spurious detections, Adjust for quicker detection
        self.speech_frames = 0
        self.is_speaking = False
        
    def _frame_generator(self, audio_data):
        """Generate audio frames from raw audio data."""
        if len(audio_data) < self.frame_size:
            self.logger.warning(f"Audio data too short: {len(audio_data)} bytes")
            return []
        
        n = len(audio_data)
        offset = 0
        frames = []
        while offset + self.frame_size <= n:
            frames.append(audio_data[offset:offset + self.frame_size])
            offset += self.frame_size
        return frames

    def _convert_audio_data(self, audio_data):
        """Convert audio data to the correct format."""
        try:
            # First try to interpret as float32
            float_array = np.frombuffer(audio_data, dtype=np.float32)
            # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
            int16_array = (float_array * 32767).astype(np.int16)
            # float_array = np.clip(float_array, -1, 1)  # Limit values to avoid overflow
            # int16_array = (float_array * 32767).astype(np.int16)

            return int16_array
        except ValueError:
            try:
                # If that fails, try direct int16 interpretation
                return np.frombuffer(audio_data, dtype=np.int16)
            except ValueError as e:
                # If both fail, try to pad the data to make it aligned
                padding_size = (2 - (len(audio_data) % 2)) % 2
                if padding_size > 0:
                    padded_data = audio_data + b'\x00' * padding_size
                    return np.frombuffer(padded_data, dtype=np.int16)
                raise e

    def detect_voice(self, audio_data):
        """
        Detect voice activity in audio data.
        
        Args:
            audio_data (bytes): Raw audio data
            
        Returns:
            bool: True if voice activity is detected, False otherwise
        """
        try:
            if audio_data is None or len(audio_data) == 0:
                self.logger.warning("Audio data is empty or None")
                return False
                
            # Convert audio data to the correct format
            try:
                audio_array = self._convert_audio_data(audio_data)
                if len(audio_array) == 0:
                    self.logger.warning("No valid audio data after conversion")
                    return False
            except ValueError as e:
                self.logger.error(f"Error converting audio data: {str(e)}")
                return False
            
            # Process frames
            frames = self._frame_generator(audio_array)
            if not frames:
                self.logger.warning("No frames generated from audio data")
                return False
                
            # Count speech frames in this chunk
            current_speech_frames = 0
            for frame in frames:
                try:
                    # Pack the frame into bytes
                    frame_bytes = struct.pack("%dh" % len(frame), *frame)
                    
                    # Check for voice activity
                    if self.vad.is_speech(frame_bytes, self.sample_rate):
                        current_speech_frames += 1
                        self.speech_frames += 1
                        self.silence_frames = 0
                    else:
                        self.silence_frames += 1
                        
                except struct.error as se:
                    self.logger.error(f"Error packing frame data: {str(se)}")
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing frame: {str(e)}")
                    continue
            
            # Update speaking state
            if current_speech_frames > 0:
                if not self.is_speaking and self.speech_frames >= self.min_speech_frames:
                    self.is_speaking = True
                return True
            elif self.silence_frames > self.max_silence_frames:
                if self.is_speaking:
                    self.is_speaking = False
                    self.speech_frames = 0
                return False
            
            # Keep current state if in transition
            return self.is_speaking
            
        except Exception as e:
            self.logger.error(f"Error in voice detection: {str(e)}")
            return False 


if __name__ == "__main__":
    voice_detector = VoiceDetector()

    # Constants
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    OUTPUT_FILENAME = "output.wav"
    MAX_SILENCE_SECONDS = 2  # Silence threshold in seconds

    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # Silence detection variables
    max_silence_frames = int((RATE / CHUNK) * MAX_SILENCE_SECONDS)  # Convert silence duration to frames
    silence_duration = 0
    audio_buffer = deque()

    # Time tracking
    speech_time = 0
    silence_time = 0
    silence_start_time = None  # To track silence after voice stops
    final_silence_time = 0  # To store total silence time after last voice


    start_time = time.time()  # Start overall recording timer

    while True:
        block_start_time = time.time()  # Track time per loop iteration

        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # Check for voice activity
        voice_detected = voice_detector.detect_voice(data)

        if voice_detected:
            # Track time spent in the "if" block
            speech_time += time.time() - block_start_time

            # Reset silence counter and add to buffer
            silence_duration = 0
            audio_buffer.append(data)

        else:
            # Track time spent in the "else" block
            silence_time += time.time() - block_start_time

            # If silence just started, record the time
            if silence_duration == 0:
                silence_start_time = time.time()

            # Increment silence counter
            silence_duration += 1

            # If we were collecting speech and hit max silence, process the buffer
            if len(audio_buffer) > 0 and silence_duration >= max_silence_frames:
                
                # Calculate silence time after last speech
                if silence_start_time:
                    final_silence_time = time.time() - silence_start_time
                
                print("Saving recording...")

                with wave.open(OUTPUT_FILENAME, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(audio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(audio_buffer))

                print(f"Audio recorded and saved as {OUTPUT_FILENAME}")

    # Cleanup
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Print final timing results
    total_time = time.time() - start_time
    print(f"Total Recording Time: {total_time:.2f} sec")
    print(f"Total Speech Time: {speech_time:.2f} sec")
    print(f"Total Silence Time (before max silence): {silence_time:.2f} sec")
    print(f"Final Silence Time (after last speech): {final_silence_time:.2f} sec")
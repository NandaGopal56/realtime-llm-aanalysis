import whisper
import pyaudio
import numpy as np
from collections import deque
import threading
import time
import warnings
from scipy import signal
import webrtcvad

# Suppress warnings
warnings.filterwarnings("ignore")

class EnhancedRealTimeSTT:
    def __init__(self, model_name="small", chunk_duration=3, overlap_duration=1):
        """
        Enhanced real-time speech-to-text with better accuracy.
        
        Args:
            model_name (str): Whisper model ('tiny', 'base', 'small', 'medium', 'large')
            chunk_duration (float): Duration of audio chunks (seconds)
            overlap_duration (float): Overlap between chunks (seconds)
        """
        print("Loading enhanced Whisper model...")
        self.model = whisper.load_model(model_name)
        
        # Audio configuration - optimized for better quality
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.channels = 1
        
        # Enhanced audio buffer with longer context
        buffer_size = int(self.sample_rate * (chunk_duration + overlap_duration))
        self.audio_buffer = deque(maxlen=buffer_size)
        
        # Voice Activity Detection
        try:
            self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3 (2 is balanced)
            self.use_vad = True
        except:
            self.use_vad = False
            print("WebRTC VAD not available, using energy-based detection")
        
        # Audio preprocessing
        self.setup_audio_filters()
        
        # Threading
        self.is_recording = False
        self.processing_thread = None
        
        # PyAudio setup with better quality settings
        self.audio = pyaudio.PyAudio()
        self.stream = None
        
        # Context tracking for better transcription
        self.last_transcription = ""
        self.transcription_history = deque(maxlen=3)
        
        print("Enhanced real-time transcription ready!")
    
    def setup_audio_filters(self):
        """Setup audio preprocessing filters."""
        try:
            # High-pass filter to remove low-frequency noise
            nyquist = self.sample_rate / 2
            hp_cutoff = 80 / nyquist  # 80 Hz high-pass
            lp_cutoff = 7800 / nyquist  # 7.8 kHz low-pass (safer than 8kHz)
            
            # Ensure cutoff frequencies are in valid range (0 < Wn < 1)
            hp_cutoff = max(0.001, min(0.99, hp_cutoff))
            lp_cutoff = max(0.001, min(0.99, lp_cutoff))
            
            self.hp_b, self.hp_a = signal.butter(4, hp_cutoff, btype='high')
            self.lp_b, self.lp_a = signal.butter(4, lp_cutoff, btype='low')
            self.use_filters = True
            
        except Exception as e:
            print(f"Filter setup failed, disabling audio filters: {e}")
            self.use_filters = False
    
    def preprocess_audio(self, audio_data):
        """Apply audio preprocessing for better quality."""
        # Convert to float
        audio = audio_data.astype(np.float32)
        
        # Apply filters only if they were successfully initialized
        if hasattr(self, 'use_filters') and self.use_filters:
            try:
                # Apply high-pass filter (remove low-frequency noise)
                audio = signal.filtfilt(self.hp_b, self.hp_a, audio)
                
                # Apply low-pass filter (remove high-frequency noise)  
                audio = signal.filtfilt(self.lp_b, self.lp_a, audio)
            except Exception:
                # If filtering fails, continue without it
                pass
        
        # Normalize audio with dynamic range compression
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            # Soft compression to even out volume levels
            audio = np.sign(audio) * np.power(np.abs(audio), 0.7)
        
        return audio
    
    def start_recording(self):
        """Start recording with enhanced settings."""
        if self.is_recording:
            return
        
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback,
                # Enhanced settings for better audio quality
                input_device_index=None,  # Use default device
            )
            
            self.is_recording = True
            self.stream.start_stream()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio_chunks)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            print("🎤 Enhanced listening mode active... (Press Ctrl+C to stop)\n")
            
        except Exception as e:
            print(f"Error starting recording: {e}")
    
    def stop_recording(self):
        """Stop recording."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Enhanced audio callback with preprocessing."""
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.audio_buffer.extend(audio_data)
        return (None, pyaudio.paContinue)
    
    def _has_voice_activity(self, audio_chunk):
        """Enhanced voice activity detection."""
        if self.use_vad:
            # Convert to 16-bit PCM for WebRTC VAD
            pcm_data = (audio_chunk * 32767).astype(np.int16).tobytes()
            
            # Check in 30ms frames (480 samples at 16kHz)
            frame_size = 480
            frames_with_voice = 0
            total_frames = 0
            
            for i in range(0, len(audio_chunk) - frame_size, frame_size):
                frame = (audio_chunk[i:i+frame_size] * 32767).astype(np.int16).tobytes()
                if len(frame) == frame_size * 2:  # 2 bytes per sample
                    if self.vad.is_speech(frame, self.sample_rate):
                        frames_with_voice += 1
                    total_frames += 1
            
            if total_frames > 0:
                voice_ratio = frames_with_voice / total_frames
                return voice_ratio > 0.3  # At least 30% of frames should have voice
        
        # Fallback to energy-based detection with improved thresholds
        if len(audio_chunk) < int(self.sample_rate * 0.5):
            return False
        
        # Multiple energy measures for better detection
        rms_energy = np.sqrt(np.mean(audio_chunk ** 2))
        zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(audio_chunk)))) / 2
        
        # Voice typically has moderate energy and zero crossing rate
        has_energy = rms_energy > 0.003  # Lower threshold for sensitivity
        has_voice_characteristics = 0.02 < zero_crossing_rate < 0.3
        
        return has_energy and has_voice_characteristics
    
    def _process_audio_chunks(self):
        """Enhanced audio chunk processing."""
        last_process_time = time.time()
        
        while self.is_recording:
            current_time = time.time()
            
            if current_time - last_process_time >= self.chunk_duration:
                if len(self.audio_buffer) > 0:
                    # Get audio data and preprocess
                    raw_audio = np.array(list(self.audio_buffer), dtype=np.int16)
                    processed_audio = self.preprocess_audio(raw_audio)
                    
                    # Check for voice activity
                    if self._has_voice_activity(processed_audio):
                        self._transcribe_chunk(processed_audio)
                    
                    # Manage buffer with overlap
                    overlap_samples = int(self.sample_rate * self.overlap_duration)
                    if len(self.audio_buffer) > overlap_samples:
                        overlap_data = list(self.audio_buffer)[-overlap_samples:]
                        self.audio_buffer.clear()
                        self.audio_buffer.extend(overlap_data)
                
                last_process_time = current_time
            
            time.sleep(0.05)
    
    def _transcribe_chunk(self, audio_chunk):
        """Enhanced transcription with context awareness."""
        try:
            # Enhanced Whisper options for better accuracy
            result = self.model.transcribe(
                audio_chunk,
                language='en',
                task='transcribe',
                fp16=False,
                verbose=False,
                # Enhanced options for better accuracy
                condition_on_previous_text=True,
                temperature=0.0,  # More deterministic output
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                # Add context from previous transcriptions
                initial_prompt=self._get_context_prompt(),
                word_timestamps=False
            )
            
            text = result['text'].strip()
            
            if text and len(text) > 1:
                # Post-process transcription
                processed_text = self._post_process_text(text)
                
                if processed_text and processed_text != self.last_transcription:
                    print(f"{processed_text}")
                    self.last_transcription = processed_text
                    self.transcription_history.append(processed_text)
                
        except Exception as e:
            pass  # Silently handle errors
    
    def _get_context_prompt(self):
        """Generate context prompt from recent transcriptions."""
        if self.transcription_history:
            # Use last few transcriptions as context
            context = " ".join(list(self.transcription_history)[-2:])
            return context
        return ""
    
    def _post_process_text(self, text):
        """Post-process transcribed text for better quality."""
        # Remove common transcription artifacts
        text = text.replace(" um ", " ")
        text = text.replace(" uh ", " ")
        text = text.replace("  ", " ")
        
        # Capitalize first letter
        text = text.strip()
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        return text
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_recording()
        if hasattr(self, 'audio'):
            self.audio.terminate()

def main():
    """Main function with enhanced settings."""
    # Enhanced configuration for better accuracy
    stt = EnhancedRealTimeSTT(
        model_name="small",     # Better accuracy than base
        chunk_duration=3.0,     # Longer chunks for more context
        overlap_duration=1.0    # More overlap for continuity
    )
    
    try:
        stt.start_recording()
        
        while stt.is_recording:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nStopping enhanced transcription...")
    finally:
        stt.cleanup()

if __name__ == "__main__":
    # Install additional dependencies if needed
    try:
        import scipy
        import webrtcvad
    except ImportError:
        print("Installing additional dependencies for better accuracy...")
        print("Run: pip install scipy webrtcvad")
        print("Falling back to basic mode...\n")
    
    main()
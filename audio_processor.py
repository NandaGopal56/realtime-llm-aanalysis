import time
import wave
import pyaudio
import asyncio
import websockets
from collections import deque
from voice_detector import VoiceDetector
from transcriber import transcribe_audio

class AudioProcessor:
    def __init__(self, output_filename="output.wav", max_silence_seconds=2):
        self.output_filename = output_filename
        self.max_silence_seconds = max_silence_seconds
        self.voice_detector = VoiceDetector()

    def process_audio_stream(self, audio_stream, audio_format, channels, rate, chunk_size):
        audio_buffer = deque()
        max_silence_frames = int((rate / chunk_size) * self.max_silence_seconds)
        silence_duration = 0

        for data in audio_stream:
            voice_detected = self.voice_detector.detect_voice(data)
            if voice_detected:
                print("Voice detected...")
                silence_duration = 0
                audio_buffer.append(data)
            else:
                print("No voice detected...")
                silence_duration += 1

                if len(audio_buffer) > 0 and silence_duration >= max_silence_frames:
                    self._save_audio(audio_buffer, audio_format, channels, rate)
                    return self.output_filename
        return None

    def _save_audio(self, audio_buffer, audio_format, channels, rate):
        with wave.open(self.output_filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(audio_format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(audio_buffer))
        print(f"Audio saved as {self.output_filename}")


    def transcribe_audio(self, filename):
        print("Transcribing audio...")
        return transcribe_audio(filename)





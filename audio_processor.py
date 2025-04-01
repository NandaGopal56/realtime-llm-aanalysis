import wave
import pyaudio
import asyncio
from collections import deque
from voice_detector import VoiceDetector
from transcriber import transcribe_audio
from audio_inputs import record_from_mic
from main import websocket_audio_generator

class AudioProcessor:
    def __init__(self, output_filename="output.wav", max_silence_seconds=2):
        self.output_filename = output_filename
        self.max_silence_seconds = max_silence_seconds
        self.voice_detector = VoiceDetector()

    def process_audio_stream(self, record_from_mic, audio_format, channels, rate, chunk_size):
        audio_buffer = deque()
        max_silence_frames = int((rate / chunk_size) * self.max_silence_seconds)
        silence_duration = 0

        for data in record_from_mic():
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
    
    async def a_process_audio_stream(self, websocket_audio_generator, audio_format, channels, rate, chunk_size):
        print("lll")
        
        audio_buffer = deque()
        max_silence_frames = int((rate / chunk_size) * self.max_silence_seconds)
        silence_duration = 0

        # Awaiting the WebSocket generator
        async for data in websocket_audio_generator():  
            print("Data from WS")
            voice_detected = self.voice_detector.detect_voice(data)
            if voice_detected:
                print("Voice detected...")
                silence_duration = 0
                audio_buffer.append(data)
            else:
                print("No voice detected...")
                silence_duration += 1

                if len(audio_buffer) > 0 and silence_duration >= max_silence_frames:
                    # Make save function async
                    # await self._save_audio(audio_buffer, audio_format, channels, rate)  
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




if __name__ == "__main__":
    ap = AudioProcessor()

    CHUNK = 1024                # Number of frames per buffer
    FORMAT = pyaudio.paFloat32  # Audio format (32-bit float)
    CHANNELS = 1                # Mono audio (1 channel)
    RATE = 44100                # Sample rate in Hz (CD quality)

    # ap.process_audio_stream(record_from_mic, FORMAT, CHANNELS, RATE, CHUNK)
    
    async def main(ap):
        # Call the method and await it
        await ap.a_process_audio_stream(websocket_audio_generator, FORMAT, CHANNELS, RATE, CHUNK)

    # Start the event loop
    asyncio.run(main(ap))
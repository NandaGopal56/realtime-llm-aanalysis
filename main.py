import pyaudio
from audio_processor import AudioProcessor

def record_from_mic(self):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("Listening for audio...")
    filename = self.process_audio_stream(iter(lambda: stream.read(CHUNK, exception_on_overflow=False), None), FORMAT, CHANNELS, RATE, CHUNK)
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    return filename

if __name__ == "__main__":
    processor = AudioProcessor()
    recorded_file = processor.record_from_mic()
    if recorded_file:
        transcription = processor.transcribe_audio(recorded_file)
        print(transcription)
import wave
import pyaudio
import asyncio
import json
import base64
from redis import asyncio as aioredis
from globals import REDIS_CHANNEL
from collections import deque
from voice_detector import VoiceDetector
from transcriber import transcribe_audio
from audio_inputs import record_from_mic


class AudioProcessor:
    def __init__(
            self, 
            voice_detector, 
            audio_format=pyaudio.paFloat32, 
            channels=1, 
            rate=44100, 
            chunk_size=1024, 
            output_filename="output.wav", 
            max_silence_seconds=2
        ):
        self.output_filename = output_filename
        self.max_silence_seconds = max_silence_seconds
        self.voice_detector = voice_detector
        self.audio_format = audio_format        # Audio format (32-bit float)
        self.channels = channels                # Mono audio (1 channel)
        self.rate = rate                        # Sample rate in Hz (CD quality)
        self.chunk_size = chunk_size            # Number of frames per buffer

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
    
    async def a_process_audio_stream(self, redis_channel):
        """Subscribe to Redis channel, deserialize JSON messages, and process audio stream."""
        
        redis = await aioredis.Redis.from_url("redis://localhost")
        
        # Subscribe to the Redis channel
        subscriber = redis.pubsub()
        await subscriber.subscribe(redis_channel)
        
        audio_buffer = deque()
        max_silence_frames = int((self.rate / self.chunk_size) * self.max_silence_seconds)
        silence_duration = 0

        print(f"Subscribed to Redis channel {redis_channel}, waiting for audio...")

        try:
            while True:
                # Wait for new message from Redis channel
                message = await subscriber.get_message(ignore_subscribe_messages=True)
                
                if message:
                    # Decode the message from Redis (it's a JSON string)
                    message_data = message['data'].decode('utf-8')
                    message_dict = json.loads(message_data)

                    # Base64 decode the audio data to get raw bytes
                    audio_data = base64.b64decode(message_dict["audio"])

                    # Process the audio data using the existing logic
                    voice_detected = self.voice_detector.detect_voice(audio_data)

                    if voice_detected:
                        print(f"Voice detected...{audio_data[0]}")
                        silence_duration = 0
                        audio_buffer.append(audio_data)
                    else:
                        print("No voice detected...")
                        silence_duration += 1
                else:
                    print("No voice detected...")
                    silence_duration += 1

                if len(audio_buffer) > 0 and silence_duration >= max_silence_frames:
                    # You can replace this with the actual save function if needed
                    await asyncio.to_thread(self._save_audio, audio_buffer, self.audio_format, self.channels, self.rate)
                    return self.output_filename 
                        
        except asyncio.CancelledError:
            print("Subscriber task was cancelled.")
        finally:
            await subscriber.unsubscribe(redis_channel)
            await redis.aclose()

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
    vad = VoiceDetector()
    audio_processor = AudioProcessor(voice_detector=vad)
         

    # ap.process_audio_stream(record_from_mic)
    
    # Run the subscriber function
    async def run_subscriber():
        stop_event = asyncio.Event()  # Create an event that will never be set
        await audio_processor.a_process_audio_stream(redis_channel="audio_stream")
        await stop_event.wait()  # This keeps the event loop running indefinitely

    # Start the event loop
    asyncio.run(run_subscriber())
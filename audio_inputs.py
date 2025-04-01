import pyaudio

def record_from_mic():
    """
    Generator function that continuously yields audio data from the microphone.
    
    Yields:
        numpy.ndarray: Audio data as float32 numpy array
    """
    CHUNK = 1024                # Number of frames per buffer
    FORMAT = pyaudio.paFloat32  # Audio format (32-bit float)
    CHANNELS = 1                # Mono audio (1 channel)
    RATE = 44100                # Sample rate in Hz (CD quality)
    
    audio = pyaudio.PyAudio()
    
    try:
        # Open audio stream
        audio_stream = audio.open(
            format=FORMAT, 
            channels=CHANNELS, 
            rate=RATE, 
            input=True, 
            frames_per_buffer=CHUNK
        )
        
        print("Microphone stream started")
        
        # Continuously read and yield audio data
        while True:
            # Read audio chunk
            audio_data = audio_stream.read(CHUNK, exception_on_overflow=False)
            
            # Yield the data
            yield audio_data
            
    except KeyboardInterrupt:
        print("Microphone stream stopped by user")
    except Exception as e:
        print(f"Error in microphone stream: {e}")
    finally:
        # Clean up resources
        if 'audio_stream' in locals() and audio_stream.is_active():
            audio_stream.stop_stream()
            audio_stream.close()
        audio.terminate()
        print("Microphone stream closed")
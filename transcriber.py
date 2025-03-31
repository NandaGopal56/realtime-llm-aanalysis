import whisper
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load the Whisper Tiny model
model = whisper.load_model("tiny")


def transcribe_audio(filename):
    # Load the audio file and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(filename)

    # Make a prediction
    result = model.transcribe(audio, word_timestamps=True)

    return result
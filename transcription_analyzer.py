from pathlib import Path
import whisper
from openai import OpenAI
from dotenv import load_dotenv
import os
import subprocess

load_dotenv()

def extract_audio(video_path: str, output_path: str) -> str:
    """Extract audio from video file and save as MP3 using ffmpeg"""
    try:
        # Using ffmpeg command directly
        command = [
            'ffmpeg', '-i', video_path,
            '-q:a', '0', '-map', 'a', output_path,
            '-y'
        ]
        subprocess.run(command, check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error extracting audio: {str(e)}")
    except Exception as e:
        raise Exception(f"Error extracting audio: {str(e)}")

def format_whisper_result(result):
    """Format Whisper result to plain text only."""
    return result.get('text', '')

def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio file using Whisper model"""
    try:
        # Load the Whisper model (tiny for faster processing, can be changed to base/small/medium/large)
        model = whisper.load_model("tiny")
        
        # Transcribe the audio
        result = model.transcribe(audio_path)
        
        # Get just the text
        text = format_whisper_result(result)
        
        return text
    except Exception as e:
        raise Exception(f"Error transcribing audio: {str(e)}")

def analyze_transcription(transcription: str, model: str = "gpt-3.5-turbo") -> str:
    """Analyze transcription using specified LLM model"""
    try:
        client = OpenAI(
            api_key=os.environ.get('GROQ_API_KEY'),
            base_url=os.environ.get('GROQ_API_BASE')
        )
        
        prompt = f"""Analyze the following transcription and provide key insights:
        
        Transcription:
        {transcription}
        
        Please provide:
        1. Main topics discussed
        2. Key points
        3. Overall sentiment

        provide the data in simple json format and not markdown.
        """
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model,
            temperature=0.5,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error analyzing transcription: {str(e)}")

def process_video(video_path: str, audio_path:str, model: str = "llama-3.3-70b-versatile") -> dict:
    """Main function to process video and return results"""
    try:
        # Process the video
        audio_file = extract_audio(video_path, audio_path)
        transcription_result = transcribe_audio(audio_file)
        analysis = analyze_transcription(transcription_result, model)
        
        return transcription_result, analysis
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")

def main():
    """Entry point of the script"""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Generate output paths
        audio_path = str(output_dir / "extracted_audio.mp3")

        video_path = "china_military_robot.mp4"
        transcription_result, analysis = process_video(video_path, audio_path)

        output_path = output_dir / "transcription_and_analysis.txt"

        # Save both transcription and analysis to the same file
        with open(output_path, "w", encoding="utf-8") as file:
            file.write("=== TRANSCRIPTION ===\n\n")
            file.write(transcription_result)
            file.write("\n\n=== ANALYSIS ===\n\n")
            file.write(analysis)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
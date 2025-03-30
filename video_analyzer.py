from pathlib import Path
from openai import OpenAI
import base64
from dotenv import load_dotenv
import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image


load_dotenv()

def extract_frames(video_path: str, interval_seconds: int = 1) -> list:
    """Extract frames from video file at specified interval in seconds"""
    try:
        frames = []
        video = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(video.get(cv2.CAP_PROP_FPS))
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Calculate frame indices for each second
        frame_indices = np.arange(0, total_frames, fps, dtype=int)
        
        for frame_index in frame_indices:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = video.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                
                # Display the frame using OpenCV
                # cv2.imshow('Frame', frame)  # Shows original BGR frame
                # cv2.waitKey(1000)  # Wait for 1 second before proceeding
                
                # Convert to base64
                buffered = BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                frames.append(img_str)
                
        video.release()
        cv2.destroyAllWindows()  # Clean up OpenCV windows
        return frames
    except Exception as e:
        raise Exception(f"Error extracting frames: {str(e)}")

def analyze_frame(client: OpenAI, frame_base64: str, model: str = "llama-3.2-11b-vision-preview") -> str:
    """Analyze a single frame using the vision model"""
    try:
        prompt = f"""Here is a base64 encoded image: {frame_base64}

        Please analyze this image and provide:
        1. Visual description of the scene
        2. Objects and entities detected
        3. Actions and movements visible
        4. Context and setting
        
        Please provide the analysis in simple JSON format."""
        
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model=model,
            temperature=0.5,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Error analyzing frame: {str(e)}")

def analyze_video(video_path: str, model: str = "llama-3.2-11b-vision-preview") -> list:
    """Analyze video content by analyzing key frames"""
    try:
        client = OpenAI(
            api_key=os.environ.get('GROQ_API_KEY'),
            base_url=os.environ.get('GROQ_API_BASE')
        )
        
        # Extract frames from video
        frames = extract_frames(video_path)
        
        print(len(frames))
        # Analyze each frame
        analyses = []
        for i, frame in enumerate(frames[1:2]):
            analysis = analyze_frame(client, frame, model)
            analyses.append(f"Frame {i+1}:\n{analysis}\n")
        
        # return analyses
    except Exception as e:
        raise Exception(f"Error analyzing video: {str(e)}")

def main():
    """Entry point of the script"""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Video file path
        video_path = "china_military_robot.mp4"
        
        # Analyze video frames
        analyses = analyze_video(video_path)
        
        # Save analysis to file
        # output_path = output_dir / "video_analysis.txt"
        # with open(output_path, "w", encoding="utf-8") as file:
        #     file.write("=== VIDEO ANALYSIS ===\n\n")
        #     file.write("\n".join(analyses))
            
        # print(f"Analysis saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 
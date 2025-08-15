# transcription_manager.py
"""Transcription management and conversation handling"""

import os
import whisper
from datetime import datetime
from typing import List, Optional
from config import MODEL_CONFIG, SYSTEM_CONFIG

class ConversationSession:
    """Manages a single conversation session with logging capabilities"""
    
    def __init__(self, session_id: Optional[str] = None, log_dir: str = "logs"):
        self.session_id = session_id or self._generate_session_id()
        self.chunks = []
        self.full_conversation = ""
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        self.events = []
        self.log_dir = log_dir
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(self.log_dir, f"session_{self.session_id}.log")
        self._log_event("session_started", {"session_id": self.session_id, "timestamp": self.start_time.isoformat()})
        
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def add_chunk(self, text: str):
        """Add a new chunk to the conversation"""
        if text and text.strip():
            chunk_data = {
                'text': text.strip(),
                'timestamp': datetime.now(),
                'chunk_id': len(self.chunks) + 1
            }
            self.chunks.append(chunk_data)
            self._update_full_conversation()
            self.last_update = datetime.now()
            
            # Log the new chunk
            self._log_event("chunk_added", {
                "chunk_id": chunk_data['chunk_id'],
                "text": chunk_data['text'],
                "timestamp": chunk_data['timestamp'].isoformat()
            })
    
    def update_last_chunk(self, text: str):
        """Update the last chunk (for ongoing speech)"""
        if text and text.strip():
            if self.chunks:
                chunk_data = self.chunks[-1]
                chunk_data['text'] = text.strip()
                chunk_data['timestamp'] = datetime.now()
                
                # Log the chunk update
                self._log_event("chunk_updated", {
                    "chunk_id": chunk_data['chunk_id'],
                    "text": chunk_data['text'],
                    "timestamp": chunk_data['timestamp'].isoformat()
                })
            else:
                self.add_chunk(text)
            self._update_full_conversation()
            self.last_update = datetime.now()
    
    def _update_full_conversation(self):
        """Update the full conversation string"""
        self.full_conversation = " ".join([chunk['text'] for chunk in self.chunks])
    
    def _log_event(self, event_type: str, data: dict):
        """Log an event to both console and file"""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.events.append(event)
        
        # Log to console
        print(f"\n[EVENT] {event_type.upper()}: {data}")
        
        # Log to file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"{event}\n")
        except Exception as e:
            print(f"[ERROR] Failed to write to log file: {e}")
    
    def get_session_summary(self) -> dict:
        """Get a summary of the current session"""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "last_update": self.last_update.isoformat(),
            "total_chunks": len(self.chunks),
            "total_chars": len(self.full_conversation),
            "chunks": [{
                "id": c['chunk_id'],
                "timestamp": c['timestamp'].isoformat(),
                "text": c['text']
            } for c in self.chunks],
            "full_text": self.full_conversation
        }
    
    def save_session_state(self):
        """Save the current session state to a file"""
        state_file = os.path.join(self.log_dir, f"session_{self.session_id}_state.json")
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                import json
                json.dump(self.get_session_summary(), f, indent=2)
            self._log_event("state_saved", {"file": state_file})
        except Exception as e:
            print(f"[ERROR] Failed to save session state: {e}")
    
    def clear_conversation(self):
        """Clear the conversation for a new session"""
        self.chunks.clear()
        self.full_conversation = ""
        self.start_time = datetime.now()
    
    def save_to_file(self):
        """Save conversation to file"""
        filename = os.path.join(SYSTEM_CONFIG.output_dir, f"conversation_{self.session_id}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Start Time: {self.start_time}\n")
            f.write(f"Last Update: {self.last_update}\n")
            f.write(f"Total Chunks: {len(self.chunks)}\n")
            f.write("="*50 + "\n\n")
            
            f.write("FULL CONVERSATION:\n")
            f.write(self.full_conversation + "\n\n")
            
            f.write("INDIVIDUAL CHUNKS:\n")
            for chunk in self.chunks:
                f.write(f"[{chunk['timestamp']}] Chunk {chunk['chunk_id']}: {chunk['text']}\n")

class TranscriptionManager:
    """Manages whisper transcription and conversation sessions with logging"""
    
    def __init__(self, log_dir: str = "logs", save_interval: int = 30):
        """Initialize the transcription manager
        
        Args:
            log_dir: Directory to store log files
            save_interval: Interval in seconds between automatic state saves
        """
        self.model = self._load_model()
        self.log_dir = log_dir
        self.save_interval = save_interval
        self.last_save_time = time.time()
        self.current_session = ConversationSession(log_dir=log_dir)
        self.phrase_bytes = bytes()
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
    def _load_model(self):
        """Load the Whisper model"""
        model_name = MODEL_CONFIG.model_name
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        print("Model loaded successfully!")
        return model
    
    def transcribe_audio(self, audio_np) -> Optional[str]:
        """Transcribe audio using Whisper"""
        if len(audio_np) == 0:
            return None
            
        try:
            result = self.model.transcribe(audio_np)
            text = result['text'].strip()
            return text if text else None
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
    
    def process_audio_chunk(self, audio_data: bytes, phrase_complete: bool = False):
        """Process an audio chunk and return transcribed text
        
        Args:
            audio_data: Raw audio data in bytes
            phrase_complete: Whether this chunk completes a phrase
            
        Returns:
            str: Transcribed text if available, None otherwise
        """
        self.phrase_bytes += audio_data
        
        if phrase_complete and len(self.phrase_bytes) > 0:
            try:
                # Convert audio data to numpy array for whisper
                audio_np = np.frombuffer(self.phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Transcribe the audio
                text = self.transcribe_audio(audio_np)
                
                if text and text.strip():
                    self.current_session.add_chunk(text.strip())
                    self.current_session.save_session_state()  # Save state after each complete phrase
                
                # Reset for next phrase
                self.phrase_bytes = bytes()
                return text
                
            except Exception as e:
                error_msg = f"Error processing audio: {e}"
                print(f"[ERROR] {error_msg}")
                self.current_session._log_event("processing_error", {"error": str(e), "context": "process_audio_chunk"})
                self.phrase_bytes = bytes()
                return None
        
        # If we're in the middle of a phrase, update the last chunk
        elif len(self.phrase_bytes) > 0:
            try:
                audio_np = np.frombuffer(self.phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                text = self.transcribe_audio(audio_np)
                
                if text and text.strip():
                    self.current_session.update_last_chunk(text.strip())
                    
                    # Periodically save state during long utterances
                    current_time = time.time()
                    if current_time - self.last_save_time >= self.save_interval:
                        self.current_session.save_session_state()
                        self.last_save_time = current_time
                
                return text
                
            except Exception as e:
                error_msg = f"Error updating chunk: {e}"
                print(f"[ERROR] {error_msg}")
                self.current_session._log_event("processing_error", {"error": str(e), "context": "update_chunk"})
                return None
                
        return None
    
    def start_new_session(self) -> str:
        """Start a new conversation session"""
        # Save current session if it has content
        if self.current_session.chunks:
            self.current_session.save_to_file()
        
        # Start new session
        self.current_session = ConversationSession()
        self.phrase_bytes = bytes()
        return self.current_session.session_id
    
    def get_session_info(self) -> dict:
        """Get information about current session"""
        return {
            'session_id': self.current_session.session_id,
            'chunk_count': len(self.current_session.chunks),
            'start_time': self.current_session.start_time,
            'last_update': self.current_session.last_update,
            'conversation_length': len(self.current_session.full_conversation)
        }
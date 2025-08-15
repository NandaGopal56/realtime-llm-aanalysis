"""Transcription management and conversation handling"""

import os
import whisper
from datetime import datetime
from typing import List, Optional
from config import MODEL_CONFIG, SYSTEM_CONFIG

class ConversationSession:
    """Manages a single conversation session"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or self._generate_session_id()
        self.chunks = []
        self.full_conversation = ""
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        
    def _generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def add_chunk(self, text: str):
        """Add a new chunk to the conversation"""
        if text and text.strip():
            self.chunks.append({
                'text': text.strip(),
                'timestamp': datetime.now(),
                'chunk_id': len(self.chunks) + 1
            })
            self._update_full_conversation()
            self.last_update = datetime.now()
    
    def update_last_chunk(self, text: str):
        """Update the last chunk (for ongoing speech)"""
        if text and text.strip():
            if self.chunks:
                self.chunks[-1]['text'] = text.strip()
                self.chunks[-1]['timestamp'] = datetime.now()
            else:
                self.add_chunk(text)
            self._update_full_conversation()
            self.last_update = datetime.now()
    
    def _update_full_conversation(self):
        """Update the full conversation string"""
        self.full_conversation = " ".join([chunk['text'] for chunk in self.chunks])
    
    def get_conversation_for_llm(self) -> str:
        """Get the full conversation formatted for LLM processing"""
        return self.full_conversation.strip()
    
    def clear_conversation(self):
        """Clear the conversation for a new session"""
        self.chunks.clear()
        self.full_conversation = ""
        self.start_time = datetime.now()
    

class TranscriptionManager:
    """Manages whisper transcription and conversation sessions"""
    
    def __init__(self, logger=None):
        self.model = self._load_model()
        self.current_session = ConversationSession()
        self.phrase_bytes = bytes()
        
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
    
    def process_audio_chunk(self, audio_data: bytes, phrase_complete: bool = False) -> Optional[str]:
        """Process an audio chunk and return transcribed text"""
        if phrase_complete:
            self.phrase_bytes = bytes()
        
        # Add new audio data to accumulated phrase
        self.phrase_bytes += audio_data
        
        # Convert to numpy array for whisper
        import numpy as np
        audio_np = np.frombuffer(self.phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe
        text = self.transcribe_audio(audio_np)
        
        if text:
            if phrase_complete:
                self.current_session.add_chunk(text)
            else:
                self.current_session.update_last_chunk(text)
        
        return text
    
    def start_new_session(self) -> str:
        """Start a new conversation session"""

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
            'conversation_length': len(self.current_session.full_conversation),
            'full_conversation': self.current_session.full_conversation
        }
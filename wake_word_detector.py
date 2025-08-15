# wake_word_detector.py
"""Wake word detection functionality"""

import time
from datetime import datetime, timedelta
from typing import List, Optional
from config import WAKE_WORD_CONFIG

class WakeWordDetector:
    """Handles wake word detection and activation state"""
    
    def __init__(self):
        self.wake_words = WAKE_WORD_CONFIG.wake_words
        self.confidence_threshold = WAKE_WORD_CONFIG.confidence_threshold
        self.timeout_duration = WAKE_WORD_CONFIG.timeout_after_wake
        
        self.is_active = False
        self.activation_time = None
        self.last_wake_detection = None
        
    def check_for_wake_word(self, text: str) -> bool:
        """
        Check if the transcribed text contains a wake word
        Returns True if wake word is detected
        """
        if not text:
            return False
            
        text_lower = text.lower().strip()
        
        for wake_word in self.wake_words:
            if wake_word.lower() in text_lower:
                print(f"🎤 Wake word detected: '{wake_word}'")
                self._activate()
                return True
        
        return False
    
    def _activate(self):
        """Activate the assistant"""
        self.is_active = True
        self.activation_time = datetime.now()
        self.last_wake_detection = datetime.now()
        print("✅ Assistant activated! Listening for commands...")
    
    def deactivate(self):
        """Deactivate the assistant"""
        self.is_active = False
        self.activation_time = None
        print("💤 Assistant deactivated. Waiting for wake word...")
    
    def update_activity(self):
        """Update activation state based on timeout"""
        if not self.is_active:
            return
            
        if self.activation_time:
            elapsed = datetime.now() - self.activation_time
            if elapsed.total_seconds() > self.timeout_duration:
                self.deactivate()
    
    def extend_activation(self):
        """Extend the activation time when new speech is detected"""
        if self.is_active:
            self.activation_time = datetime.now()
    
    def get_status(self) -> dict:
        """Get current status of the wake word detector"""
        status = {
            'is_active': self.is_active,
            'wake_words': self.wake_words,
        }
        
        if self.is_active and self.activation_time:
            elapsed = datetime.now() - self.activation_time
            remaining = max(0, self.timeout_duration - elapsed.total_seconds())
            status['time_remaining'] = remaining
            status['activation_time'] = self.activation_time
        
        return status
    
    def should_process_speech(self) -> bool:
        """Check if speech should be processed (assistant is active)"""
        self.update_activity()
        return self.is_active
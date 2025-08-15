"""Main voice assistant coordinator"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional

from audio_handler import AudioHandler
from wake_word_detector import WakeWordDetector
from transcription_manager import TranscriptionManager


class VoiceAssistant:
    """Main voice assistant that coordinates all components"""
    
    def __init__(self):
        self.audio_handler = AudioHandler()
        self.wake_word_detector = WakeWordDetector()
        self.transcription_manager = TranscriptionManager()
        
        # State management
        self.phrase_time = None
        self.phrase_timeout = 3.0  # seconds
        self.is_running = False
        
        # Periodic state saving
        self.last_state_save = datetime.now()
        self.state_save_interval = 5.0  # seconds
    
    def _display_status(self):
        """Display current status and conversation"""
        
        # Wake word status
        wake_status = self.wake_word_detector.get_status()
        if wake_status['is_active']:
            remaining = wake_status.get('time_remaining', 0)
            print(f"✅ Status: ACTIVELY LISTENING (⏰ {remaining:.1f}s remaining)")
        else:
            print(f"💤 Status: WAITING for wake word. Say: {' or '.join(wake_status['wake_words'])}")
        
        print()
        
        # Session info
        session = self.transcription_manager.get_session_info()
        print(session)
        
        if wake_status['is_active']:
            print()
            print("💬 CURRENT CONVERSATION:")
            print("-" * 30)
            
            # Show individual chunks
            for chunk in self.transcription_manager.current_session.chunks:
                timestamp = chunk['timestamp'].strftime('%H:%M:%S')
                print(f"[{timestamp}] {chunk['text']}")
            
            if not self.transcription_manager.current_session.chunks:
                print("(Listening for your command...)")
                
            # Show full conversation if available
            if session_info['full_conversation']:
                print("\n" + "-" * 30)
                print("📝 FULL CONVERSATION:")
                print(session_info['full_conversation'])
        
        print("\n" + "=" * 50)
        print("Press Ctrl+C to exit")
    
    def _process_transcription(self, text: str, phrase_complete: bool):
        """Process transcribed text"""
            
        # Check for wake word if not active
        if not self.wake_word_detector.is_active:
            if self.wake_word_detector.check_for_wake_word(text):
                session_id = self.transcription_manager.start_new_session()
            return
        
        # If assistant is active, extend activation and log conversation updates
        self.wake_word_detector.extend_activation()
        
    
    def run(self):
        """Main run loop"""
        if not self.audio_handler.source:
            print(f"❌ No microphone available. Exiting.")
            return
        
        # Start background audio recording
        self.audio_handler.start_listening()
        self.is_running = True
        
        # Initial display
        self._display_status()
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                # Check for audio data
                audio_data = self.audio_handler.get_audio_data()
                
                if audio_data:
                    phrase_complete = False
                    
                    # Check if enough time has passed for a complete phrase
                    if (self.phrase_time and 
                        current_time - self.phrase_time > timedelta(seconds=self.phrase_timeout)):
                        phrase_complete = True
                    
                    self.phrase_time = current_time
                    
                    # Process the audio
                    text = self.transcription_manager.process_audio_chunk(
                        audio_data, phrase_complete
                    )
                    
                    if text:
                        self._process_transcription(text, phrase_complete)
                        self._display_status()
                
                else:
                    # No audio data, just update display and sleep
                    time.sleep(0.25)
                    
                    # Update wake word detector (handles timeout)
                    prev_active = self.wake_word_detector.is_active
                    self.wake_word_detector.update_activity()
                    
                    # Refresh display if status changed
                    if prev_active != self.wake_word_detector.is_active:
                        self._display_status()
                        
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the voice assistant"""
        print("\n🛑 Stopping SARS Voice Assistant...")
        self.is_running = False
        
        # Save current session
        if self.transcription_manager.current_session.chunks:
            session_id = self.transcription_manager.current_session.session_id
            conversation = self.transcription_manager.current_session.full_conversation
        
        # Final status summary
        print("👋 Goodbye!")
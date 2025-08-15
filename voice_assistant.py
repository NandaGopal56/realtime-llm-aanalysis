# voice_assistant.py
"""Main voice assistant coordinator"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, Callable

from audio_handler import AudioHandler
from wake_word_detector import WakeWordDetector
from transcription_manager import TranscriptionManager

class VoiceAssistant:
    """Main voice assistant that coordinates all components with logging"""
    
    def __init__(self, log_dir: str = "logs", save_interval: int = 30):
        """Initialize the voice assistant with logging capabilities
        
        Args:
            log_dir: Directory to store log files
            save_interval: Interval in seconds between automatic state saves
        """
        self.audio_handler = AudioHandler()
        self.wake_word_detector = WakeWordDetector()
        self.transcription_manager = TranscriptionManager(log_dir=log_dir, save_interval=save_interval)
        
        # State management
        self.phrase_time = None
        self.phrase_timeout = 3.0  # seconds
        self.is_running = False
        self.log_dir = log_dir
        self.last_wake_word_time = 0
        self.is_awake = False
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
    def _clear_console(self):
        """Clear the console"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def _log_wake_word_detected(self, wake_word: str):
        """Log when a wake word is detected"""
        self.is_awake = True
        self.last_wake_word_time = time.time()
        
        # Log to console
        print(f"\n🔔 WAKE WORD DETECTED: '{wake_word}'")
        
        # Log to session
        self.transcription_manager.current_session._log_event("wake_word_detected", {
            "wake_word": wake_word,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.transcription_manager.current_session.session_id
        })
        
        # Start a new session when wake word is detected
        self.transcription_manager.start_new_session()
    
    def _check_wake_word_timeout(self):
        """Check if we should go back to sleep due to inactivity"""
        if not self.is_awake:
            return False
            
        current_time = time.time()
        time_since_last_wake = current_time - self.last_wake_word_time
        
        if time_since_last_wake > self.phrase_timeout:
            self.is_awake = False
            
            # Log sleep event
            self.transcription_manager.current_session._log_event("going_to_sleep", {
                "inactive_seconds": time_since_last_wake,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save session state when going to sleep
            self.transcription_manager.current_session.save_session_state()
            
            return True
        return False
    
    def _display_status(self):
        """Display current status and conversation"""
        self._clear_console()
        
        # Header
        print("🤖 SARS - Voice Assistant")
        print("=" * 70)
        
        # Wake word status
        wake_status = self.wake_word_detector.get_status()
        
        # Check if we should go to sleep due to inactivity
        if self.is_awake:
            time_since_wake = time.time() - self.last_wake_word_time
            time_until_sleep = max(0, self.phrase_timeout - time_since_wake)
            
            if time_until_sleep <= 0:
                self.is_awake = False
                print("💤 Status: INACTIVE (sleeping due to inactivity)")
            else:
                print(f"✅ Status: ACTIVE (⏰ {time_until_sleep:.1f}s until sleep)")
        else:
            print("💤 Status: WAITING for wake word")
        
        # Display wake words
        print(f"🎯 Wake words: {', '.join(wake_status['wake_words'])}")
        print("-" * 70)
        
        # Session info
        session_info = self.transcription_manager.get_session_info()
        
        # Display session information
        if 'status' in session_info and session_info['status'] == 'active':
            print(f"📝 Session: {session_info['session_id']}")
            print(f"🕒 Started: {session_info['start_time']}")
            print(f"💬 Chunks: {session_info['total_chunks']} | Characters: {session_info['total_chars']}")
            print("-" * 70)
            
            # Display recent chunks
            if session_info['chunks']:
                print("📜 Recent transcriptions:")
                for chunk in session_info['chunks'][-5:]:  # Show last 5 chunks
                    print(f"   • {chunk['text']}")
                print()
            
            # Display full conversation (truncated if too long)
            if session_info['full_text']:
                full_text = session_info['full_text']
                if len(full_text) > 200:  # Truncate if too long
                    full_text = full_text[:197] + "..."
                print(f"📜 Full text: {full_text}")
        else:
            print("No active session. Say a wake word to begin.")
        
        print("=" * 70)
        print(f"📝 Session: {session_info['session_id']}")
        print(f"📊 Chunks: {session_info['chunk_count']}")
        
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
        
        print("\n" + "=" * 50)
        print("Press Ctrl+C to exit")
    
    def _process_transcription(self, text: str, phrase_complete: bool):
        """Process transcribed text"""
        if not text:
            return
            
        # Check for wake word if not active
        if not self.wake_word_detector.is_active:
            if self.wake_word_detector.check_for_wake_word(text):
                # Wake word detected, start new session
                self.transcription_manager.start_new_session()
            return
        
        # If assistant is active, extend activation and process speech
        self.wake_word_detector.extend_activation()
        
    
    def run(self):
        """Main run loop"""
        if not self.audio_handler.source:
            print("❌ No microphone available. Exiting.")
            return
        print("🚀 Starting SARS Voice Assistant...")
        print("📡 Initializing audio recording...")
        
        # Start background audio recording
        self.audio_handler.start_listening()
        
        self._display_status()
        
        try:
            while self.is_running:
                # Get audio data from microphone
                audio_data = self.audio_handler.get_audio()
                
                # Check for wake word
                wake_word_detected = self.wake_word_detector.detect_wake_word(audio_data)
                
                # Handle wake word detection
                if wake_word_detected:
                    self._log_wake_word_detected(wake_word_detected)
                
                # Check if we should go to sleep due to inactivity
                self._check_wake_word_timeout()
                
                # Process audio for transcription if wake word detected or system is active
                if self.is_awake or self.wake_word_detector.is_active():
                    # Check if this is a new phrase
                    now = time.time()
                    if self.phrase_time and now - self.phrase_time > self.phrase_timeout:
                        phrase_complete = True
                    else:
                        phrase_complete = False
                    
                    # Process the audio chunk
                    text = self.transcription_manager.process_audio_chunk(
                        audio_data, 
                        phrase_complete=phrase_complete
                    )
                    
                    # Update phrase time if we got text
                    if text:
                        self.phrase_time = now
                        self.last_wake_word_time = now  # Reset inactivity timer
                    
                    # Update display periodically (throttle to avoid excessive updates)
                    if now % 1 < 0.1:  # Update about once per second
                        self._display_status()
                
                # Small sleep to prevent high CPU usage
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the voice assistant"""
        print("\n🛑 Stopping SARS Voice Assistant...")
        self.is_running = False
        
        # Save current session
        if self.transcription_manager.current_session.chunks:
            self.transcription_manager.current_session.save_to_file()
            print(f"💾 Session saved: {self.transcription_manager.current_session.session_id}")
        
        print("👋 Goodbye!")
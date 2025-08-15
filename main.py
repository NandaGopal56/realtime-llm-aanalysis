import argparse
import sys
from sys import platform

from config import AUDIO_CONFIG, MODEL_CONFIG, WAKE_WORD_CONFIG, SYSTEM_CONFIG
from voice_assistant import VoiceAssistant

def setup_argument_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="SARS - Voice Assistant with Wake Word Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        python main.py                                  # Use default settings
        python main.py --model large                    # Use large model
        python main.py --wake-words "hey sars" "sars"   # Custom wake words
        python main.py --energy 500                     # Lower energy threshold
        python main.py --list-mics                      # List available microphones (Linux)
        """
    )
    
    # Model configuration
    parser.add_argument(
        "--model", 
        default=MODEL_CONFIG.model_size,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to use (default: %(default)s)"
    )
    parser.add_argument(
        "--non-english", 
        action='store_true',
        help="Don't use the English-specific model"
    )
    
    # Audio configuration
    parser.add_argument(
        "--energy", 
        type=int, 
        default=AUDIO_CONFIG.energy_threshold,
        help="Energy threshold for microphone detection (default: %(default)s)"
    )
    parser.add_argument(
        "--record-timeout", 
        type=float, 
        default=AUDIO_CONFIG.record_timeout,
        help="Recording timeout in seconds (default: %(default)s)"
    )
    parser.add_argument(
        "--phrase-timeout", 
        type=float, 
        default=AUDIO_CONFIG.phrase_timeout,
        help="Phrase timeout in seconds (default: %(default)s)"
    )
    
    # Wake word configuration
    parser.add_argument(
        "--wake-words", 
        nargs='+', 
        default=WAKE_WORD_CONFIG.wake_words,
        help="Wake words to activate assistant (default: %(default)s)"
    )
    parser.add_argument(
        "--wake-timeout", 
        type=float, 
        default=WAKE_WORD_CONFIG.timeout_after_wake,
        help="Seconds to stay active after wake word (default: %(default)s)"
    )
    
    # System configuration
    parser.add_argument(
        "--output-dir", 
        default=SYSTEM_CONFIG.output_dir,
        help="Output directory for transcriptions (default: %(default)s)"
    )
    
    # Linux-specific options
    if 'linux' in platform:
        parser.add_argument(
            "--default-microphone", 
            default=SYSTEM_CONFIG.default_microphone,
            help="Default microphone name for Linux (default: %(default)s)"
        )
        parser.add_argument(
            "--list-mics", 
            action='store_true',
            help="List available microphones and exit"
        )
    
    return parser

def update_config_from_args(args):
    """Update global configuration based on command line arguments"""
    # Update model config
    MODEL_CONFIG.model_size = args.model
    MODEL_CONFIG.non_english = args.non_english
    
    # Update audio config
    AUDIO_CONFIG.energy_threshold = args.energy
    AUDIO_CONFIG.record_timeout = args.record_timeout
    AUDIO_CONFIG.phrase_timeout = args.phrase_timeout
    
    # Update wake word config
    WAKE_WORD_CONFIG.wake_words = args.wake_words
    WAKE_WORD_CONFIG.timeout_after_wake = args.wake_timeout
    
    # Update system config
    SYSTEM_CONFIG.output_dir = args.output_dir
    
    # Linux-specific
    if 'linux' in platform:
        SYSTEM_CONFIG.default_microphone = args.default_microphone


def main():
    """Main entry point with enhanced logging and session management"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Handle special cases
    if 'linux' in platform and args.list_mics:
        from audio_handler import AudioHandler
        handler = AudioHandler()
        handler._list_microphones()
        return
    
    # Update configuration
    update_config_from_args(args)
    
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    print("🚀 Starting SARS Voice Assistant with enhanced logging...")
    print(f"📂 Logs will be saved to: {os.path.abspath(log_dir)}")
    # Print startup information
    print("🤖 SARS Voice Assistant")
    print("=" * 40)
    print(f"📝 Model: {MODEL_CONFIG.model_name}")
    print(f"🎤 Energy Threshold: {AUDIO_CONFIG.energy_threshold}")
    print(f"⏱️ Record Timeout: {AUDIO_CONFIG.record_timeout}s")
    print(f"⏳ Phrase Timeout: {AUDIO_CONFIG.phrase_timeout}s")
    print(f"🎯 Wake Words: {', '.join(WAKE_WORD_CONFIG.wake_words)}")
    print(f"⏰ Wake Timeout: {WAKE_WORD_CONFIG.timeout_after_wake}s")
    print(f"📁 Output Directory: {SYSTEM_CONFIG.output_dir}")
    print("=" * 40)
    print()
    
    try:
        # Create and configure voice assistant
        assistant = VoiceAssistant()
        
        # Start the assistant
        assistant.run()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
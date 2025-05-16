"""
Example of a virtual telephone operator using real-time voice chat.
"""

import logging
import time

from realtime_voice_chat.core.audio_handler import SoundDeviceAudioHandler
from realtime_voice_chat.core.audio_processor import Float32AudioProcessor
from realtime_voice_chat.core.openai_client import OpenAIRealtimeClient

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# System instructions for the telephone operator
TELEPHONE_OPERATOR_INSTRUCTION = """You are a professional telephone operator. Your role is to:
1. Greet callers professionally and identify their needs
2. Handle common requests like call transfers, directory assistance, and general inquiries
3. Maintain a calm and helpful demeanor
4. Use clear and concise language
5. Confirm understanding before taking action
6. End calls professionally

Remember to:
- Speak clearly and at a moderate pace
- Listen carefully to caller requests
- Be patient and understanding
- Follow proper telephone etiquette
- Keep responses brief and to the point"""


class TelephoneOperator:
    """Virtual telephone operator using real-time voice chat."""

    def __init__(self):
        """Initialize the telephone operator."""
        # Initialize audio handler with float32 processor
        self.audio_handler = SoundDeviceAudioHandler(processor=Float32AudioProcessor())

        # Initialize OpenAI client with float32 processor
        self.openai_client = OpenAIRealtimeClient(audio_handler=self.audio_handler)

    def start(self):
        """Start the telephone operator."""
        try:
            # Connect to OpenAI
            self.openai_client.connect()

            # Set system instructions
            self.openai_client.set_instructions(TELEPHONE_OPERATOR_INSTRUCTION)

            # Start audio streams
            self.audio_handler.start_input_stream()
            self.audio_handler.start_output_stream()
            logger.info("Telephone operator started. Press Ctrl+C to stop.")
            while True:
                chunk = self.audio_handler.get_input_audio()
                if chunk:
                    self.openai_client.send_audio(chunk)
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping telephone operator...")
        except Exception as e:
            logger.error(f"Error running telephone operator: {e}")
        finally:
            self.close()

    def close(self):
        """Close the telephone operator."""
        try:
            # Save recorded audio
            self.openai_client.save_audio("telephone_operator_output.wav")
            self.audio_handler.save_input_audio("telephone_operator_input.wav")
            # Close connections
            self.audio_handler.close()
            self.openai_client.close()

        except Exception as e:
            logger.error(f"Error closing telephone operator: {e}")


if __name__ == "__main__":
    operator = TelephoneOperator()
    operator.start()

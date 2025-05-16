"""
OpenAI WebSocket client for real-time voice chat.
Handles communication with OpenAI's real-time API.
"""

import base64
import json
import logging
import os
import threading
import time
import wave
from queue import Queue
from typing import Optional

import websocket

from realtime_voice_chat.core import AudioHandler

logger = logging.getLogger(__name__)


class OpenAIRealtimeClient:
    """
    Client for OpenAI's real-time WebSocket API.

    Attributes:
        api_key (str): OpenAI API key.
        ws (websocket.WebSocketApp): WebSocket connection.
        connected (bool): Connection status.
        ws_thread (Thread): Thread running the WebSocket connection.
        audio_handler (AudioHandler): Handler for processing audio data.
        raw_b64_chunks (List[str]): List of received base64 audio chunks.
    """

    def __init__(
        self, audio_handler: AudioHandler | Queue, api_key: Optional[str] = None
    ):
        """
        Initialize the OpenAI client.

        Args:
            api_key (str, optional): OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable.
            processor_type (str): Type of audio processor to use ('float32' or 'int16')
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
            )

        self.ws = None
        self.connected = False
        self.audio_handler = audio_handler
        self.raw_b64_chunks = []

    def connect(self) -> None:
        """Connect to OpenAI's WebSocket API."""
        url = (
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
        )
        headers = [f"Authorization: Bearer {self.api_key}", "OpenAI-Beta: realtime=v1"]

        self.ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread( # pylint: disable=W0201
            target=self.ws.run_forever
        )
        self.ws_thread.daemon = True
        self.ws_thread.start()

        # Wait for connection
        time.sleep(2)
        if not self.connected:
            raise ConnectionError("Failed to connect to OpenAI WebSocket API")

    def set_instructions(self, instructions: str) -> None:
        """
        Set the system instructions for the conversation.

        Args:
            instructions (str): The system instructions to set.
        """
        if not self.connected:
            raise ConnectionError("Not connected to OpenAI WebSocket API")

        self.ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "instructions": instructions,
                        "turn_detection": {"type": "server_vad", "threshold": 0.1},
                    },
                }
            )
        )
        logger.info("System instructions set")

    def send_audio(self, audio_data: str) -> None:
        """
        Send audio data to OpenAI.

        Args:
            audio_data (str): Base64 encoded audio data.
        """
        if not self.connected:
            raise ConnectionError("Not connected to OpenAI WebSocket API")

        self.ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": audio_data})
        )
        logger.debug(f"Sent audio data of length {len(audio_data)}")

    def save_audio(self, filename: str = "output.wav") -> None:
        """
        Save the received audio to a WAV file.

        Args:
            filename (str): The name of the WAV file to save to.
        """
        if not self.raw_b64_chunks:
            logger.warning("No audio received to save")
            return

        # Combine all chunks
        combined = "".join(self.raw_b64_chunks)

        try:
            # Decode base64 to bytes
            raw_bytes = base64.b64decode(combined)

            # Save as WAV
            with wave.open(filename, "wb") as wf:
                # pylint: disable=E1101
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(24000)  # 24kHz sample rate
                wf.writeframes(raw_bytes)

            logger.info(f"Saved audio to {filename}")

        except Exception as e:
            logger.error(f"Failed to save audio: {e}")

    def close(self) -> None:
        """Close the WebSocket connection."""
        if self.ws:
            self.ws.close()
            self.connected = False

    def _on_open(self, ws) -> None:
        """WebSocket on_open callback."""
        logger.info("Connected to OpenAI server")
        self.connected = True

    def _on_message(self, ws, message: str) -> None:
        """WebSocket on_message callback."""
        try:
            data = json.loads(message)

            if data.get("type") == "response.audio.delta":
                delta = data.get("delta", "")
                if not isinstance(delta, str):
                    logger.warning(f"Unexpected delta format: {type(delta)}")
                    return

                self.raw_b64_chunks.append(delta)  # still keep it for saving
                try:
                    # Process the audio data using the processor
                    if isinstance(self.audio_handler, AudioHandler):
                        self.audio_handler.add_audio_data(delta)
                    elif isinstance(self.audio_handler, Queue):
                        self.audio_handler.put(delta)
                    logger.debug(f"Received audio data of length: {len(delta)} samples")

                except Exception as e:
                    logger.error(f"Failed to process audio delta: {e}")

            elif data.get("type") == "response.audio.done":
                logger.info("Audio stream completed")

            else:
                logger.debug(f"Received message: {json.dumps(data, indent=2)}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    def _on_error(self, ws, error) -> None:
        """WebSocket on_error callback."""
        logger.error(f"WebSocket error: {error}")
        self.connected = False

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """WebSocket on_close callback."""
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.connected = False

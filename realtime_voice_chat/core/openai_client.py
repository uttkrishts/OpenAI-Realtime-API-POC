"""
OpenAI WebSocket client for real-time voice chat.
Handles communication with OpenAI's real-time API.
Allows complete override of default event handlers via custom callbacks for on_open, on_message, on_error, and on_close events.
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

from realtime_voice_chat.schemas.event import InputAudioBufferAppendEvent, BaseEvent
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

    Custom Callbacks:
        on_open_callback (callable): If provided, overrides the default on_open behavior.
        on_message_callback (callable): If provided, overrides the default on_message behavior.
        on_error_callback (callable): If provided, overrides the default on_error behavior.
        on_close_callback (callable): If provided, overrides the default on_close behavior.
    """

    def __init__(
        self,
        audio_handler: AudioHandler | Queue,
        openai_model: Optional[str] = "gpt-4o-realtime-preview-2024-12-17",
        api_key: Optional[str] = None,
        on_open_callback: Optional[callable] = None,
        on_message_callback: Optional[callable] = None,
        on_error_callback: Optional[callable] = None,
        on_close_callback: Optional[callable] = None,
    ):
        """
        Initialize the OpenAI client.

        Args:
            audio_handler (AudioHandler|Queue): Handler for processing audio data.
            openai_model (str, optional): Model identifier. Default is "gpt-4o-realtime-preview-2024-12-17".
            api_key (str, optional): OpenAI API key. If not provided, will use the OPENAI_API_KEY environment variable.
            on_open_callback (callable, optional): Custom callback to override on_open event.
            on_message_callback (callable, optional): Custom callback to override on_message event.
            on_error_callback (callable, optional): Custom callback to override on_error event.
            on_close_callback (callable, optional): Custom callback to override on_close event.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
            )
        self.openai_model = openai_model
        self.ws = None
        self.connected = False
        self.audio_handler = audio_handler
        self.raw_b64_chunks = []
        # Save custom callbacks
        self._custom_on_open = on_open_callback
        self._custom_on_message = on_message_callback
        self._custom_on_error = on_error_callback
        self._custom_on_close = on_close_callback

    def connect(self) -> None:
        """Connect to OpenAI's WebSocket API."""
        url = f"wss://api.openai.com/v1/realtime?model={self.openai_model}"
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
        self.ws_thread = threading.Thread(  # pylint: disable=W0201
            target=self.ws.run_forever
        )
        self.ws_thread.daemon = True
        self.ws_thread.start()

        # Wait for connection
        time.sleep(2)
        if not self.connected:
            raise ConnectionError("Failed to connect to OpenAI WebSocket API")

    def send_event(self, event: BaseEvent) -> None:
        """
        Emit a event to server.

        Args:
            event (BaseEvent): The event object to be sent.
        """
        if not self.connected:
            raise ConnectionError("Not connected to OpenAI WebSocket API")
        self.ws.send(event.model_dump_json())
        logger.info(f"{event.__class__.__name__} event sent: {event.model_dump_json()}")

    def send_audio(self, audio_data: str) -> None:
        """
        Send audio data to OpenAI.

        Args:
            audio_data (str): Base64 encoded audio data.
        """
        if not self.connected:
            raise ConnectionError("Not connected to OpenAI WebSocket API")

        audio_event = InputAudioBufferAppendEvent(audio=audio_data)
        self.ws.send(audio_event.model_dump_json())
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
        if self._custom_on_open:
            self._custom_on_open(ws)
        logger.info("Connected to OpenAI server")
        self.connected = True

    def _on_message(self, ws, message: str) -> None:
        try:
            if self._custom_on_message:
                self._custom_on_message(ws, message)
                return
            
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
        if self._custom_on_error:
            self._custom_on_error(ws, error)
        logger.error(f"WebSocket error: {error}")
        self.connected = False

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        if self._custom_on_close:
            self._custom_on_close(ws, close_status_code, close_msg)
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.connected = False

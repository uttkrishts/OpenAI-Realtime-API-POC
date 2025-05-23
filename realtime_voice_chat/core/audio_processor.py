"""
Audio processor module for real-time voice chat.
Handles audio data format conversion and encoding/decoding.
"""

import base64
import logging
import audioop
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AudioProcessor(ABC):
    """Abstract base class for audio processors."""

    @abstractmethod
    def encode(self, audio_data: np.ndarray) -> str:
        """Encode audio data to base64 string."""
        pass

    @abstractmethod
    def decode(self, encoded_data: str) -> np.ndarray:
        """Decode base64 string to audio data."""
        pass


class Float32AudioProcessor(AudioProcessor):
    """Processor for float32 audio data."""

    def encode(self, audio_data: np.ndarray) -> str:
        """Encode float32 audio data to base64 string."""
        clipped = np.clip(audio_data, -1.0, 1.0)
        pcm16 = (clipped * 32767).astype(np.int16)
        raw = pcm16.tobytes()
        encoded = base64.b64encode(raw).decode("ascii")
        return encoded

    def decode(self, encoded_data: str) -> np.ndarray:
        """Decode base64 string to float32 audio data."""
        padding = (4 - len(encoded_data) % 4) % 4
        padded_delta = encoded_data + "=" * padding
        # Decode the audio data
        raw_bytes = base64.b64decode(padded_delta)
        # First convert to int16, then to float32
        audio_data = np.frombuffer(raw_bytes, dtype=np.int16)
        # Convert to float32 in range [-1, 1]
        audio_data = audio_data.astype(np.float32) / 32767.0
        return audio_data


class G711uLawAudioProcessor(AudioProcessor):
    """Processor for G.711 µ-law audio data."""

    def encode(self, audio_data: np.ndarray) -> str:
        """
        Encode float32 audio data to G.711 μ-law base64 string.
        Converts float32 to int16 PCM, then to μ-law.
        """
        # Convert float32 audio data to int16 PCM
        clipped = np.clip(audio_data, -1.0, 1.0)
        pcm16 = (clipped * 32767).astype(np.int16)
        pcm16_bytes = pcm16.tobytes()
        # Convert PCM16 to G.711 uLaw bytes
        ulaw_bytes = audioop.lin2ulaw(pcm16_bytes, 2)
        encoded = base64.b64encode(ulaw_bytes).decode("ascii")
        return encoded

    def decode(self, encoded_data: str) -> np.ndarray:
        """
        Decode a G.711 μ-law base64 string to float32 audio data.
        Base64 decodes, then converts μ-law bytes to PCM int16, and finally to float32.
        """
        # Base64 decode to μ-law bytes
        ulaw_bytes = base64.b64decode(encoded_data)
        # Convert μ-law bytes to PCM int16 bytes
        pcm16_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
        # Convert bytes to numpy int16 array
        pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
        # Convert to float32 and scale to [-1,1]
        audio_data = pcm16.astype(np.float32) / 32767.0
        return audio_data


class AudioQueueProcessor:
    """Processor for managing audio data in queues."""

    def __init__(
        self,
        processor: AudioProcessor,
        input_callback: Optional[callable] = None,
        output_callback: Optional[callable] = None,
    ):
        """
        Initialize the queue processor.

        Args:
            processor (AudioProcessor): The audio processor to use
            input_callback (callable, optional): Callback for input processing
            output_callback (callable, optional): Callback for output processing
        """
        self.processor = processor
        self.input_callback = input_callback
        self.output_callback = output_callback

    def process_input(self, audio_data: np.ndarray) -> str:
        """
        Process input audio data.

        Args:
            audio_data (np.ndarray): Input audio data

        Returns:
            str: Processed audio data
        """
        return self.processor.encode(audio_data)

    def process_output(self, encoded_data: str) -> np.ndarray:
        """
        Process output audio data.

        Args:
            encoded_data (str): Encoded audio data

        Returns:
            np.ndarray: Processed audio data
        """
        return self.processor.decode(encoded_data)

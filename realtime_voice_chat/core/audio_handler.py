"""
Audio handling module for real-time voice chat.
Provides interfaces for audio input/output operations.
"""

import logging
import wave
from abc import ABC, abstractmethod
from queue import Queue
from typing import Callable, Dict, Optional, Union

import numpy as np
import sounddevice as sd

from realtime_voice_chat.core.audio_processor import (
    AudioProcessor,
    Float32AudioProcessor,
)

logger = logging.getLogger(__name__)


class AudioDeviceError(Exception):
    """Exception raised for audio device related errors."""


class AudioHandler(ABC):  # pylint: disable=R0902
    """Abstract base class for audio handling operations."""

    def __init__(
        self,
        processor: AudioProcessor = Float32AudioProcessor(),
        sample_rate: int = 24000,
        channels: int = 1,
        dtype: str = "float32",
        input_queue_callback: Optional[Callable] = None,
        output_queue_callback: Optional[Callable] = None,
    ):
        """
        Initialize the audio handler.

        Args:
            input_device (str, optional): Name of the input device to use
            output_device (str, optional): Name of the output device to use
            sample_rate (int): Audio sample rate in Hz
            channels (int): Number of audio channels
            dtype (str): Audio data type
            processor_type (str): Type of audio processor to use
            input_queue_callback (Callable, optional): Callback for input audio processing
            output_queue_callback (Callable, optional): Callback for output audio processing
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.processor = processor
        self.input_queue_callback = input_queue_callback
        self.output_queue_callback = output_queue_callback

        # Initialize streams
        self.input_stream = None
        self.output_stream = None

        # Initialize audio data storage
        self.raw_input_chunks = []
        self.current_audio_data = np.array([], dtype=np.float32)

        # Initialize queues
        self.input_queue = Queue()
        self.output_queue = Queue()

    @abstractmethod
    def start_input_stream(self) -> None:
        """Start the audio input stream."""
        pass

    @abstractmethod
    def start_output_stream(self) -> None:
        """Start the audio output stream."""
        pass

    @abstractmethod
    def add_audio_data(self, audio_data: Union[np.ndarray, bytes, str]) -> None:
        """Add new audio data to the playback buffer."""
        pass

    @abstractmethod
    def get_input_audio(self) -> Optional[Union[np.ndarray, bytes, str]]:
        """Get the next chunk of input audio."""
        pass

    @abstractmethod
    def save_input_audio(self, filename: str) -> None:
        """Save the recorded input audio to a file."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close all audio streams."""
        pass


class SoundDeviceAudioHandler(AudioHandler):
    """
    Handles audio input/output using sounddevice.

    Attributes:
        sample_rate (int): Audio sample rate in Hz
        channels (int): Number of audio channels
        dtype (str): Audio data type
        input_stream (sd.InputStream): Input audio stream
        output_stream (sd.OutputStream): Output audio stream
        input_queue (Queue): Queue for input audio data
        output_queue (Queue): Queue for output audio data
        raw_input_chunks (List[np.ndarray]): List of raw input audio chunks
        current_audio_data (np.ndarray): Current audio data being played
        processor_type (AudioProcessor): Audio processor to use
        input_queue_callback (Callable): Callback for input audio processing
        output_queue_callback (Callable): Callback for output audio processing
    """

    def __init__(
        self,
        processor: AudioProcessor = Float32AudioProcessor(),
        sample_rate: int = 24000,
        channels: int = 1,
        dtype: str = "float32",
        input_queue_callback: Optional[Callable] = None,
        output_queue_callback: Optional[Callable] = None,
    ):
        """
        Initialize the audio handler.

        Args:
            input_device (str, optional): Name of the input device to use
            output_device (str, optional): Name of the output device to use
            sample_rate (int): Audio sample rate in Hz
            channels (int): Number of audio channels
            dtype (str): Audio data type
            processor_type (str): Type of audio processor to use
            input_queue_callback (Callable, optional): Callback for input audio processing
            output_queue_callback (Callable, optional): Callback for output audio processing
        """
        super().__init__(
            processor,
            sample_rate,
            channels,
            dtype,
            input_queue_callback,
            output_queue_callback,
        )
        # Initialize audio devices
        self._input_device = self._get_input_device()
        self._output_device = self._get_output_device()

    def _get_input_device(self) -> Optional[str]:
        """
        Get the input device to use.

        Args:
            device_name (str, optional): Name of the input device to use

        Returns:
            str: Name of the input device to use
        """

        devices = sd.query_devices()
        for device in devices:
            if device["max_input_channels"] > 0:
                logger.info(f"Using input device: {device['name']}")
                return device["name"]

        logger.warning("No input device found")
        return None

    def _get_output_device(self) -> Optional[str]:
        """
        Get the output device to use.

        Args:
            device_name (str, optional): Name of the output device to use

        Returns:
            str: Name of the output device to use
        """
        devices = sd.query_devices()
        for device in devices:
            if device["max_output_channels"] > 0:
                logger.info(f"Using output device: {device['name']}")
                return device["name"]

        logger.warning("No output device found")
        return None

    def start_input_stream(self) -> None:
        """Start the input audio stream."""
        if not self._input_device:
            logger.warning("No input device available")
            return

        try:
            self.input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=self.sample_rate,
                callback=self._input_callback,
            )
            self.input_stream.start()
            logger.info("Microphone input stream started")
        except Exception as e:
            logger.error(f"Failed to start input stream: {e}")
            raise AudioDeviceError(f"Failed to start input stream: {e}")

    def start_output_stream(self) -> None:
        """Start the output audio stream."""
        if not self._output_device:
            logger.warning("No output device available")
            return

        try:
            self.output_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=2400,
                callback=self._output_callback,
            )
            self.output_stream.start()
            logger.info("Audio output stream started")
        except Exception as e:
            logger.error(f"Failed to start output stream: {e}")
            raise AudioDeviceError(f"Failed to start output stream: {e}")

    def _input_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Dict[str, float],
        status: sd.CallbackFlags,
    ) -> None:
        """
        Callback for input audio processing.

        Args:
            indata (np.ndarray): Input audio data
            frames (int): Number of frames
            time_info (Dict[str, float]): Timing information
            status (sd.CallbackFlags): Status flags
        """
        if status:
            logger.warning(f"Input stream status: {status}")
        flat = indata[:, 0] if indata.ndim > 1 else indata
        # Call user callback if provided
        try:
            # Store it before encoding to save input audio.
            self.raw_input_chunks.append(flat.copy())
            encoded_chunk = self.processor.encode(flat.copy())
            # Push the encoded chunk to the input queue
            self.input_queue.put(encoded_chunk)
        except Exception as e:
            logger.error(f"Error in input callback: {e}")

    def _output_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Dict[str, float],
        status: sd.CallbackFlags,
    ) -> None:
        """
        Callback for output audio processing.

        Args:
            outdata (np.ndarray): Output audio data
            frames (int): Number of frames
            time_info (Dict[str, float]): Timing information
            status (sd.CallbackFlags): Status flags
        """
        if status:
            logger.warning(f"Output stream status: {status}")

        try:
            # Create a copy of the output array that we can modify
            output = np.zeros_like(outdata)

            # Check if we have data in the output queue
            if not self.output_queue.empty():
                # Non-blocking get from queue
                audio_chunk = self.output_queue.get_nowait()
                # Add new chunk to current buffer
                audio_chunk = self.processor.decode(audio_chunk)
                self.current_audio_data = np.concatenate(
                    [self.current_audio_data, audio_chunk]
                )

            # If we have audio data to play
            if len(self.current_audio_data) > 0:
                # Calculate how many samples we can play in this callback
                samples_to_play = min(frames, len(self.current_audio_data))

                # Copy the samples to the output
                if samples_to_play > 0:
                    output[:samples_to_play, 0] = self.current_audio_data[
                        :samples_to_play
                    ]
                    # Remove the played samples from our buffer
                    self.current_audio_data = self.current_audio_data[samples_to_play:]
                    logger.debug(
                        f"Playing {samples_to_play} samples, remaining: {len(self.current_audio_data)}" #pylint: disable=C0301
                    )

                # If we didn't fill the entire output buffer, fill the rest with silence
                if samples_to_play < frames:
                    output[samples_to_play:, 0] = 0
            else:
                # No data available, output silence
                output[:] = 0

            # Call user callback if provided
            if self.output_queue_callback:
                try:
                    self.output_queue_callback(output)
                except Exception as e:
                    logger.error(f"Error in output callback: {e}")

            # Copy our modified output back to the read-only array
            outdata[:] = output

        except Exception as e:
            logger.error(f"Error in output callback: {e}")
            outdata[:] = 0

    def get_input_audio(self) -> Optional[str]:
        """
        Get the next chunk of input audio data.

        Returns:
            Optional[str]: Base64 encoded audio data, or None if no data available
        """
        if self.input_queue.empty():
            return None
        # Non-blocking get from queue
        try:
            audio_data = self.input_queue.get_nowait()
        except Exception as e:
            logger.error(f"Error getting input audio data: {e}")
            return None
        return audio_data

    def add_audio_data(self, audio_data: Union[np.ndarray, bytes, str]) -> None:
        """Add new audio data to the playback buffer."""
        try:
            self.output_queue.put(audio_data)
        except Exception as e:
            logger.error(f"Error adding audio data to output queue: {e}")

    def save_input_audio(self, filename: str = "input.wav") -> None:
        """
        Save the recorded input audio to a WAV file.

        Args:
            filename (str): The name of the WAV file to save to
        """
        if not self.raw_input_chunks:
            logger.warning("No input audio recorded")
            return

        try:
            # Concatenate all chunks
            audio_data = np.concatenate(self.raw_input_chunks)

            # Convert to 16-bit PCM
            pcm16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)

            # Save as WAV
            with wave.open(filename, "wb") as wf:
                # pylint: disable=E1101
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm16.tobytes())

            logger.info(f"Saved input audio to {filename}")

        except Exception as e:
            logger.error(f"Failed to save input audio: {e}")

    def close(self) -> None:
        """Close all audio streams."""
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()

        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()

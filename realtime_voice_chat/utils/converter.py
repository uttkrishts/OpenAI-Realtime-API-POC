"""
Audio processing utilities for converting audio formats.
"""

import audioop

import numpy as np
from scipy.signal import resample_poly


def convert_pcm24k_to_mulaw8k(pcm_data: bytes) -> bytes:
    """
    Convert 16-bit PCM data from 24 kHz to µ-law 8 kHz.

    This function performs the following steps:
    1. Decodes the input bytes into a NumPy array of 16-bit PCM values.
    2. Resamples the audio from 24000 Hz to 8000 Hz using a
    high-quality polyphase filtering approach.
    3. Converts the resampled data back to 16-bit PCM byte format.
    4. Transforms the 16-bit linear PCM data into µ-law encoded audio.

    Parameters:
        pcm_data (bytes): Raw 16-bit PCM audio data sampled at 24 kHz.

    Returns:
        bytes: µ-law encoded audio data sampled at 8 kHz.
    """
    # Step 1: Decode raw 16-bit PCM to numpy
    pcm_np = np.frombuffer(pcm_data, dtype=np.int16)

    # Step 2: Resample from 24000 → 8000 using polyphase (high quality)
    resampled = resample_poly(pcm_np, up=1, down=3)

    # Step 3: Convert to 16-bit PCM again
    pcm_8k = resampled.astype(np.int16).tobytes()

    # Step 4: Convert Linear PCM to µ-law
    mulaw_8k = audioop.lin2ulaw(pcm_8k, 2)  # '2' means 16-bit input

    return mulaw_8k

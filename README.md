# Realtime Voice Chat

A Python library for real-time voice chat using OpenAI's API. This library provides a simple interface for creating voice-based conversational AI applications.

## Features

- Real-time audio input/output using sounddevice
- WebSocket communication with OpenAI's API
- Support for custom AI instructions
- Audio recording and playback
- Easy-to-use interface

## Installation

```bash
pip install -e
```

## Quick Start

```python
from realtime_voice_chat.core import SoundDeviceAudioHandler, OpenAIRealtimeClient

# Initialize components
openai_client = OpenAIRealtimeClient()
audio_handler = SoundDeviceAudioHandler()

# Connect to OpenAI
openai_client.connect()
openai_client.set_instructions("Your AI instructions here")

# Start audio streams
audio_handler.start_input_stream()
audio_handler.start_output_stream()

# Main loop
while True:
    # Get input audio and send to OpenAI
    audio_data = audio_handler.get_input_audio()
    if audio_data is not None:
        openai_client.send_audio(audio_data)
    
    # Get response audio and add to playback buffer
    if not openai_client.audio_queue.empty():
        response_audio = openai_client.audio_queue.get()
        audio_handler.add_audio_data(response_audio)
```

## Examples

Check out the `examples` directory for complete working examples:

- `telephone_operator.py`: A virtual telephone operator that greets callers and handles inquiries

## Requirements

- Python 3.12+

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

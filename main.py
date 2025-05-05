import os
import json
import base64
import threading
import time
import wave
from queue import Queue
import queue

import numpy as np
import sounddevice as sd
import websocket
import dotenv

# Audio playback buffer
current_audio_data = np.array([], dtype=np.float32)  # Current audio data being played
output_stream = None  # Global output stream handle
CHUNK_SIZE = int(24000 * 0.1)  # 0.1 second chunks
playback_started = False  # Flag to track if we've started playback
response_over=False
telephone_operator_instruction = """
You are a virtual telephone operator working for **Company X**. Your primary role is to greet callers, assist with inquiries, and route calls or provide information as needed. Follow these behavior guidelines:

1. **Greeting**: Always begin with a professional and friendly greeting.
   - Example: "Thank you for calling Company X. This is the virtual assistant. How can I help you today?"

2. **Tone and Style**: Be polite, calm, helpful, and efficient. Avoid slang. Respond in a concise and professional tone.

3. **Identify Intent**: Determine the reason for the call by asking clear, polite questions.
   - Examples:  
     - "Are you calling about a product, a service, or something else?"  
     - "Can you please tell me the name of the person or department you're trying to reach?"

4. **Handle Common Requests**: Prepare to answer FAQs such as:
   - Company hours and location
   - Product/service overviews
   - Billing or support contact details

5. **Escalation and Limits**:
   - If a question is outside your knowledge, say:  
     "I'm sorry, I don't have that information, but I'll make sure someone from our team follows up with you."

6. **Closing**: End calls with a courteous farewell.  
   - Example: "Thank you for calling Company X. Have a great day!"

7. **Always let user know uf they are audible or not.**
"""

# Constants
SAMPLE_RATE = 24000  # Hz

# Queues
send_queue = Queue()  # raw float32 blocks from mic to send

# Received audio storage
raw_b64_chunks = []  # list of base64-encoded float32 PCM deltas
raw_input_chunks = []   # collect float32 blocks for saving input

# Stream handles
event_stream = None
mic_stream = None

# ---------- Audio Capture ----------

def float_to_16bit_pcm(float32_array):
    clipped = np.clip(float32_array, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16)

def base64_encode_audio(float32_array):
    # First convert float32 to int16
    pcm16 = float_to_16bit_pcm(float32_array)
    # Then encode the int16 bytes
    raw = pcm16.tobytes()
    return base64.b64encode(raw).decode("ascii")

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    flat = indata[:, 0] if indata.ndim > 1 else indata
    send_queue.put(flat.copy())
    raw_input_chunks.append(flat.copy())

# ---------- WebSocket Handlers ----------

def on_open(ws):
    print("Connected to server.")

def on_message(ws, message):
    global current_audio_data, playback_started
    
    data = json.loads(message)
    if data.get("type") == "response.audio.delta":
        delta = data.get("delta", "")
        if not isinstance(delta, str):
            print(f"Unexpected delta format: {type(delta)}")
            return
            
        raw_b64_chunks.append(delta)  # still keep it for saving
        try:
            # Ensure proper base64 padding
            padding = (4 - len(delta) % 4) % 4
            padded_delta = delta + '=' * padding
            # Decode the audio data
            raw_bytes = base64.b64decode(padded_delta)
            # First convert to int16, then to float32
            audio_data = np.frombuffer(raw_bytes, dtype=np.int16)
            # Convert to float32 in range [-1, 1]
            audio_data = audio_data.astype(np.float32) / 32767.0
            
            print(f"Received audio data of length: {len(audio_data)} samples")
            
            # Concatenate the new audio data to our playing buffer
            current_audio_data = np.concatenate([current_audio_data, audio_data])
            print(f"Total audio length: {len(current_audio_data)} samples")
                
            # Start playback if not already started
            if not playback_started:
                playback_started = True
                print("Starting audio playback")
        except Exception as e:
            print("Failed to process audio delta:", e)

    elif data.get("type") == "response.audio.done":
        global response_over
        response_over=True
        print("Audio stream completed.")
    else:
        print(json.dumps(data, indent=2))

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, code, msg):
    print("Connection closed:", code, msg)

# ---------- Helpers ----------

def save_input_audio(filename="input.wav"):
    if not raw_input_chunks:
        print("No input audio recorded.")
        return
    # concatenate all float32 blocks
    float_arr = np.concatenate(raw_input_chunks)
    # convert to 16‑bit PCM
    pcm16 = float_to_16bit_pcm(float_arr)
    # write WAV
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())
    print(f"Saved input audio to {filename}")

def save_audio(filename="output.wav"):
    if not raw_b64_chunks:
        print("No audio received.")
        return
    combined = "".join(raw_b64_chunks)
    try:
        raw_bytes = base64.b64decode(combined)
    except Exception as e:
        print("Failed to decode combined base64:", e)
        return

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_bytes)
    print(f"Saved received audio to {filename}")

# ---------- Stream Starters ----------

def start_microphone_stream():
    global mic_stream
    mic_stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * 2),
        callback=audio_callback,
    )
    mic_stream.start()
    print("Mic stream started (0.5s blocks).")

def start_output_stream():
    global output_stream
    output_stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
        callback=audio_output_callback,
    )
    output_stream.start()
    print(f"Output stream started with chunk size {CHUNK_SIZE} samples.")

def audio_output_callback(outdata, frames, time, status):
    global current_audio_data
    
    if status:
        print(f"Output stream status: {status}")
    
    try:
        # Create a copy of the output array that we can modify
        output = np.zeros_like(outdata)
        
        # If we have audio data to play
        if len(current_audio_data) > 0:
            # Calculate how many samples we can play in this callback
            samples_to_play = min(frames, len(current_audio_data))
            
            # Copy the samples to the output
            if samples_to_play > 0:
                output[:samples_to_play, 0] = current_audio_data[:samples_to_play]
                # Remove the played samples from our buffer
                current_audio_data = current_audio_data[samples_to_play:]
                print(f"Playing {samples_to_play} samples, remaining: {len(current_audio_data)}")
            
            # If we didn't fill the entire output buffer, fill the rest with silence
            if samples_to_play < frames:
                output[samples_to_play:, 0] = 0
        else:
            # No data available, output silence
            output[:] = 0
        
        # Copy our modified output back to the read-only array
        outdata[:] = output
            
    except Exception as e:
        print(f"Error in audio callback: {e}")
        outdata[:] = 0

# ---------- Main ----------

def main():
    global event_stream
    global response_over
    dotenv.load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set.")
        return

    url = "wss://api.openai.com/v1/realtime" "?model=gpt-4o-realtime-preview-2024-12-17"
    headers = [f"Authorization: Bearer {api_key}", "OpenAI-Beta: realtime=v1"]

    event_stream = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    ws_thread = threading.Thread(target=event_stream.run_forever, daemon=True)
    ws_thread.start()
    time.sleep(2)

    if not (event_stream.sock and event_stream.sock.connected):
        print("WebSocket not connected. Exiting.")
        return
    event_stream.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "instructions": telephone_operator_instruction,
                    "turn_detection": {"type": "server_vad", "threshold": 0.9},
                },
            }
        )
    )
    event_stream.send(
        json.dumps(
            {
                "type": "conversation.item.create",
                "previous_item_id": None,
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Location for X company in Surat: Plot No 357, Gr Flr, Gopal Nagar Society Near Piyus Point Pandesara Surat Gujarat 394221, Surat, 394221"
                        }
                    ]
                }
            }
        )
    )
    start_microphone_stream()
    start_output_stream()

    print("Streaming... press Ctrl+C to stop and save.")

    try:
        while True:
            if not send_queue.empty():
                audio = send_queue.get()
                encoded = base64_encode_audio(audio)
                event_stream.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": encoded,
                        }
                    )
                )
                # event_stream.send(json.dumps({"type": "response.create"}))
            time.sleep(2)
    except KeyboardInterrupt:
        print("Stopping streams and saving...")
    finally:
        if mic_stream:
            mic_stream.stop()
            mic_stream.close()
        if event_stream:
            event_stream.close()
        save_audio()
        save_input_audio("input.wav")
        # if output_stream:
        #     output_stream.stop()
        #     output_stream.close()
        time.sleep(50)

if __name__ == "__main__":
    main()

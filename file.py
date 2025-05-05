import os
import json
import base64
import struct
import soundfile as sf
import websocket
import threading
import time

ws_app_instance = None  # Global or shared ref to the WebSocketApp

def float_to_16bit_pcm(float32_array):
    clipped = [max(-1.0, min(1.0, x)) for x in float32_array]
    pcm16 = b''.join(struct.pack('<h', int(x * 32767)) for x in clipped)
    return pcm16

def base64_encode_audio(float32_array):
    pcm_bytes = float_to_16bit_pcm(float32_array)
    encoded = base64.b64encode(pcm_bytes).decode('ascii')
    return encoded

def send_audio_event(ws):
    # Send event separately
    fullAudio = "/Users/tagline/Documents/openai/input.wav"
    data, samplerate = sf.read(fullAudio, dtype='float32')  
    channel_data = data[:, 0] if data.ndim > 1 else data
    encoded_audio = base64_encode_audio(channel_data)

    event = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "audio": encoded_audio,
                }
            ],
        },
    }
    ws.send(json.dumps(event))
    print("Audio event sent.")

def on_open(ws):
    print("Connected to server.")

def on_message(ws, message):
    print("Message received.")
    data = json.loads(message)
    # with open("data.json", "w") as file:
    print(json.dumps(data, indent=2))

def on_error(ws, error):
    print("Error:", error)

def on_close(ws, code, msg):
    print("Closed:", code, msg)

def main():
    global ws_app_instance
    import dotenv
    dotenv.load_dotenv()

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
    headers = [
        "Authorization: Bearer " + OPENAI_API_KEY,
        "OpenAI-Beta: realtime=v1"
    ]

    ws_app = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    ws_app_instance = ws_app

    # Run in background thread
    def run_ws():
        ws_app.run_forever()

    threading.Thread(target=run_ws, daemon=True).start()

    # Wait until connected (you could add a more robust wait strategy)
    time.sleep(2)

    # Now send the event separately
    ws_app.send(
        json.dumps(
            {
                "type": "session.update",
                "session": {
                    "turn_detection":{"type":"server_vad","threshold":1},
                    "max_response_output_tokens": 100,
                },
            }
        )
    )
    if ws_app.sock and ws_app.sock.connected:
        send_audio_event(ws_app)
        # ws_app.send(json.dumps({"type":"response.create",}))
    else:
        print("WebSocket not connected.")
    time.sleep(100)

if __name__ == "__main__":
    main()

import sounddevice as sd
import numpy as np
import wave
import time

def play_wav_file(filename):
    # Open the WAV file
    with wave.open(filename, 'rb') as wf:
        # Get the audio parameters
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        n_frames = wf.getnframes()
        
        # Read the audio data
        audio_data = wf.readframes(n_frames)
        
        # Convert to numpy array
        if sample_width == 2:  # 16-bit PCM
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        elif sample_width == 4:  # 32-bit PCM
            audio_array = np.frombuffer(audio_data, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        # Convert to float32 in range [-1, 1]
        audio_array = audio_array.astype(np.float32) / (2**(8*sample_width-1))
        
        print(f"Playing {filename}:")
        print(f"Channels: {n_channels}")
        print(f"Sample rate: {frame_rate} Hz")
        print(f"Duration: {n_frames/frame_rate:.2f} seconds")
        print(f"Total samples: {len(audio_array)}")
        
        # Play in chunks of 24000 samples
        chunk_size = 24000
        total_chunks = len(audio_array) // chunk_size
        if len(audio_array) % chunk_size != 0:
            total_chunks += 1
            
        print(f"Playing in {total_chunks} chunks of {chunk_size} samples each")
        
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(audio_array))
            chunk = audio_array[start_idx:end_idx]
            
            print(f"Playing chunk {i+1}/{total_chunks} ({len(chunk)} samples)")
            sd.play(chunk, frame_rate)
            sd.wait()  # Wait until this chunk is finished
            time.sleep(0.1)  # Small pause between chunks
            
        print("Playback complete")

if __name__ == "__main__":
    play_wav_file("output.wav") 
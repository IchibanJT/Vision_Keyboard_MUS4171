import wave
import math
import struct

# Frequencies for a C Major Pentatonic Scale (C4, D4, E4, G4, A4)
tuning_system = {
    'thumb.wav': 261.63, 
    'index.wav': 293.66, 
    'middle.wav': 329.63, 
    'ring.wav': 392.00, 
    'pinky.wav': 440.00
}

sample_rate = 44100.0
duration = 0.4 

for filename, frequency in tuning_system.items():
    wavef = wave.open(filename, 'w')
    wavef.setnchannels(1) 
    wavef.setsampwidth(2) 
    wavef.setframerate(sample_rate)

    for i in range(int(duration * sample_rate)):
        fade = 1.0 - (i / (duration * sample_rate))
        value = int(32767.0 * math.cos(frequency * math.pi * float(i) / float(sample_rate)) * fade)
        data = struct.pack('<h', value)
        wavef.writeframesraw(data)

    wavef.writeframes(b'')
    wavef.close()
    print(f"Created {filename} at {frequency} Hz")
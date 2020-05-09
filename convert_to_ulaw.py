import wave
import audioop

wave_file = wave.open("test.wav", mode='rb')
num_frames = wave_file.getnframes()
linear_audio = wave_file.readframes(num_frames)
ulaw_audio = audioop.lin2ulaw(linear_audio, 2)

with open("ghibli", 'wb') as dataset_file:
  dataset_file.write(ulaw_audio)

'''
wave_file = wave.open("test1.wav", mode='wb')
wave_file.setnchannels(1)
wave_file.setsampwidth(1)
wave_file.setframerate(11025)
wave_file.writeframes(ulaw_audio)
'''

audio_file = open("result.wav", 'r+b')
audio_file.seek(20)
audio_format = audio_file.read(2)
print(audio_format)
audio_file.seek(20)
audio_file.write(b'\x07\x00')
audio_file.close()
#print(b'\x07\x00')

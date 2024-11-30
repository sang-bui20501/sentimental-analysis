import whisperx

device = "cpu"  
audio_file = "./dataset/test-case-2.mp3"
compute_type = "float32"  

model = whisperx.load_model("large-v2", device=device, compute_type=compute_type)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=16, language="vi")  

print("Transcribed Text in Vietnamese:")
for segment in result["segments"]:
    print(segment["text"])

del model

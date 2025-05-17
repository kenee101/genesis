from faster_whisper import WhisperModel # type: ignore

model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cpu", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("Future - Life Is Good (Audio) ft. Drake.mp3", beam_size=5)
segments = list(segments)  # The transcription will actually run here.
print(segments)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
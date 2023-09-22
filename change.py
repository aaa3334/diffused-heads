
from pydub import AudioSegment
import torchaudio


input_file = "/Users/a/Documents/Automations/git talking heads/audio/testaudio.mp3"

# Convert MP3 to WAV and resample to 16kHz
AudioSegment.from_mp3(input_file).export("temp.wav", format="wav")
waveform, sample_rate = torchaudio.load("temp.wav")
resampled_waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
torchaudio.save("output_16kHz.wav", resampled_waveform, 16000)
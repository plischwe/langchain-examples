import argparse
import os
import openvino_genai
import csv
import librosa

parser = argparse.ArgumentParser()
parser.add_argument('--audio_dir', type=str, help='Path containing audio files')
parser.add_argument('--device', type=str, help='Device to run OV whisper on', default="GPU")
parser.add_argument('--language', type=str, default="<|en|>", help='Target language to transcribe to. <|hi|> for Hindi, <|bn|> for Bengali or <|zh|> for Chinese language')
args = parser.parse_args()

def read_wav(filepath):
    raw_speech, samplerate = librosa.load(filepath, sr=16000)
    return raw_speech.tolist()

pipe = openvino_genai.WhisperPipeline('ov_whisper_largev3', args.device)
config = pipe.get_generation_config()
config.language = args.language  # can switch to <|hi|> for Hindi, <|bn|> for Bengali or <|zh|> for Chinese language
config.task = "transcribe"

class_transcriptions = []
audio_extensions = ('.mp3', '.wav')
audio_dir = args.audio_dir
for class_audio in os.listdir(audio_dir):
    if class_audio.lower().endswith(audio_extensions):
        raw_speech = read_wav(os.path.join(audio_dir, class_audio))
        lesson = pipe.generate(raw_speech, config)
        print(f"Transcribed text: {lesson}")
        class_transcriptions.append({'audio_file': f"{class_audio}", 'transcription': lesson})
        print(f"Successfully transcribed {class_audio}")

transcribed_lessons_file = "course_lessons.csv"
with open(transcribed_lessons_file, mode='w', newline='', encoding='utf-8') as file:
    fieldnames = class_transcriptions[0].keys()
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(class_transcriptions)
    print(f"Successfully created inital transcription dataset at {transcribed_lessons_file}")

import os
import csv
import json
import re 
from datasets import Dataset
from youtube_transcript_api import YouTubeTranscriptApi

# Define set of yt videos using video_id
# Using Algebra course playlist from https://www.youtube.com/playlist?list=PLgIi4lM74yW0ChmzTdT1w5ruCnqP0bv3J
video_playlist = {
    'VXzm8ReImG0': 'Pre-Algebra Full Course', #15 hrs
    'GAN-jgzYsIo': 'Algebra 1 Full Course', #27 hrs
    '2wrPGtP61fo': 'Algebra 2 Full Course', #35 hrs
    'wioFZFIDniQ': 'Algebra 1 Practice Full Course | Practice Sets | Practice Test Solutions', #36 hrs
    'n2Zj2NxoyBk': 'Pre-Algebra Practice Full Course | Practice Sets | Practice Test Solutions', #23 hrs,
    'AuixBGkLjfo': 'College Algebra Full Course' #54 hrs
}

# Get transcription of yt video
def get_yt_transcription(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    transcript = [{'start': s['start'], 'text': s['text']} for s in transcript]
   #print(f"First 4 rows of transcribed audio: {transcript[0:4]}")
    return transcript

# Structure transcription into paragraphs for LLM
def get_transcript_as_text(transcript):
    temp_list = [s['text'] for s in transcript]
    transcript_as_text = ' '.join(temp_list)

    return transcript_as_text

# Create a lesson segmenting helper for transcripts
def segment_transcript(transcript):
    # start_patterns = [r"(hello and welcome to )", 
    #                  r"(alright lets take a look )",
    #                  r"(in this lesson )"]
    # pattern = "(" + "|".join(start_patterns) + ")"
    if transcript.startswith('in this lesson'):
        pattern = r"(in this lesson )"
    else:
        pattern = r"(hello and welcome to )"
    segments = re.split(pattern, transcript)

    if len(segments) < 2:
        return [transcript.strip()]

    lessons = []
    for i in range(1, len(segments), 2):
        lesson_title = segments[i].strip()
        lesson_content = segments[i + 1].strip() if i + 1 < len(segments) else ""
        lessons.append(f"{lesson_title}\n{lesson_content}")
    return lessons

# Create a list of dictionaries of video_name and corresponding transcriptions for HF Dataset
video_transcriptions = []
print(video_playlist)
for video_id in video_playlist:
    video_name = video_playlist[video_id]
    transcript = get_yt_transcription(video_id)
    transcript_as_text = get_transcript_as_text(transcript)

    #Here is where you can clean/segment transcripts by lesson
    lessons = segment_transcript(transcript_as_text)
    for i, lesson in enumerate(lessons, 1):
         video_transcriptions.append({'video_name': f"{video_name}_lesson_{i}", 'transcription': lesson})

    print(f"Successfully transcribed {video_name}")

transcribed_lessons_file = "math_course_lessons.csv"

with open(transcribed_lessons_file, mode='w', newline='', encoding='utf-8') as file:
    fieldnames = video_transcriptions[0].keys()
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(video_transcriptions)

print(f"Transcribed lessons file created at {transcribed_lessons_file}")

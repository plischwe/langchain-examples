import pandas as pd
import requests
import time
from openai import OpenAI

df = pd.read_csv("course_lessons.csv")
if "lesson_plan" not in df.columns:
    df["lesson_plan"] = ""

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123"
)

def generate_lesson_plan(transcription):
    prompt = """You are an intelligent AI assistant who is an expert at making lesson plans for elementary school math teachers. Given the following 
    trancription of a class session, you are tasked to generate a structured, concise lesson plans that comprehensively cover the material presented in the transcription 
    while strictly focusing on the substantive content covered by the teacher in the lesson.
    """
    content = f"""Please create a lesson plan for the following lesson transcription. The lesson plan should include:
    - Learning objectives & goals
    - Key concepts or formulas introduces.
    - Steps or proceedures taught.
    - Example problems or problem types discussed.
    - Summary of what the students should understand by the end.
    Do not include and behvioral tips or any introduction/conclusion to your response - focus only on the course content.
    Here is the lesson transcription: {transcription} \n"""

    chat_response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ],
    )
    response = chat_response.choices[0].message.content
    print(f"Llama3.3-70B lesson plan: {response}")
    return response

for idx, row in df.iterrows():
    if pd.notna(row["lesson_plan"]) and row["lesson_plan"].strip() != "":
        continue

    print(f"Processing row {idx+1}/{len(df)}: {row['video_name']}")
    plan = generate_lesson_plan(row['transcription'])
    df.at[idx, "lesson_plan"] = plan
    df.to_csv("lessons_with_plans.csv", index=False)

    print(f"Finished labelling row {idx+1}")
    time.sleep(1)

import argparse
import openvino_genai
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('device')
    args = parser.parse_args()

    adapter = openvino_genai.Adapter('outputs/adapter_model.safetensors')
    adapter_config = openvino_genai.AdapterConfig(adapter)
    pipe = openvino_genai.LLMPipeline('llama3.1-8b-Instruct-INT4', args.device, adapters=adapter_config)  # register all required adapters here

    query = f"""Create a lesson plan given the following lesson transcript. Please be extremely concise and 
            only include information relevant to the transcription and the structure below. Please format and include:
            - Learning concepts
            - Key concepts and any formulas introduced
            - Steps or procedures taught
            - Example problems or problem types discussed
            - Summary of what the students should understand by end
            Here is the lesson transcript to make the lesson plan from:
            \n {transcript_1}"""

    print("Generate with LoRA adapter and alpha set to 0.75:")
    print(pipe.generate(query, max_new_tokens=1000, adapters=openvino_genai.AdapterConfig(adapter, 0.75)))

    print("\n-----------------------------")
    print("Generate without LoRA adapter:")
    print(pipe.generate(query, max_new_tokens=1000, adapters=openvino_genai.AdapterConfig()))

if '__main__' == __name__:
    main()

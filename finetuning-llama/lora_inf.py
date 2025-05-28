import argparse
import openvino_genai
import csv
import time
from langchain_community.embeddings import OpenVINOEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
import docs_loader_utils as docs_loader
import numpy as np
from sklearn.cluster import KMeans
from langchain_core.documents import Document

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, default='GPU')
    parser.add_argument('-text', '--transcription')
    parser.add_argument('-llm', '--llm_path', type=str, default='llama3.1-8b-Instruct-INT4', help="Path to quantized OV Llama model")
    parser.add_argument('-lora', '--lora_path', type=str, default='outputs/adapter_model.safetensors', help="Path to LoRA adapter weights from finetuning")
    args = parser.parse_args()

    device = args.device
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": device}
    encode_kwargs = {"mean_pooling": True, "normalize_embeddings": True}

    ov_embeddings = OpenVINOEmbeddings(
        model_name_or_path=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    with open(args.transcription, "r") as file:
        content = file.read()
    words = ' '.join(content.split())
    word_count = len(words)
    print(f"Length of Class Transcription: {len(words.split())} words")

    text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=0)

    docs = text_splitter.split_text(content)

    # K-means logic below
    # 768 dimensional
    vectors = ov_embeddings.embed_documents([doc for doc in docs])
    print(f"Length of docs: {len(docs)}")
    print(f"Number of vector embeddings: {len(vectors)}")
    num_clusters = 6 # TODO: adjust for smaller size text docs
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)

    closest_indices = []

    # Loop through the number of clusters you have, selecting top clusters
    for i in range(num_clusters):
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)
        closest_index = np.argmin(distances)
        closest_indices.append(closest_index)

    selected_indices = sorted(closest_indices)
    last_idx_sel = selected_indices[len(selected_indices)-1]
    last_idx_docs = len(docs) - 1
    docs = [docs[doc] for doc in selected_indices]

    # Note: this single reduced batch method for large text is only supported 
    # due to limited GPU/OpenCL memory allocation limit
    docs = [Document(page_content=docs_loader.combine_docs(docs))]

    for i in range(0, len(docs)):
        doc_in = [docs[i]]
        text = docs_loader.format_docs(doc_in)

        query = f"""As an intelligent AI teacher assistant, please create a lesson plan from this teacher's transcribed lesson from class. Please be extremely concise and 
            only include information relevant to the transcription and the structure below. Please format your response as the below example:\n
            **Learning concepts**
		- Here are the higher level concepts introduced
            **Key concepts and any formulas introduced**
		- Here is how the concepts are related to formulas/equations introduced
            **Steps or procedures taught**
		- Here are steps/proceedures/approaches for solving/breaking down the above concepts
            **Example problems or problem types discussed**
            	- Be explicit in grounding the concepts to problems
            **Summary of what the students should understand by end**
		- Summary of the goals students should achieve when done
            Here is the transcribed lesson from class to construct the above lesson plan from: {text}\n Answer: """

        adapter = openvino_genai.Adapter(args.lora_path)
        adapter_config = openvino_genai.AdapterConfig(adapter)

        pipe = openvino_genai.LLMPipeline(args.llm_path, device, adapters=adapter_config)  # register all required adapters here

        print("Generate with LoRA adapter and alpha set to 0.75:")
        lora_start = time.time()
        lesson_plan = pipe.generate(query, max_new_tokens=750, adapters=openvino_genai.AdapterConfig(adapter, 0.75))
        lora_end = time.time()
        print(f"Lesson Plan: {lesson_plan}")
        print(f"Time for lora generation: {lora_end - lora_start} seconds")

        print("\n-----------------------------")
        print("Generate without LoRA adapter:")

        non_lora_start = time.time()
        print(pipe.generate(query, max_new_tokens=750, adapters=openvino_genai.AdapterConfig()))
        non_lora_end = time.time()
        print(f"Time for non-lora generation: {non_lora_end - non_lora_start}")


if '__main__' == __name__:
    main()

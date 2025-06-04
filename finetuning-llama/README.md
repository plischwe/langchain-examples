# How to finetune an LLM on a personal dataset

Below are steps to create a training dataset from a set of Youtube videos. In this example, a directory of audio file, or algebra course videos are used to represent typical class lessons. The first set of steps involves finetuning an INT4 weight-compressed Llama3.1-8B model using the transcriptions of the class sessions to create a labelled, Supervised FineTuning dataset that will be used in the jupyter notebook to train the model. The second set of instructions details how to inference this finetuned model on an Intel Core platform.

### Prerequisites (Intel Core Platform)
**Dependencies**- Python >= 3.10 - [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux)/[miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#unix-like-platforms-macos-linux--wsl)
Create new environment, activate it, and install necessary packages for class audio transcription.
Firstly, convert whisper model to OV for optimized transcription
```
conda create -n transcribe python=3.10
conda activate transcribe
pip install -r tr_requirements.txt
optimum-cli export openvino --model openai/whisper-large-v3 ov_whisper_largev3
```

Transcribe directory of audio files. Each audio file should correspond to one class session. These will be the transcriptions used in bootstrapping the training data. By default, GPU is used, and english is the selected language. You can change both of those with `--langauge` (<|hi|> for hindi, <|bn|> for bengali, <|zh|> for chinese) and `--device`.
```
python transcribe_audio.py --audio_dir <path_to_audio_files> --language '<|en|>'
```

Note: If you do not have audio files, you can transcribe youtube videos to create your training dataset
Transcribe a set of youtube videos from youtube [playlist](https://www.youtube.com/watch?v=VXzm8ReImG0&list=PLgIi4lM74yW0ChmzTdT1w5ruCnqP0bv3J&index=2)
```
python transcribe.py
```

### Part 1: Finetune Model (NVIDIA hardware)
Move resulting `course_lessons.csv` file to the current directory you are working in, on the NVIDIA machine.

**Dependencies**- Python >= 3.10 - [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux)/[miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#unix-like-platforms-macos-linux--wsl)
Create new environment, activate it, and install necessary packages for finetuning
```
conda create -n bootstrap python=3.10
conda activate bootstrap
pip install -r ft_requirements.txt
```

2) Setup a VLLM server running online Llama3.3-70B
```
VLLM_SKIP_WARMUP=true vllm serve meta-llama/Llama-3.3-70B-Instruct --task generate --trust-remote-code --tensor-parallel 4 --max_model_len 16384 --port 8000
```
Note: You can scale `--tensor-parallel` with the amount of GPUs that are accessible. So if you have 4 cards, you can utilize all with `--tensor-parallel 4`

3) In another terminal window while the vLLM Llama server is running, create a labelled dataset from your transcribed algebra lessons
```
conda activate bootstrap
python bootstrap_lessons.py
```
Note: Ensure `course_lessons.csv` is in the same directory as `bootstrap_lessons.py`.

4) Once done labelling data, it is time to use it to finetune Llama. Now launch and step through the unsloth QLoRA finetuning notebook
```
jupyter lab unsloth_llama3_8B_SFT.ipynb
```

Note: The resulting LoRA adapter weights are stored in the `outputs/checkpoint-#` directory (mulitple training runs can result in multiple checkpoints). This path is used to load LoRA weights during inference.

## Part 2: Inference the offline finetuned model (Intel Core Platform)

5) Move to Intel Core Platform machine, clone this repo, and move the `outputs/checkpoint-#` directory containing LoRA adapters to this working directory
--> Path to LoRA adapters should look like `.../finetuning-llama/outputs/checkpoint-#/adapter_model.safetensors`

6) Get access to gated [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model from HF using an [access token](https://huggingface.co/docs/hub/en/security-tokens)

7) Get started by running the below command to install necessary drivers/packages.

```
./install.sh
```

Note: if this script has already been performed and you'd like to re-install the sample project only then the below command can be used to skip the re-install of dependencies.

```
./install.sh --skip
```

8) Run inference on the example math lesson transcription `class_transcription.txt`, specifying the path to LoRA weights and device you'd like to inference on (default is GPU.0 for iGPU) 
```
./run-demo.sh -tr class_transcription.txt -lora <path_to_adapter_model.safetensors> -d <device>
```

Note: You will see both the LoRA generated response as well as the non-LoRA generated response in the output.

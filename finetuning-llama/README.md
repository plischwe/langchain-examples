# How to finetune an LLM on a personal dataset

## Below are steps to create a training dataset from a set of Youtube videos. In this example, algebra course videos are used to represent typical math lessons.

### Prerequisite
Install necessary packages
```
pip install -r requirements.txt
```

1) Transcribe a set of youtube videos from youtube [playlist](https://www.youtube.com/watch?v=VXzm8ReImG0&list=PLgIi4lM74yW0ChmzTdT1w5ruCnqP0bv3J&index=2)
```
python transcribe.py
```

2) Setup a VLLM server running online Llama3.3-70B in another terminal window
```
VLLM_SKIP_WARMUP=true vllm serve meta-llama/Llama-3.3-70B-Instruct --task generate --trust-remote-code --tensor-parallel 4 --max_model_len 16384
```

3) Create a labelled dataset from your transcribed algebra lessons
```
python bootstrap_lessons.py
```

4) Launch unsloth QLoRA finetuning script on the resulting labelled data
```
jupyter lab unsloth_llama3_8B_SFT.ipynb
```

--> The resulting LoRA adapter weights are stored in the `outputs` directory. This path is used to load LoRA weights during inference.

5) Move to Intel Core Platform machine, clone the repo and move the `outputs` directory containing LoRA adapters to the `finetuning-llama` directory to begin inference
```
git clone https://github.com/plischwe/langchain-examples.git
git checkout finetuning-llama
cd finetuning-llama
```

6) Install necessary packages for inference:
```
pip install -r inf_requirements.txt
```

7) Get access to gated [Llama3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model from HF using an [access token](https://huggingface.co/docs/hub/en/security-tokens) and use optimum-cli to convert/quantize model
```
optimum-cli export openvino -m meta-llama/Meta-Llama-3.1-8B-Instruct --trust-remote-code --weight-format int4 llama3.1-8b-Instruct-INT4
```

8) Run inference on example lesson transcription 
```
python lora_test.py --device <CPU/GPU/NPU>
```
